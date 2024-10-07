#include "pathtrace.h"
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define TOGGLE_BVH 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
static PathSegment* dev_final_paths = NULL;
static BVH_Main_Data* bvh_main_data = NULL;
static Mesh_Data* mesh_data = NULL;
static glm::vec4* pixel_data = NULL;
static Texture_Data* texture_data = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    dev_final_paths = dev_paths;

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&bvh_main_data, scene->bvh_main_datas.size() * sizeof(BVH_Main_Data));
    cudaMemcpy(bvh_main_data, scene->bvh_main_datas.data(), scene->bvh_main_datas.size() * sizeof(BVH_Main_Data), cudaMemcpyHostToDevice);

    cudaMalloc(&mesh_data, scene->m_data.size() * sizeof(Mesh_Data));
    cudaMemcpy(mesh_data, scene->m_data.data(), scene->m_data.size() * sizeof(Mesh_Data), cudaMemcpyHostToDevice);

    cudaMalloc(&pixel_data, scene->pixels.size() * sizeof(glm::vec4));
    cudaMemcpy(pixel_data, scene->pixels.data(), scene->pixels.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

    cudaMalloc(&texture_data, scene->m_textures.size() * sizeof(Texture_Data));
    cudaMemcpy(texture_data, scene->m_textures.data(), scene->m_textures.size() * sizeof(Texture_Data), cudaMemcpyHostToDevice);


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    // TODO: clean up any extra device memory you created
    cudaFree(bvh_main_data);
    cudaFree(mesh_data);
    cudaFree(texture_data);
    cudaFree(pixel_data);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // DONE: implement antialiasing by jittering the ray
        thrust::random::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::random::uniform_real_distribution<float> u05(-0.5f, 0.5f);
        float dx = u05(rng);
        float dy = u05(rng);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + dx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + dy - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}
__device__ glm::vec4 sampleTexture(Texture_Data texture, glm::vec4* pixels, glm::vec2 uvCoord)
{
    // Map UV coordinates to texture resolution
    const float xPos = glm::fract(uvCoord.x) * texture.width;
    const float yPos = glm::fract(1.0f - uvCoord.y) * texture.height;

    // Get integer pixel indices (floor and ceiling)
    const float xLower = glm::floor(xPos);
    const float yLower = glm::floor(yPos);
    const float xUpper = glm::ceil(xPos);
    const float yUpper = glm::ceil(yPos);

    // Compute fractional offsets for interpolation
    const float xFraction = xPos - xLower;
    const float yFraction = yPos - yLower;

    // Get the four neighboring pixels
    const glm::vec4 lowerLeftPixel = pixels[texture.index + static_cast<int>(yLower) * texture.width + static_cast<int>(xLower)];
    const glm::vec4 upperLeftPixel = pixels[texture.index + static_cast<int>(yUpper) * texture.width + static_cast<int>(xLower)];
    const glm::vec4 lowerRightPixel = pixels[texture.index + static_cast<int>(yLower) * texture.width + static_cast<int>(xUpper)];
    const glm::vec4 upperRightPixel = pixels[texture.index + static_cast<int>(yUpper) * texture.width + static_cast<int>(xUpper)];

    // Perform bilinear interpolation between the four pixels
    glm::vec4 leftInterp = glm::mix(lowerLeftPixel, upperLeftPixel, yFraction);
    glm::vec4 rightInterp = glm::mix(lowerRightPixel, upperRightPixel, yFraction);

    return glm::mix(leftInterp, rightInterp, xFraction);
}

__device__ void updateIntersectionData(glm::vec3 w, Mesh_Data* mesh_data, int idx, glm::vec3& n, glm::vec3& t, glm::vec2& uv)
{
    n = mesh_data[idx].normal * w.x;
    n += mesh_data[idx + 1].normal * w.y;
    n += mesh_data[idx + 2].normal * w.z;
    t = mesh_data[idx].tangent * w.x;
    t += mesh_data[idx + 1].tangent * w.y;
    t += mesh_data[idx + 2].tangent * w.z;
    uv = mesh_data[idx].coordinate * w.x;
    uv += mesh_data[idx + 1].coordinate * w.y;
    uv += mesh_data[idx + 2].coordinate * w.z;
}

__global__ void m_computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    int vertices_size,
    Mesh_Data* mesh_data,
    int num_bvh,
    BVH_Main_Data* bvh_main_data,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths) {
        PathSegment pathSegment = pathSegments[path_index];
        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec2 uvCoord;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        int material = -1;
        int hit_mesh_index = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        
        // iter through global geoms
        for (int i = 0; i < geoms_size; ++i)
        {
            // current geometry
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                material = geom.materialid;
                t_min = t;
                //hit_geom_index = i;
                //intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }
        
#if TOGGLE_BVH
        glm::vec3 origin = pathSegment.ray.origin;
        int curr_bvh_index = (num_bvh == 0) ? -1 : 0;
        int prev_bvh_index = -1;
        int count = 0;
        int traversed_indices[32];
        while (curr_bvh_index != -1) {
            BVH_Main_Data bvh = bvh_main_data[curr_bvh_index];
            // check previous index
            if (prev_bvh_index == bvh.child_indices[1]) {
                if (count == 0) break;
                else {
                    count--;
                    prev_bvh_index = curr_bvh_index;
                    curr_bvh_index = traversed_indices[count];
                }
            }
            else if (prev_bvh_index == bvh.child_indices[0]) {
                traversed_indices[count++] = curr_bvh_index;
                prev_bvh_index = curr_bvh_index;
                curr_bvh_index = bvh.child_indices[1];
            }
            else {
                // precompute vectors and distance
                glm::vec3 v = bvh.center - origin;
                float f = glm::dot(v, pathSegment.ray.direction);
                glm::vec3 point = origin + pathSegment.ray.direction * f;
                float dist = glm::distance(point, bvh.center) - bvh.radius;
                if (dist <= 0.0f && bvh.count > 0) {
                    for (int i = bvh.index; i < bvh.index + bvh.count * 3; i += 3) {
                        glm::vec3 points[3] = {
                            mesh_data[i + 0].point,
                            mesh_data[i + 1].point,
                            mesh_data[i + 2].point
                        };
                        t = meshIntersectionTest(pathSegment.ray, points);
                        if (t > 0.0f && t_min > t) {
                            hit_mesh_index = i;
                            t_min = t;
                        }
                    }
                }
                if (dist > 0.0f || bvh.count > 0) {
                    if (count == 0) break;
                    else {
                        count--;
                        prev_bvh_index = curr_bvh_index;
                        curr_bvh_index = traversed_indices[count];
                    }
                }
                else {
                    traversed_indices[count++] = curr_bvh_index;
                    prev_bvh_index = curr_bvh_index;
                    curr_bvh_index = bvh.child_indices[0];
                }
            }
        }
#else
        // mesh intersection test
        for (int i = 0; i < vertices_size; i += 3) {
            glm::vec3 points[3] = {
                mesh_data[i + 0].point,
                mesh_data[i + 1].point,
                mesh_data[i + 2].point
            };
            t = meshIntersectionTest(pathSegment.ray, points);
            if (t > 0.0f && t_min > t)
            {
                hit_mesh_index = i;
                t_min = t;
            }
        }
#endif
        if (hit_mesh_index != -1) {
            material = mesh_data[hit_mesh_index].material;
            glm::vec3 points[3] = {
                mesh_data[hit_mesh_index + 0].point,
                mesh_data[hit_mesh_index + 1].point,
                mesh_data[hit_mesh_index + 2].point
            };
            glm::vec3 w = barycentricWeightCompute(
                pathSegment.ray.origin + pathSegment.ray.direction * t_min,
                points
            );
            updateIntersectionData(w, mesh_data, hit_mesh_index, normal, tangent, uvCoord);
        }

        if (material == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = material;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].tangent = tangent;
            intersections[path_index].uvCoord = uvCoord;
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture_Data* textures,
    glm::vec4* pixels
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0) // if the intersection exists...
    {
        // Set up the RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;
        glm::vec3 surfaceNormal = intersection.surfaceNormal;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0) {
            pathSegments[idx].color *= (materialColor * material.emittance);
            pathSegments[idx].remainingBounces = 0;
        }
        else {
            // check if material has albedo
            if (material.albedo >= 0) {
                glm::vec4 p = sampleTexture(textures[material.albedo], pixels, intersection.uvCoord);
                materialColor = glm::vec3(p.x, p.y, p.z);
                // gamma correction
                materialColor = glm::pow(materialColor, glm::vec3(1.0 / 2.2));
            }
            // check if material has normal
            if (material.normal >= 0) {
                glm::vec4 p = sampleTexture(textures[material.normal], pixels, intersection.uvCoord);
                glm::vec3 tangent = intersection.tangent;
                glm::mat3 M = glm::mat3{ 
                    glm::normalize(tangent), 
                    glm::normalize(glm::cross(intersection.surfaceNormal, tangent)), 
                    glm::normalize(intersection.surfaceNormal)
                };
                glm::vec3 V = glm::vec3(p.x, p.y, p.z);
                surfaceNormal = M * V;
                surfaceNormal = glm::normalize(surfaceNormal);
            }
            glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
            scatterRay(intersection, pathSegments[idx], intersect, materialColor, surfaceNormal, material, rng);
            pathSegments[idx].remainingBounces--;
        }
    }
    // nothing exist
    else {
        pathSegments[idx].color = glm::vec3(0.0);
        pathSegments[idx].remainingBounces = 0;
    }
}

__global__ void shadeBasicBSDF(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0) // if the intersection exists...
    {
        // Set up the RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0) {
            pathSegments[idx].color *= (materialColor * material.emittance);
            pathSegments[idx].remainingBounces = 0;
        }
        else {
            glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
            kernBasicScatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
            pathSegments[idx].remainingBounces--;
        }
    }
    else {
        pathSegments[idx].color = glm::vec3(0.0);
        pathSegments[idx].remainingBounces = 0;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}



/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int N = pixelcount;
    int num_paths = N;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, N * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        // compute mesh intersection
        m_computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            hst_scene->m_data.size(),
            mesh_data,
            hst_scene->bvh_main_datas.size(),
            bvh_main_data,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;
        
        // TODO:
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            texture_data,
            pixel_data
            );
        // stream compaction
        dev_paths = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, checkPathComplete());
        num_paths = dev_path_end - dev_paths;
        if (num_paths == 0) iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(N, dev_image, dev_final_paths);
    // reset
    dev_paths = dev_final_paths;
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
