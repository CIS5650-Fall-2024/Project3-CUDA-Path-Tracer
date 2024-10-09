#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "OpenImageDenoise/oidn.hpp"

#define ERRORCHECK 1
#define COMPACT_PATHS 1
#define SORT_MATERIAL 0
#define RUSSIAN_ROULETTE 1
#define BVH 1
#define DENOISE 1
#define DENOISE_ON_INTERVAL 0
#define DENOISE_INTERVAL 10
#define DOF 0

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
static BVHNode* dev_nodes = NULL;
static int* dev_indices = NULL;
static glm::vec3* dev_textures = NULL;
static glm::vec3* dev_oidn_image = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static glm::vec3* dev_envMap = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_nodes, scene->nodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_nodes, scene->nodes.data(), scene->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_indices, scene->indices.size() * sizeof(int));
    cudaMemcpy(dev_indices, scene->indices.data(), scene->indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    //textures
    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

#if DENOISE
    //OIDN
    cudaMalloc(&dev_oidn_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_oidn_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
#endif

    //Envmap
    if (scene->env.size() > 0)
    {
        cudaMalloc(&dev_envMap, scene->env.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_envMap, scene->env.data(), scene->env.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_nodes);
    cudaFree(dev_indices);
    cudaFree(dev_textures);
    cudaFree(dev_oidn_image);
    cudaFree(dev_albedo);
    cudaFree(dev_normal);
    cudaFree(dev_envMap);
    // TODO: clean up any extra device memory you created

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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float randX = u01(rng);
        float randY = u01(rng);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + randX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + randY - (float)cam.resolution.y * 0.5f)
        );

#if DOF
        float r = glm::sqrt(u01(rng));
        float theta = 2.f * PI * u01(rng);
        glm::vec2 sample = cam.apertureRadius * glm::vec2(r * glm::cos(theta), r * glm::sin(theta));
        glm::vec3 lensPoint = cam.position + sample.x * cam.right + sample.y * cam.up;
        glm::vec3 focalPoint = segment.ray.direction * cam.focalLength + cam.position;

        segment.ray.origin = lensPoint;
        segment.ray.direction = glm::normalize(focalPoint - lensPoint);
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.hitLight = false;
    }
}

__device__ void traverseBVH(
    int path_index,
    Ray& r,
    BVHNode* nodes,
    int* indices,
    Geom* geoms,
    ShadeableIntersection* intersections)
{
    int stack[64] = { 0 }; //should be enough
    int idx = 0;

    int nodeIdx = 0;

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    glm::vec2 uv;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_uv;

    while (nodeIdx >= 0)
    {
        BVHNode& node = nodes[nodeIdx];
        //printf("Node index: %d, Left child: %d, Right child: %d, Start index: %d, Num primitives: %d\n",
            //nodeIdx, node.leftChild, node.rightChild, node.startIndex, node.numPrimitives);
        if (AABBIntersectionTest(node, r)) {
            if (node.numPrimitives > 0) {
                for (int i = 0; i < node.numPrimitives; ++i)
                {
                    int primitiveIdx = indices[node.startIndex + i];

                    Geom& geom = geoms[primitiveIdx];

                    if (geom.type == CUBE)
                    {
                        t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
                    }
                    else if (geom.type == SPHERE)
                    {
                        t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
                    }
                    else if (geom.type == TRIANGLE)
                    {
                        t = triangleIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside);
                    }
                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        hit_geom_index = primitiveIdx;
                        intersect_point = tmp_intersect;
                        normal = tmp_normal;
                        uv = tmp_uv;
                    }
                }
                //printf("stack idx: %d \n", idx);
                if (idx > 0) {
                    nodeIdx = stack[--idx];  // Pop from stack
                }
                else {
                    //printf("I've breaked");
                    nodeIdx = -1;
                    break;  // If the stack is empty, end the traversal
                }
            }
            else {
                stack[idx++] = node.rightChild;
                nodeIdx = node.leftChild;
            }
        }
        else {
            if (idx > 0) {
                nodeIdx = stack[--idx];
            }
            else {
                nodeIdx = -1;
                break;
            }
        }
    }

    //if (ever)
    //printf("out of while loop");

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        // The ray hits something
        intersections[path_index].t = t_min;
        //printf("matrial id: %d\n", geoms[hit_geom_index].materialid);
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].uv = uv;
        intersections[path_index].surfaceNormal = normal;
        //printf("Path_Index: %d, Normal (%f, %f, %f)\n", path_index, normal.x, normal.y, normal.z);
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    BVHNode* nodes,
    int* indices,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("path_index: %d", path_index);

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

#if BVH
        traverseBVH(path_index, pathSegment.ray, nodes, indices, geoms, intersections);
#else
        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                //printf("primitiveIdx: %d\n", i);
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].uv = uv;
            intersections[path_index].surfaceNormal = normal;
        }
#endif
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* textures,
    glm::vec3* dev_albedo,
    glm::vec3* dev_normal,
    glm::vec3* dev_envMap,
    int env_width,
    int env_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (pathSegment.remainingBounces <= 0) return;
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                //printf("hit light");
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegment.remainingBounces = 0;
                pathSegment.hitLight = true;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 intersect = getPointOnRay(pathSegment.ray, intersection.t);
                if (material.diffuseMap.index != -1)
                {
                    int x = (int)(intersection.uv.x * material.diffuseMap.width);
                    int y = (int)((1.f - intersection.uv.y) * material.diffuseMap.height);
                    int idx = material.diffuseMap.startIdx + material.diffuseMap.width * y + x;
                    Material copyMaterial = Material();
                    copyMaterial.color = textures[idx];
                    copyMaterial.specular.color = material.specular.color;
                    copyMaterial.microfacet.isMicrofacet = material.microfacet.isMicrofacet;
                    copyMaterial.hasReflective = material.hasReflective;
                    copyMaterial.hasRefractive = material.hasRefractive;
                    copyMaterial.indexOfRefraction = material.indexOfRefraction;
                    copyMaterial.microfacet.roughness = material.microfacet.roughness;
                    material = copyMaterial;
                }
                scatterRay(pathSegment, intersect, intersection.surfaceNormal, material, rng);
#if RUSSIAN_ROULETTE
                glm::vec3 color = pathSegments[idx].color;
                //printf("Color (%f, %f, %f)\n", color.x, color.y, color.z);
                float prob = fmaxf(color.x, fmaxf(color.y, color.z));
                float rand = u01(rng);
                if (pathSegment.remainingBounces > 1 && rand < prob) {
                    pathSegment.color *= 1.f / prob;
                    pathSegment.remainingBounces--;
                }
                else {
                    pathSegment.remainingBounces = 0;
                }
#else
                if (pathSegment.remainingBounces > 1) {
                    pathSegment.remainingBounces--;
                }
#endif
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            if (dev_envMap != NULL) {
                float u = 0.5f + (atan2(-pathSegment.ray.direction.z, -pathSegment.ray.direction.x) / (2.f * PI));
                float v = 0.5f - (asin(-pathSegment.ray.direction.y) / PI);
                int x = (int)(u * env_width);
                int y = (int)((1.f - v) * env_height);
                int index = env_width * y + x;
                glm::vec3 color = dev_envMap[index];
                //This prevents firefly to some extent, basic sampling results in too much firefly
                float maxCol = fmaxf(fmaxf(color.x, color.y), color.z);
                if (maxCol > 4.f)
                {
                    color = color / maxCol;
                }
                pathSegments[idx].color *= color;
                pathSegments[idx].remainingBounces = 0;
                pathSegment.hitLight = true;
            }
            else {
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0;
            }
        }
        dev_albedo[pathSegments[idx].pixelIndex] = pathSegments[idx].color;
        dev_normal[pathSegments[idx].pixelIndex] = intersection.surfaceNormal;
    }
    //printf("remaining bounces: %d", pathSegments[idx].remainingBounces);
}

/* Reference: https://github.com/RenderKit/oidn */
void OIDN_Denoise() {
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    const int width = hst_scene->state.camera.resolution.x;
    const int height = hst_scene->state.camera.resolution.y;

    oidn::FilterRef filter = device.newFilter("RT"); 
    filter.setImage("color", dev_image, oidn::Format::Float3, width, height);
    filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    filter.setImage("output", dev_oidn_image, oidn::Format::Float3, width, height);
    filter.set("hdr", true);
    filter.set("cleanAux", true);
    filter.commit();

    oidn::FilterRef albedoFilter = device.newFilter("RT");
    albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    oidn::FilterRef normalFilter = device.newFilter("RT");
    normalFilter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.setImage("output", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.commit();

    albedoFilter.execute();
    normalFilter.execute();
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "Error! " << errorMessage << std::endl;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.hitLight){
            if (isnan(iterationPath.color.x) || isnan(iterationPath.color.y) || isnan(iterationPath.color.z)) {
                return;
            }
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
    }
}

struct isPathActive
{
    __host__ __device__
        bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

struct sortMaterialID
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2)
    {
        return s1.materialId < s2.materialId;
    }
};

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

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        //BVHNode* copy = (BVHNode*)malloc(hst_scene->nodes.size() * sizeof(BVHNode));
        //cudaMemcpy(copy, dev_nodes, hst_scene->nodes.size() * sizeof(BVHNode), cudaMemcpyDeviceToHost);
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            dev_nodes,
            dev_indices,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIAL
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortMaterialID());
        cudaDeviceSynchronize();
#endif
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_albedo,
            dev_normal,
            dev_envMap,
            hst_scene->env_width,
            hst_scene->env_height
        );
        cudaDeviceSynchronize();

#if COMPACT_PATHS
        // compact paths
        //std::cout << "numPath: " << num_paths << std::endl;
        thrust::device_ptr<PathSegment> dev_ptr(dev_paths);
        thrust::device_ptr<PathSegment> dev_ptr_end = thrust::stable_partition(thrust::device, dev_ptr, dev_ptr + num_paths, isPathActive());
        cudaDeviceSynchronize();
        num_paths = dev_ptr_end - dev_ptr;
        //std::cout << "numPath: " << num_paths << std::endl;
#endif

        iterationComplete = (num_paths == 0) || (depth == traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    //num_paths = dev_path_end - dev_paths;

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

#if DENOISE
#if DENOISE_ON_INTERVAL
    if (iter % DENOISE_INTERVAL == 0 || iter == hst_scene->state.iterations) {
        OIDN_Denoise();
    }
    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_oidn_image);
#else
    if (iter == hst_scene->state.iterations) {
        OIDN_Denoise();
        std::swap(dev_image, dev_oidn_image);
    }
    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
#endif
#else 
    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
#endif
    //Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}
