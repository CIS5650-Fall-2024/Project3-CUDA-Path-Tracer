#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

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
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhnodes = NULL;
static glm::vec4* dev_tex_data = NULL;

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

    if (scene->meshes.size() > 0) {
        int num_tris = scene->triangle_count;
        cudaMalloc(&dev_triangles, num_tris * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->mesh_triangles.data(), num_tris * sizeof(Triangle), cudaMemcpyHostToDevice);

        int num_bvhnodes = scene->bvhNodes.size();
        cudaMalloc(&dev_bvhnodes, num_bvhnodes * sizeof(BVHNode));
        cudaMemcpy(dev_bvhnodes, scene->bvhNodes.data(), num_bvhnodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }

    if (scene->textures.size() > 0) {
        int num_colors = scene->textures.at(0).color_data.size();
        cudaMalloc(&dev_tex_data, num_colors * sizeof(glm::vec4));
        cudaMemcpy(dev_tex_data, scene->textures.at(0).color_data.data(), num_colors * sizeof(glm::vec4), cudaMemcpyHostToDevice);
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
    cudaFree(dev_triangles);
    cudaFree(dev_bvhnodes);
    cudaFree(dev_tex_data);

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

        // antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__device__ glm::vec3 barycentricCoordinates(const glm::vec3& P,
                                            const glm::vec3& A,
                                            const glm::vec3& B,
                                            const glm::vec3& C) {
    glm::vec3 v0 = B - A;
    glm::vec3 v1 = C - A;
    glm::vec3 v2 = P - A;

    float d00 = glm::dot(v0, v0);
    float d01 = glm::dot(v0, v1);
    float d11 = glm::dot(v1, v1);
    float d20 = glm::dot(v2, v0);
    float d21 = glm::dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return glm::vec3(u, v, w);
}

__device__ glm::vec2 interpolateUV(const glm::vec3& P,
    const glm::vec3& A, const glm::vec2& UV_A,
    const glm::vec3& B, const glm::vec2& UV_B,
    const glm::vec3& C, const glm::vec2& UV_C) {
    glm::vec3 bary = barycentricCoordinates(P, A, B, C);

    return bary.x * UV_A + bary.y * UV_B + bary.z * UV_C;
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
    int geoms_size,
    ShadeableIntersection* intersections,
    Triangle* tris,
    int num_tris,
    BVHNode* bvhnodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        bool iter_hit_mesh = false, overall_hit_mesh = false;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                iter_hit_mesh = false;
            }
            else if (geom.type == SPHERE)
            { 
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                intersections->outside = outside;
                iter_hit_mesh = false;
            }
            else if (geom.type == TRIANGLE) {
                //t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                iter_hit_mesh = false;
            }
            else if (geom.type == MESH) {
                Triangle tri_hit;
#define USE_BVH 1
#if USE_BVH
                t = bvhIntersectionTest(pathSegment.ray, tmp_intersect, tmp_normal, outside, bvhnodes, tris, num_tris, tri_hit);

#else
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, tris, num_tris, tri_hit);
#endif

                glm::vec2 uv = interpolateUV(tmp_intersect, tri_hit.v0.pos, tri_hit.v0.uv,
                    tri_hit.v1.pos, tri_hit.v1.uv, tri_hit.v2.pos, tri_hit.v2.uv);

                intersections->uv = uv;
                iter_hit_mesh = true;
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                overall_hit_mesh = iter_hit_mesh ? true : false;
            }
        }

        intersections[path_index].outside = outside;

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].hitMesh = overall_hit_mesh ? true : false;
        }
    }
}

__device__ glm::vec3 fresnelDielectricEval(float etaI, float etaT, float cosThetaI) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    bool entering = cosThetaI > 0.f;
    if (!entering) {
        //swap etaI and etaT
        float temp = etaI;
        etaI = etaT;
        etaT = temp;

        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2.f);
}

__device__ glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
    float cos_theta = std::fmin(dot(-uv, n), 1.f);
    glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    glm::vec3 r_out_parallel = -glm::sqrt(glm::abs(1.f - glm::length2(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

__device__ float cosTheta(glm::vec3 v1, glm::vec3 v2) {
    return glm::cos(glm::acos(glm::dot(v1, v2)));
}

__global__ void shadeMaterials(int iter,
                               int num_paths,
                               int depth,
                               ShadeableIntersection* shadeableIntersections,
                               PathSegment* pathSegments,
                               Material* materials,
                               glm::vec4* texture_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_paths || pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) // if the intersection exists...
    {
        // Set up the RNG
        // LOOK: this is how you use thrust's RNG! Please look at
        // makeSeededRandomEngine as well.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor;
        if (!material.isTexture || !intersection.hitMesh) {
            materialColor = material.color;
        }
        else {
            glm::vec2 uv = intersection.uv;
            //pass width and height in, mult u * width, v * height
            int tex_x_idx = uv.x * 64; //scale from 0..1 to 0..63
            int tex_y_idx = uv.y * 64; //scale from 0..1 to 0..63 
            int tex_1d_idx = tex_x_idx + tex_y_idx * 64;
            materialColor = glm::vec3(texture_data[tex_1d_idx]);
        }
        PathSegment& curr_seg = pathSegments[idx];
        Ray& curr_ray = curr_seg.ray;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0f) {
            curr_seg.color *= (materialColor * material.emittance);
            curr_seg.remainingBounces = 0;
        } 
        else  if (material.specular_transmissive.isSpecular == false) {
            //perfectly diffuse for now
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            glm::vec3 wi;
            scatterRay(curr_seg, isect_pt, intersection.surfaceNormal, material, rng, wi);

            wi = glm::normalize(wi);

            float costheta = cosTheta(wi, intersection.surfaceNormal);
            float pdf = costheta * INV_PI;
            if (pdf == 0.f) {
                curr_seg.remainingBounces = 0;
                return;
            }

            glm::vec3 bsdf = materialColor * INV_PI;
            float lambert = glm::abs(glm::dot(wi, intersection.surfaceNormal));

            curr_seg.color *= (bsdf * lambert) / pdf;

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        else if (material.specular_transmissive.isSpecular == true && material.specular_transmissive.isTransmissive == false) {
            //perfectly specular
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            glm::vec3 wi = glm::reflect(curr_ray.direction, intersection.surfaceNormal);

            wi = glm::normalize(wi);

            //took out lambert and INV_PI from bsdf
            glm::vec3 bsdf = materialColor;
            float lambert = glm::abs(glm::dot(wi, intersection.surfaceNormal));

            curr_seg.color *= (bsdf); //pdf = 1

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        else if (material.specular_transmissive.isSpecular == true && material.specular_transmissive.isTransmissive == true) {
            
            glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 isect_pt = glm::normalize(curr_ray.direction) * intersection.t + curr_ray.origin;

            float rand_num = u01(rng);

            glm::vec3 wi, bsdf;

            float etaA = material.specular_transmissive.eta.x;
            float etaB = material.specular_transmissive.eta.y;

            float costheta = cosTheta(curr_ray.direction, intersection.surfaceNormal);
            bool entering = intersection.outside;
            float etaI = entering ? etaA : etaB;
            float etaT = entering ? etaB : etaA;
            float eta = etaI / etaT;

            bool reflected = false;

            wi = refract(curr_ray.direction, intersection.surfaceNormal, etaI / etaT);

            float cosThetaI = dot(intersection.surfaceNormal, wi);
            float sin2ThetaI = max(0.f, 1.f - cosThetaI * cosThetaI);
            float sin2ThetaT = eta * eta * sin2ThetaI;

            if (rand_num < 0.5f || sin2ThetaI >= 1.f) {
                //using specular reflection
                reflected = true;
                wi = glm::reflect(curr_ray.direction, intersection.surfaceNormal);
                bsdf = materialColor;
            }
            else {

                //using specular refraction
                glm::vec3 T = materialColor / glm::abs(cosTheta(wi, intersection.surfaceNormal));
                bsdf = (glm::vec3(1.) - fresnelDielectricEval(etaI, etaT, glm::dot(nor, normalize(wi)))) * T;

                bsdf *= glm::abs(glm::dot(wi, intersection.surfaceNormal));
                
            }

            curr_seg.color *= (bsdf); //pdf = 1

            glm::vec3 new_dir = wi;
            glm::vec3 new_origin;
            if (reflected) {
                new_origin = isect_pt + intersection.surfaceNormal * 0.01f;
            }
            else {
                new_origin = isect_pt - intersection.surfaceNormal * 0.01f;
            }
            curr_seg.ray.origin = new_origin;
            curr_seg.ray.direction = new_dir;
            curr_seg.remainingBounces--;
        }
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
    }
    else {
        pathSegments[idx].color = glm::vec3(0.0f);
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

struct ShouldTerminate {
    __host__ __device__ bool operator()(const PathSegment& x)
    {
        return x.remainingBounces > 0;
    }
};

struct CompareMaterials
{
    __host__ __device__ bool operator()(const ShadeableIntersection& first, const ShadeableIntersection& second)
    {
        return first.materialId < second.materialId;
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
    int num_paths = dev_path_end - dev_paths; //just the pixel count for now

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_triangles,
            hst_scene->triangle_count,
            dev_bvhnodes
        );
        checkCUDAError("compute intersections");
        cudaDeviceSynchronize();
        depth++;

#define SORTBYMATERIAL 1
#if SORTBYMATERIAL
        thrust::device_ptr<ShadeableIntersection> dev_inters_to_sort(dev_intersections);
        thrust::device_ptr<PathSegment> dev_paths_to_sort(dev_paths); //values
        thrust::stable_sort_by_key(dev_inters_to_sort, dev_inters_to_sort + num_paths, dev_paths_to_sort, CompareMaterials());
#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_tex_data
            );

#define USE_COMPACTION 1
#if USE_COMPACTION
        thrust::device_ptr<PathSegment> dev_paths_to_compact(dev_paths);
        thrust::device_ptr<PathSegment> last_elt = thrust::stable_partition(thrust::device, dev_paths_to_compact, dev_paths_to_compact + num_paths, ShouldTerminate());
        num_paths = last_elt.get() - dev_paths;
#endif

        iterationComplete = (depth >= traceDepth || num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //NOTE: changed N to pixelcount from num paths, still want to check paths that will be terminated
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
