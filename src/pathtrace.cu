#include "pathtrace.h"
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "device_launch_parameters.h"

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

//  Set seed for thrust random number generator
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Thrust sort materials comparator
struct CompInterMats {
    __host__ __device__ bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2) {
        return s1.materialId <= s2.materialId;
    }
};

// Thrust compact terminated rays conditioner
struct CheckRayBounce {
    __host__ __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
    }
};

// Kernel that writes the image to the OpenGL PBO directly
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

// Device variables
// Render components
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;

// Scene components
static Geom* dev_geoms = NULL;
static Primitive* dev_primitives = NULL;
static BVHNode* dev_bvhNodes = NULL;
static BSDF* dev_materials = NULL;
static Light* dev_lights = NULL;

// Compute components
static glm::vec3* dev_image = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    // Render
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // Scene components
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_primitives, scene->prims.size() * sizeof(Primitive));
    cudaMemcpy(dev_primitives, scene->prims.data(), scene->prims.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    
#if BVH
    cudaMalloc(&dev_bvhNodes, scene->bvh.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvh.data(), scene->bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
#endif

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(BSDF));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(BSDF), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    // Compute components
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_geoms);
    cudaFree(dev_primitives);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_materials);
    cudaFree(dev_lights);
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_intersections);

    checkCUDAError("pathtraceFree");
}

// Concentric disk Sampling on thin lens
__device__ glm::vec2 concentricDiskSampling(thrust::default_random_engine rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u1 = u01(rng);
    float u2 = u01(rng);

    float r, theta;
    //  Center of the disk
    if (u1 == 0 && u2 == 0) {
        return glm::vec2(0, 0);
    }

    // Map square to disk
    if (abs(u1) > abs(u2)) {
        r = u1;
        theta = (PI / 4) * (u2 / u1);
    }
    else {
        r = u2;
        theta = (PI / 2) - (PI / 4) * (u1 / u2);
    }

    // Convert to Cartesian coordinates
    return glm::vec2(r * cos(theta), r * sin(theta));
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
        segment.ray.tmin = 0.0f;
        segment.ray.tmax = 1e38f;
        segment.L_out = glm::vec3(0.0f);
        segment.beta = glm::vec3(1.0f);

        // Stochastic Antialiasing
#if StochasticAntialiasing
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> uh(-0.5, 0.5);
        float jitterX = uh(rng);
        float jitterY = uh(rng);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

#if DOF
        glm::vec2 lensSample = cam.aperture * concentricDiskSampling(rng);
        glm::vec3 lensPoint = cam.position + lensSample.x * cam.right + lensSample.y * cam.up;
        glm::vec3 focalPoint = cam.position + segment.ray.direction * cam.focal;
        glm::vec3 newDir = glm::normalize(focalPoint - lensPoint);
        segment.ray.origin = lensPoint;
        segment.ray.direction = newDir;
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    Primitive* prims,
    int prims_size,
    ShadeableIntersection* intersections)
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

        // naive parse through global geoms

        for (int i = 0; i < prims_size; i++)
        {
            Primitive& prim = prims[i];

            if (prim.type == CUBEP)
            {
                t = boxIntersectionTest(geoms[prim.geomId], pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (prim.type == SPHEREP)
            {
                t = sphereIntersectionTest(geoms[prim.geomId], pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (prim.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geoms[prim.geomId], prim, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else
            {
                t = -1;
            }

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
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
            intersections[path_index].materialId = geoms[prims[hit_geom_index].geomId].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__global__ void computeIntersectionBVH(
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    Primitive* prims,
    ShadeableIntersection* intersections,
    BVHNode* bvh)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment& pathSegment = pathSegments[path_index];
        ShadeableIntersection& intersection = intersections[path_index];
        intersectBVH(pathSegment.ray, intersection, geoms, prims, bvh);
    }
}

__global__ void oneBounceRadiance(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    BSDF* materials,
    int num_lights,
    int num_samples,
    Light* lights,
    Geom* geoms,
    Primitive* prims,
    BVHNode* bvh) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        
        ShadeableIntersection& intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];

        if (intersection.t > 0.0f) {

            BSDF material = materials[intersection.materialId];

            if (material.getType() == EMISSION) {
                pathSegment.L_out += pathSegment.beta * material.getEmission();
                pathSegment.remainingBounces = 0;
            }
            else 
            {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
                // Build local coorindate system
                glm::mat3 o2w;
                makeCoordTrans(o2w, intersection.surfaceNormal);
                glm::mat3 w2o = glm::transpose(o2w);

                glm::vec3 hit = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                glm::vec3 wo = w2o * (-pathSegment.ray.direction);

#if NEE
                if (!material.isDelta())
                {
                    // Direct lighting
                    glm::vec3 directLightingSp = glm::vec3(0.0f);
                    glm::vec3 spL, wiL, w_in;
                    float pdfL;

                    for (int i = 0; i < num_lights; i++)
                    {
                        Light& light = lights[i];
                        if (light.isDelta())
                        {
                            num_samples = 1;
                        }

                        for (int j = 0; j < num_samples; j++)
                        {
                            spL = light.sampleL(hit, wiL, &pdfL, rng);
                            w_in = w2o * wiL;
                            if (w_in.z < 0)
                            {
                                continue;
                            }

                            Ray shadowRay{ hit + EPSILON * wiL, wiL };
                            ShadeableIntersection nextIntersection{ -1.0f, glm::vec3(0.0f), -1, -1 };
                            intersectBVH(shadowRay, nextIntersection, geoms, prims, bvh);
                            if ((nextIntersection.t - intersection.t) < EPSILON)
                            {
                                directLightingSp += spL * absCosThetaUnit(w_in) * material.f(wo, w_in) / (num_samples * pdfL);
                            }
                        }
                    }

                    pathSegment.L_out += pathSegment.beta * directLightingSp;
                }
#endif

                glm::vec3 wi;
                float pdf;

                glm::vec3 sp = material.sampleF(wo, wi, &pdf, rng);
                
                glm::vec3 w_world = glm::normalize(o2w * wi);
                pathSegment.beta *= (sp * absCosThetaUnit(wi) / pdf);
                pathSegment.ray.origin = EPSILON * w_world + hit;
                pathSegment.ray.direction = w_world;
                pathSegment.ray.tmax = 1e38f;
#if RussianRoulette
                float continueP = glm::clamp(glm::max(glm::max(pathSegment.beta.x, pathSegment.beta.y), pathSegment.beta.z), 0.0f, 1.0f);
                thrust::uniform_real_distribution<float> u01(0, 1);
                float terminateP = u01(rng);
                if (terminateP > continueP) 
                {
                    pathSegment.remainingBounces = 0;
                }
                else 
                {
                    pathSegment.beta /= continueP;
                }
#else
                pathSegment.remainingBounces -= 1;
#endif
            }
        }
        else {
            pathSegment.remainingBounces = 0;
        }

    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.L_out;
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

    // Generate jittered rays from the camera into the screen
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // Ready to traverse
    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    bool iterationComplete = false;

    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if BVH
        computeIntersectionBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths,
            dev_paths,
            dev_geoms,
            dev_primitives,
            dev_intersections,
            dev_bvhNodes
            );
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            num_paths,
            dev_paths,
            dev_geoms,
            dev_primitives,
            hst_scene->prims.size(),
            dev_intersections
        );
#endif

        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // sort by intersection materials
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, CompInterMats());

        // naive shading
        oneBounceRadiance<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            hst_scene->lights.size(),
            hst_scene->state.camera.sample,
            dev_lights,
            dev_geoms,
            dev_primitives,
            dev_bvhNodes
        );
        checkCUDAError("one bounce shading");
        cudaDeviceSynchronize();

        // compact terminated rays
        PathSegment* path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, CheckRayBounce());
        num_paths = path_end - dev_paths;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
        depth++;

        if (num_paths == 0 || traceDepth == depth) iterationComplete = true;
    }

    // Recover total number path segments
    num_paths = dev_path_end - dev_paths;
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
