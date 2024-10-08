#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "postprocess.h"
#include "bxdf.h"

#define ERRORCHECK 1

#define PARTITION_PATHS_BY_TERMINATION 0
#define REMOVE_TERMINATED_PATHS 1
#define ACTIVE_PATH_ARRANGEMENT_METHOD PARTITION_PATHS_BY_TERMINATION
#if ACTIVE_PATH_ARRANGEMENT_METHOD == PARTITION_PATHS_BY_TERMINATION
    typedef PathSegment PathSegmentT;
#else // ACTIVE_PATH_ARRANGEMENT_METHOD REMOVE_TERMINATED_PATHS
    typedef PathSegment* PathSegmentT;
#endif

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

        pix /= iter;

        pix = postprocess::ACESToneMapping(pix);

        pix = postprocess::gammaCorrection(pix, /*gamma=*/2.2f);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

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
static PathSegment** dev_paths_ptrs = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static glm::vec3* dev_meshes_positions;
static uint16_t* dev_meshes_indices;
static glm::vec3* dev_meshes_normals;
static glm::vec2* dev_meshes_uvs;
static BVHNode* topLevelBVHGPU;
// __device__ glm::vec3* dev_symbol_meshes_positions;
// __device__ uint16_t* dev_symbol_meshes_indices;
// __device__ glm::vec3* dev_symbol_meshes_normals;
// __device__ glm::vec2* dev_symbol_meshes_uvs;

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
    cudaMalloc(&dev_paths_ptrs, pixelcount * sizeof(PathSegment*));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_meshes_positions, scene->meshesPositions.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_meshes_positions, scene->meshesPositions.data(), scene->meshesPositions.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes_indices, scene->meshesIndices.size() * sizeof(uint16_t));
    cudaMemcpy(dev_meshes_indices, scene->meshesIndices.data(), scene->meshesIndices.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes_normals, scene->meshesNormals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_meshes_normals, scene->meshesNormals.data(), scene->meshesNormals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes_uvs, scene->meshesUVs.size() * sizeof(glm::vec2));
    cudaMemcpy(dev_meshes_uvs, scene->meshesUVs.data(), scene->meshesUVs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    // for (auto& geom : scene->geoms) {
    //   if (geom.type != MESH) continue;
    //     cudaMalloc(&geom.triangleIndicesGPU, geom.triangleCount * sizeof(int));
    //     cudaMemcpy(geom.triangleIndicesGPU, geom.triangleIndices, geom.triangleCount * sizeof(int), cudaMemcpyHostToDevice);

    //     cudaMalloc(&geom.meshBVHGPU, geom.meshBVHCount * sizeof(BVHNode));
    //     cudaMemcpy(geom.meshBVHGPU, geom.meshBVH, geom.meshBVHCount * sizeof(BVHNode), cudaMemcpyHostToDevice);
    // }

    // cudaMalloc(&topLevelBVHGPU, scene->topLevelBVHCount * sizeof(BVHNode));
    // cudaMemcpy(topLevelBVHGPU, scene->topLevelBVH, scene->topLevelBVHCount * sizeof(BVHNode), cudaMemcpyHostToDevice);


    // cudaMemcpyToSymbol(dev_symbol_meshes_positions, &dev_meshes_positions, sizeof(glm::vec3*));
    // cudaMemcpyToSymbol(dev_symbol_meshes_indices, &dev_meshes_indices, sizeof(uint16_t*));
    // cudaMemcpyToSymbol(dev_symbol_meshes_normals, &dev_meshes_normals, sizeof(glm::vec3*));
    // cudaMemcpyToSymbol(dev_symbol_meshes_uvs, &dev_meshes_uvs, sizeof(glm::vec2*));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_paths_ptrs);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_meshes_positions);
    cudaFree(dev_meshes_indices);
    cudaFree(dev_meshes_normals);
    cudaFree(dev_meshes_uvs);

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

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        float pixelCenterX = (float)x + jitterX;
        float pixelCenterY = (float)y + jitterY;

        glm::vec3 pixelDirection = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (pixelCenterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (pixelCenterY - (float)cam.resolution.y * 0.5f));

        float lensRadius = cam.aperture / 2.0f;
        float lensU = u01(rng) * 2.0f * PI;
        float lensR = sqrtf(u01(rng)) * lensRadius;
        glm::vec3 lensOffset = cam.right * lensR * cosf(lensU) + cam.up * lensR * sinf(lensU);

        float focusDist = cam.focusDistance;
        glm::vec3 focusPoint = cam.position + pixelDirection * focusDist;

        segment.ray.origin = cam.position + lensOffset;

        segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// __global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
// {
//     int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//     int y = (blockIdx.y * blockDim.y) + threadIdx.y;

//     if (x < cam.resolution.x && y < cam.resolution.y) {
//         int index = x + (y * cam.resolution.x);
//         PathSegment& segment = pathSegments[index];

//         segment.ray.origin = cam.position;
//         segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

//         // TODO: implement antialiasing by jittering the ray
//         segment.ray.direction = glm::normalize(cam.view
//             - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
//             - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
//         );

//         segment.pixelIndex = index;
//         segment.remainingBounces = traceDepth;
//     }
// }

template <typename PathSegmentT>
__device__ inline PathSegment* getPathSegment(PathSegmentT* pathSegments, int idx);

template <>
__device__ inline PathSegment* getPathSegment(PathSegment* pathSegments, int index) {
    return &pathSegments[index];
}

template <>
__device__ inline PathSegment* getPathSegment(PathSegment** pathSegments, int index) {
    return pathSegments[index];
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
template <typename PathSegmentT>
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegmentT* pathSegments,
    Geom* geoms,
    int geoms_size,
    glm::vec3* meshes_positions,
    uint16_t* meshes_indices,
    glm::vec3* meshes_normals,
    glm::vec2* meshes_uvs,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment* pathSegment = getPathSegment(pathSegments, path_index);

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment->ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment->ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(
                    geom,
                    pathSegment->ray,
                    meshes_positions,
                    meshes_indices,
                    meshes_normals,
                    meshes_uvs,
                    tmp_intersect,
                    tmp_normal,
                    outside
                );
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        // intersectBVH(
        //     topLevelBVH,
        //     pathSegment->ray,
        //     0,
        //     geoms,
        //     meshes_positions,
        //     meshes_indices,
        //     meshes_normals,
        //     hit_geom_index,
        //     tmp_intersect,
        //     tmp_normal,
        //     outside,
        //     t_min
        // );

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
            pathSegment->materialType = INVALID;
            pathSegment->materialId = -1;
        }
        else
        {
            intersections[path_index].t = t_min;
            int materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].materialId = materialId;
            intersections[path_index].surfaceNormal = normal;
            pathSegment->materialId = materialId;
            pathSegment->materialType = geoms[hit_geom_index].matType;
        }
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
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
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
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

template <typename PathSegmentT>
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegmentT* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment *pathSegment = getPathSegment(pathSegments, idx);
        ShadeableIntersection intersection = shadeableIntersections[idx];

        if (intersection.t > 0.0f) {
            Material material = materials[intersection.materialId];

            if (material.emittance > 0.0f) {
                pathSegment->color *= (material.color * material.emittance);
                pathSegment->remainingBounces = 0;
            } else {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment->remainingBounces);
                glm::vec3 hitPoint = getPointOnRay(pathSegment->ray, intersection.t);
                glm::vec3 normal = intersection.surfaceNormal;
                scatterRay(*pathSegment, hitPoint, normal, material, rng);
                pathSegment->remainingBounces--;
            }
        } else {
            pathSegment->color = glm::vec3(0.0f);
            pathSegment->remainingBounces = 0;
        }
    }
}

__global__ void shadeDiffuseRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        Material& material = materials[intersection.materialId]; // Access the material from the intersection

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
        glm::vec3 hitPoint = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= material.color;
        pathSegment.ray.origin = hitPoint + 0.1f * normal;

        pathSegment.remainingBounces--;
    }
}

__global__ void shadeEmissiveRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        Material& material = materials[intersection.materialId];

        pathSegment.color *= (material.color * material.emittance);
        pathSegment.remainingBounces = 0;
    }
}

__global__ void shadeSpecularRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        Material& material = materials[intersection.materialId];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
        glm::vec3 hitPoint = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        SpecularBRDF(pathSegment, material, hitPoint, normal);

        pathSegment.remainingBounces--;
    }
}

__global__ void shadeDielectricRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        Material& material = materials[intersection.materialId];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
        glm::vec3 hitPoint = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        DielectricBxDF(pathSegment, material, hitPoint, normal, rng);

        pathSegment.remainingBounces--;
    }
}

__global__ void shadeGGXRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        Material& material = materials[intersection.materialId];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
        glm::vec3 hitPoint = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        glm::vec3 H = sampleGGXNormal(normal, material.roughness, rng);

        glm::vec3 incomingRay = pathSegment.ray.direction;
        glm::vec3 reflectedRay = glm::reflect(incomingRay, H);

        glm::vec3 brdfValue = GGXBRDF(hitPoint, normal, incomingRay, reflectedRay, material);

        pathSegment.ray.direction = reflectedRay;
        pathSegment.color *= brdfValue;
        pathSegment.ray.origin = hitPoint + 0.1f * reflectedRay;

        pathSegment.remainingBounces--;
    }
}

__global__ void shadeSkinRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        Material& material = materials[intersection.materialId];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
        glm::vec3 hitPoint = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        glm::vec3 diffuseDir = calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 subsurfaceNormal = normal + material.subsurfaceScattering * calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 finalDirection = glm::normalize(glm::mix(diffuseDir, subsurfaceNormal, material.subsurfaceScattering));
        pathSegment.ray.direction = finalDirection;
        pathSegment.color *= material.color;
        pathSegment.ray.origin = hitPoint + 0.1f * finalDirection;

        pathSegment.remainingBounces--;
    }
}

__global__ void shadeInvalidRays(PathSegment* pathSegments, int numRays, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        PathSegment& pathSegment = pathSegments[idx];
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
    }
}

#define BLOCK_SIZE 128

__global__ void extractMaterialTypes(PathSegment* pathSegments, int* materialTypes, int numPaths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        materialTypes[idx] = pathSegments[idx].materialType;
    }
}

__global__ void extractMaterialIds(PathSegment* pathSegments, int* materialIds, int numPaths) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numPaths) {
    materialIds[idx] = pathSegments[idx].materialId;
  }
}

void shadeRaysByMaterial(PathSegment* dev_pathSegments, int numPaths, Material* materials, ShadeableIntersection* shadeableIntersections, int iter) {
    int* dev_materialIds;
    cudaMalloc(&dev_materialIds, numPaths * sizeof(int));
    extractMaterialIds<<<(numPaths + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dev_pathSegments, dev_materialIds, numPaths);
    cudaDeviceSynchronize();

    int* hostMaterialIds = new int[numPaths];
    cudaMemcpy(hostMaterialIds, dev_materialIds, numPaths * sizeof(int), cudaMemcpyDeviceToHost);

    int start = 0;

    while (start < numPaths) {
        int currentMaterialId = hostMaterialIds[start];
        int end = start;

        while (end < numPaths && hostMaterialIds[end] == currentMaterialId) {
            end++;
        }

        int numRays = end - start;
        int blocksPerGrid = (numRays + BLOCK_SIZE - 1) / BLOCK_SIZE;

        MatType currentMaterialType;
        if (currentMaterialId == -1) currentMaterialType = INVALID;
        else cudaMemcpy(&currentMaterialType, &materials[currentMaterialId].type, sizeof(MatType), cudaMemcpyDeviceToHost);

        switch (currentMaterialType) {
            case DIFFUSE:
                shadeDiffuseRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
            case SPECULAR:
                shadeSpecularRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
            case DIELECTRIC:
                shadeDielectricRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
            case GGX:
                shadeGGXRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
            case SKIN:
                shadeSkinRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
            case LIGHT:
                shadeEmissiveRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
            default:
                shadeInvalidRays<<<blocksPerGrid, BLOCK_SIZE>>>(
                    dev_pathSegments + start,
                    numRays,
                    materials,
                    shadeableIntersections + start,
                    iter
                );
                break;
        }

        start = end;
    }

    cudaFree(dev_materialIds);
    delete[] hostMaterialIds;
}

struct TerminatedPathChecker {
    __device__ bool operator()(const PathSegment* path) {
        return path->remainingBounces <= 0;
    }

    __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces <= 0;
    }
};

struct ActivePathChecker {
    __device__ bool operator()(const PathSegment* path) {
        return path->remainingBounces > 0;
    }

    __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
    }
};

struct PathSegmentPointerGenerator {
    PathSegment* base;

    __host__ PathSegmentPointerGenerator(PathSegment* base) : base(base) {}

    __device__ PathSegment* operator()(int idx) const {
        return &base[idx];
    }
};

struct RaySorter {
    __device__ bool operator()(const PathSegment &a, const PathSegment &b) {
        return a.materialId < b.materialId;
    }
};

PathSegmentT* initArrangedPaths(PathSegment* dev_paths, int pixelcount) {
#if ACTIVE_PATH_ARRANGEMENT_METHOD == PARTITION_PATHS_BY_TERMINATION
    return dev_paths;
#else // ACTIVE_PATH_ARRANGEMENT_METHOD REMOVE_TERMINATED_PATHS
    PathSegmentPointerGenerator segmentPtrGenerator(dev_paths);
    thrust::transform(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(pixelcount),
        dev_paths_ptrs,
        segmentPtrGenerator
    );
    return dev_paths_ptrs;
#endif
}

template <typename PathSegmentT>
PathSegmentT* arrangePathsByTermination(PathSegmentT* dev_paths, int num_paths);

template <>
PathSegment** arrangePathsByTermination(PathSegment** dev_paths, int num_paths) {
    auto dev_terminated_paths = thrust::remove_if(
        thrust::device,
        dev_paths,
        dev_paths + num_paths,
        TerminatedPathChecker()
    );

    return dev_terminated_paths;
}

PathSegment* arrangePathsByTermination(PathSegment* dev_paths, int num_paths) {
    auto dev_terminated_paths = thrust::stable_partition(
        thrust::device,
        dev_paths,
        dev_paths + num_paths,
        ActivePathChecker()
    );

    return dev_terminated_paths;
}

void sortRaysByMaterial(PathSegment* pathSegments, int numPaths) {
    thrust::device_ptr<PathSegment> dev_ptr(pathSegments);
    
    thrust::sort(dev_ptr, dev_ptr + numPaths, RaySorter());
}

void sortRaysAndIntersectionsByMaterial(PathSegment* pathSegments, ShadeableIntersection* intersections, int numPaths) {
    thrust::device_ptr<PathSegment> dev_pathSegments(pathSegments);
    thrust::device_ptr<ShadeableIntersection> dev_intersections(intersections);

    thrust::sort_by_key(
        dev_pathSegments,
        dev_pathSegments + numPaths,
        dev_intersections,
        RaySorter()
    );
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, bool doMaterialSorting)
{
    doMaterialSorting = true;

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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    PathSegmentT* dev_arranged_paths = initArrangedPaths(dev_paths, pixelcount);

    int depth = 0;

    int num_paths = pixelcount;

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
            dev_arranged_paths,
            // topLevelBVHGPU,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_meshes_positions,
            dev_meshes_indices,
            dev_meshes_normals,
            dev_meshes_uvs,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        #if ACTIVE_PATH_ARRANGEMENT_METHOD == PARTITION_PATHS_BY_TERMINATION
        if (doMaterialSorting) {
            sortRaysAndIntersectionsByMaterial(dev_arranged_paths, dev_intersections, num_paths);
            shadeRaysByMaterial(dev_arranged_paths, num_paths, dev_materials, dev_intersections, iter);
        }
        else 
        #endif
        {
            shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
                iter,
                num_paths,
                dev_intersections,
                dev_arranged_paths,
                dev_materials
            );
        }

        auto dev_terminated_paths = arrangePathsByTermination(dev_arranged_paths, num_paths);

        num_paths = dev_terminated_paths - dev_arranged_paths;

        iterationComplete = num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
