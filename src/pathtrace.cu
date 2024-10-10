#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

// Performance Improvements
#define USE_STREAM_COMPACTION 1
#define USE_MATERIAL_SORT 1
#define USE_RUSSIAN_ROULETTE 1
// #define USE_BVH 0

// Visual Improvements
#define USE_ANTIALIASING 1
#define USE_CHECKERBOARD_TEXTURE 0 // This is the basic procedural texture

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;

static Material* dev_materials = NULL;
static int numAlbedoTextures = 0;
static int numNormalTextures = 0;
static int numBumpTextures = 0;
static Texture* dev_albedoTextures = NULL;
static Texture* dev_normalTextures = NULL;
static Texture* dev_bumpTextures = NULL;
static Geom* dev_geoms = NULL;
static Geom* dev_lights = NULL;
static Triangle* dev_geomTriangles = NULL;
static Triangle* dev_lightTriangles = NULL;
static int* dev_totalNumberOfLights = NULL;

static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

struct sortMaterialCondition
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2)
    {
        return s1.materials.materialId < s2.materials.materialId;
    }
};

struct has_remaining_bounces
{
    __host__ __device__
        bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

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

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void initialiseTriangles(Triangle* dev_triangles, std::vector<Geom>& geometries, const int totalNumberOfGeom)
{   
    if (totalNumberOfGeom == 0) {
        return;
    }

    int totalNumberOfTriangles = 0;
    for (int i = 0; i < totalNumberOfGeom; i++) {
        if (geometries[i].type == MESH) {
            totalNumberOfTriangles += geometries[i].numTriangles;
        }
    }

    if (totalNumberOfTriangles == 0) {
        return;
    }

    cudaMalloc(&dev_triangles, totalNumberOfTriangles * sizeof(Triangle));
    int offset = 0;
    for (int i = 0; i < totalNumberOfGeom; i++) {
        if (geometries[i].type == MESH) {
            // Copy each geometry's triangles to the device memory
            cudaMemcpy(dev_triangles + offset, geometries[i].triangles, geometries[i].numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
            
            // Update the device pointer in the geometry struct to point to device memory
            geometries[i].devTriangles = dev_triangles + offset;
            
            // Move the offset by the number of triangles in this geometry
            offset += geometries[i].numTriangles;
        }
    }
}

// void copyBvhTrianglesFromHostToDevice(Geom &curGeometry) {
//     int numOfBvhTriangles = curGeometry.numTriangles;
//     if (numOfBvhTriangles <= 0) {
//         // printf("You have a mesh with 0 triangles.\n");
//         return;
//     }

//     if (curGeometry.bvhTriangles == nullptr) {
//         // printf("BVH triangles are nullptr.\n");
//         return;
//     }

//     printf("Copying %d BVH triangles to device memory\n", numOfBvhTriangles);


//     cudaMalloc(&curGeometry.devBvhTriangles, numOfBvhTriangles * sizeof(Triangle));
//     cudaMemcpy(curGeometry.devBvhTriangles, curGeometry.bvhTriangles, numOfBvhTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
//     delete[] curGeometry.bvhTriangles;
//     curGeometry.bvhTriangles = nullptr;
// }

// void copyBvhNodesFromHostToDevice(Geom &curGeometry) {
//     int numOfBvhNodes = curGeometry.numBvhNodes;
//     if (numOfBvhNodes <= 0) {
//         printf("You have a mesh with 0 BVH nodes.\n");
//         return;
//     }

//     if (curGeometry.bvhNodes == nullptr) {
//         printf("BVH nodes are nullptr.\n");
//         return;
//     }

//     printf("Copying %d BVH nodes to device memory\n", numOfBvhNodes);

//     cudaMalloc(&curGeometry.devBvhNodes, numOfBvhNodes * sizeof(BVHNode));
//     cudaMemcpy(curGeometry.devBvhNodes, curGeometry.bvhNodes, numOfBvhNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

//     delete[] curGeometry.bvhNodes;
//     curGeometry.bvhNodes = nullptr;
// }

// void copyBvhInfoFromHostToDevice(std::vector<Geom> &geometries) {
//     int totalNumberOfGeom = geometries.size();
//     for (int i = 0; i < totalNumberOfGeom; i++) {
//         Geom &curGeometry = geometries[i];
//         if (curGeometry.type != MESH) {
//             continue;
//         }

//         copyBvhTrianglesFromHostToDevice(curGeometry); // Copy the BVH triangles to the device memory
//         copyBvhNodesFromHostToDevice(curGeometry); // Copy the BVH nodes to the device memory
//     }
// }

void copyTexturesFromHostToDevice(const int numTextures, const std::vector<std::tuple<glm::vec4*, glm::ivec2>> &textures, Texture* &dev_textures) {
    // Step 1: Allocate memory on the device for the Texture array
    cudaMalloc(&dev_textures, numTextures * sizeof(Texture));

    // Step 2: Loop over each texture and copy its data to the device
    std::vector<Texture> h_textures(numTextures);  // Host-side Texture array to temporarily hold device pointers
    for (int i = 0; i < numTextures; i++) {
        // Get texture data and size from the input vector
        glm::vec4* hostTextureData = std::get<0>(textures[i]);
        glm::ivec2 textureSize = std::get<1>(textures[i]);

        // Allocate memory on the device for the texture data
        glm::vec4* dev_textureData;
        size_t textureDataSize = textureSize.x * textureSize.y * sizeof(glm::vec4);
        cudaMalloc(&dev_textureData, textureDataSize);

        // Copy texture data from host to device
        cudaMemcpy(dev_textureData, hostTextureData, textureDataSize, cudaMemcpyHostToDevice);

        // Fill in the host Texture struct with the size and device pointer
        h_textures[i].size = textureSize;
        h_textures[i].dev_data = dev_textureData;
    }

    // Step 3: Copy the array of Texture objects from host to device
    cudaMemcpy(dev_textures, h_textures.data(), numTextures * sizeof(Texture), cudaMemcpyHostToDevice);

    // Step 4: Free the memory allocated for the host Texture array
    // delete[] h_textures.data();

    // Optionally, add error checks after each CUDA call:
    checkCUDAError("Texture Copying");
}

void initialiseTextures(Scene* scene) {
    std::vector<tuple<glm::vec4*, glm::ivec2>> &albedoTextures = scene->albedoTextures;
    std::vector<tuple<glm::vec4*, glm::ivec2>> &normalTextures = scene->normalTextures;
    std::vector<tuple<glm::vec4*, glm::ivec2>> &bumpTextures = scene->bumpTextures;

    if (albedoTextures.size() > 0) {
        numAlbedoTextures = albedoTextures.size();
        copyTexturesFromHostToDevice(numAlbedoTextures, albedoTextures, dev_albedoTextures);
        checkCUDAError("Alebdo Textures Initialisation");
    }

    if (normalTextures.size() > 0) {
        numNormalTextures = normalTextures.size();
        copyTexturesFromHostToDevice(numNormalTextures, normalTextures, dev_normalTextures);
        checkCUDAError("Normal Textures Initialisation");
    }

    if (bumpTextures.size() > 0) {
        numBumpTextures = bumpTextures.size();
        copyTexturesFromHostToDevice(numBumpTextures, bumpTextures, dev_bumpTextures);
        checkCUDAError("Bump Textures Initialisation");
    }

    checkCUDAError("Texture Initialisation");
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    // #if USE_BVH
    //     // Copy the BVH triangles and nodes to the device memory
    //     copyBvhInfoFromHostToDevice(scene->geoms);
    // #else
    //     for (Geom &curGeom : scene->geoms) {
    //         if (curGeom.type != MESH) {
    //             continue;
    //         }

    //         if (curGeom.bvhTriangles != nullptr) {
    //             delete[] curGeom.bvhTriangles;
    //             curGeom.bvhTriangles = nullptr;
    //         }
            
    //         if (curGeom.bvhNodes != nullptr) {
    //             delete[] curGeom.bvhNodes;
    //             curGeom.bvhNodes = nullptr;
    //         }
    //     }
    // #endif

    int totalNumberOfGeom = scene->geoms.size();
    initialiseTriangles(dev_geomTriangles, scene->geoms, totalNumberOfGeom); // Must appear before initializing dev_geoms
    cudaMalloc(&dev_geoms, totalNumberOfGeom * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    int totalNumberOfLights = scene->lights.size();
    initialiseTriangles(dev_lightTriangles, scene->lights, totalNumberOfLights); // Must appear before initializing dev_lights
    cudaMalloc(&dev_lights, totalNumberOfLights * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_totalNumberOfLights, sizeof(int));
    cudaMemcpy(dev_totalNumberOfLights, &totalNumberOfLights, sizeof(int), cudaMemcpyHostToDevice);

    // We've already got the triangles in the device memory, so we can delete them from the host memory
    delete scene->geoms.data()->triangles;

    initialiseTextures(scene);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void freeTexturesOnDevice(const int numTextures, Texture* dev_textures) {
    // Step 1: Allocate a host-side array to copy the device-side Texture array
    std::vector<Texture> h_textures(numTextures);

    // Step 2: Copy the Texture array from the device to the host
    cudaMemcpy(h_textures.data(), dev_textures, numTextures * sizeof(Texture), cudaMemcpyDeviceToHost);

    // Step 3: Loop through each texture and free the device memory for dev_data
    for (int i = 0; i < numTextures; i++) {
        if (h_textures[i].dev_data != nullptr) {
            cudaFree(h_textures[i].dev_data);  // Free each texture's dev_data
        }
    }

    // Step 4: Free the memory allocated for the dev_textures array itself
    cudaFree(dev_textures);
}


void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_lights);
    cudaFree(dev_geomTriangles);
    cudaFree(dev_lightTriangles);
    cudaFree(dev_totalNumberOfLights);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    if (dev_albedoTextures != NULL) {
        freeTexturesOnDevice(numAlbedoTextures, dev_albedoTextures);
        cudaFree(dev_albedoTextures);
    }
    if (dev_normalTextures != NULL) {
        freeTexturesOnDevice(numNormalTextures, dev_normalTextures);
        cudaFree(dev_normalTextures);
    }
    if (dev_bumpTextures != NULL) {
        freeTexturesOnDevice(numBumpTextures, dev_bumpTextures);
        cudaFree(dev_bumpTextures);
    }

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

    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }

    int index = x + (y * cam.resolution.x);

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
    thrust::uniform_real_distribution<float> u01(0, 1);

    PathSegment& segment = pathSegments[index];

    glm::vec3 rayOrigin = cam.position;
    
    #if USE_ANTIALIASING
        // Jittering for anti-aliasing
        glm::vec2 offset = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));

        // Compute primary ray direction
        glm::vec3 rayDirection = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + offset[0])
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + offset[1])
        );
    #else 
        glm::vec3 rayDirection = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
    #endif

    // Depth of field
    float lensRadius = cam.lensRadius;
    float focalDistance = cam.focalDistance;
    if (lensRadius > 0.0f && focalDistance > 0.0f) {
        // Compute the focal point
        glm::vec3 focalPoint = rayOrigin + cam.focalDistance * rayDirection;

        // Sample point on lens (circular aperture sampling)
        glm::vec2 apartureSample = glm::vec2(u01(rng), u01(rng));
        glm::vec3 newOrigin = glm::vec3(lensRadius * squareToUniformDisk(apartureSample), 0.0f);

        // Offset the ray origin based on lens sampling
        rayOrigin += cam.right * newOrigin.x + cam.up * newOrigin.y;

        // Recalculate the direction to pass through the focal point
        rayDirection = glm::normalize(focalPoint - rayOrigin);
    }

    // Assign values to the path segment
    segment.ray.origin = rayOrigin;
    segment.ray.direction = rayDirection;
    segment.color = glm::vec3(1.0f);
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.hasHitLight = false;
    segment.eta = 1.0f;
}


// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

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
            else if (geom.type == MESH) {
                // #if USE_BVH
                //     t = meshIntersectionTestBVH(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                // #else
                //     t = meshIntersectionTestNaive(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
                // #endif
                t = meshIntersectionTestNaive(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
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
            Geom hitGeom = geoms[hit_geom_index];
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materials.materialId = hitGeom.material.materialid;
            intersections[path_index].materials.albedoTextureID = hitGeom.material.albedoTextureID;
            intersections[path_index].materials.normalTextureID = hitGeom.material.normalTextureID;
            intersections[path_index].materials.bumpTextureID = hitGeom.material.bumpTextureID;

            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
        }
    }
}

__device__ glm::vec4 sampleTexture(Texture texture, glm::vec2 uv, bool isBump = false) {
    int x = static_cast<int>(uv.x * (texture.size.x - 1));
    int y = static_cast<int>(uv.y * (texture.size.y - 1));
    int idx = y * texture.size.x + x; // Row-major order indexing
    glm::vec4 val = texture.dev_data[idx]; // Access the texel
    
    if (!isBump) {
        return val; // Regular texture sampling if not a bump map
    }

    // If it's a bump map, calculate du and dv using finite differences
    float epsilon = 1.0f / texture.size.x;  // Small step, inverse of texture width

    // Sample neighboring texels for finite difference
    int xPlus = static_cast<int>((uv.x + epsilon) * (texture.size.x - 1));
    int yPlus = static_cast<int>((uv.y + epsilon) * (texture.size.y - 1));

    // Ensure we stay within texture bounds
    xPlus = min(xPlus, texture.size.x - 1);
    yPlus = min(yPlus, texture.size.y - 1);

    // Indices for neighboring texels
    int idxXPlus = y * texture.size.x + xPlus;
    int idxYPlus = yPlus * texture.size.x + x;

    // Access neighboring texels
    float height = val.r;                 // Current height (R channel for bump)
    float heightXPlus = texture.dev_data[idxXPlus].r;  // Height at (u + epsilon, v)
    float heightYPlus = texture.dev_data[idxYPlus].r;  // Height at (u, v + epsilon)

    // Compute finite differences (du, dv)
    float du = (heightXPlus - height) / epsilon;
    float dv = (heightYPlus - height) / epsilon;

    // Return the du and dv as a vec4 for further processing
    // You can store du in the R channel and dv in the G channel
    return glm::vec4(du, dv, 0.0f, 0.0f);  // Return du, dv for normal perturbation
}

__device__ glm::vec3 checkerboard(float u, float v, int checkerSize) {
    int u_check = static_cast<int>(floor(u * checkerSize)) % 2;
    int v_check = static_cast<int>(floor(v * checkerSize)) % 2;

    if (u_check == v_check) {
        return glm::vec3(1.0f, 1.0f, 1.0f);  // white square
    } else {
        return glm::vec3(0.0f, 0.0f, 0.0f);  // black square
    }
}

__global__ void shadeNaive(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* albedoTextures,
    Texture* normalTextures,
    Texture* bumpTextures) {
    // As long as we enter here, it means the ray has remaining bounces > 0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    #if !USE_STREAM_COMPACTION
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }
    #endif

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];
    if (intersection.t <= 0.0f) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materials.materialId];
    glm::vec2 uv = intersection.uv;
    TextureValues texVals;

    bool hasAlbedoTexture = intersection.materials.albedoTextureID != -1;
    bool hasNormalTexture = intersection.materials.normalTextureID != -1;
    bool hasBumpTexture = intersection.materials.bumpTextureID != -1;

    texVals.albedo = glm::vec4(INFINITY);
    texVals.normal = glm::vec4(INFINITY);
    texVals.bump = glm::vec4(INFINITY);

    #if USE_CHECKERBOARD_TEXTURE
        if (hasAlbedoTexture) { 
            texVals.albedo = glm::vec4(checkerboard(uv.x, uv.y, 101), 1.0f);
        }
    #else
        if (hasAlbedoTexture) {
            texVals.albedo = sampleTexture(albedoTextures[intersection.materials.albedoTextureID], uv);
        }
    #endif
    
    if (hasNormalTexture) {
        texVals.normal = sampleTexture(normalTextures[intersection.materials.normalTextureID], uv);
    }

    if (hasBumpTexture) {
        texVals.bump = sampleTexture(bumpTextures[intersection.materials.bumpTextureID], uv, true);
    }

    glm::vec3 materialColor = material.color;
    
    if (material.emittance > 0.0f) {
        pathSegment.color *= materialColor * material.emittance;
        pathSegment.remainingBounces = 0;
        pathSegment.hasHitLight = true;
    }
    else {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        glm::vec3 oldIntersect = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 surfaceNormal = glm::normalize(intersection.surfaceNormal);
        glm::vec3 woW = -pathSegment.ray.direction;
        glm::vec3 wiW;
        glm::vec3 c;
        float pdf;
        float eta;
        
        scatterRay(pathSegment, woW, surfaceNormal, wiW, pdf, c, eta, material, texVals, rng); 

        pathSegment.ray.direction = wiW; // wiW should already be normalized
        // Without the offset, when the ray immediately intersects the surface it originated from, the refraction calculations may fail or yield invalid results, such as:
        // Total Internal Reflection: The refracted ray might get treated as a reflective ray due to intersection problems, resulting in no transmitted light.
        // Black Pixels: The lack of refraction or valid light contribution can result in areas appearing black, as seen in your case.
        pathSegment.ray.origin = oldIntersect + pathSegment.ray.direction * 0.01f;
        pathSegment.color *= c; 
        
        #if (USE_RUSSIAN_ROULETTE) // Possibly terminate the path with Russian roulette
            if (depth > 3) {
                // So that the ray can bounce for a bit before we start terminating it
                float maxComponent = fmaxf(c.x, fmaxf(c.y, c.z));
                float survivalProbability = u01(rng);
                float eta_sq = eta * eta;
                float q = fminf(maxComponent * eta_sq, 0.99f);
                
                if (q < survivalProbability) {
                    pathSegment.remainingBounces = 0;
                    return;
                }
                else {
                    pathSegment.color /= q;
                }
            }        
        #endif
        
        pathSegment.remainingBounces--;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.hasHitLight) {
            image[iterationPath.pixelIndex] += iterationPath.color;  
        }   
    }
}

__global__ void computeIsIntersected(int num_paths, int* isIntersected, const ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    isIntersected[idx] = intersections[idx].t != -1.0f;
}

void partitionRays(int &num_paths, PathSegment* dev_paths, const ShadeableIntersection* dev_intersections) {
    thrust::device_ptr<PathSegment> dev_ptr(dev_paths);
    thrust::device_ptr<PathSegment> dev_ptr_end = thrust::stable_partition(thrust::device, dev_ptr, dev_ptr + num_paths, has_remaining_bounces());
    cudaDeviceSynchronize();
    num_paths = dev_ptr_end - dev_ptr;
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int total_num_paths = dev_path_end - dev_paths;
    int num_paths = total_num_paths;

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
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // Sort materials by type
        #if USE_MATERIAL_SORT
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortMaterialCondition());
            cudaDeviceSynchronize();
        #endif

        shadeNaive<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_albedoTextures,
            dev_normalTextures,
            dev_bumpTextures
        );
        cudaDeviceSynchronize();

        #if USE_STREAM_COMPACTION
            // compact paths
            partitionRays(num_paths, dev_paths, dev_intersections);
        #endif

        iterationComplete = (depth >= traceDepth) || (num_paths == 0);
        
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(total_num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}