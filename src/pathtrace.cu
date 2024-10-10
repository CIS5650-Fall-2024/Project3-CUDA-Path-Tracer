#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define DEBUG_SHADER 0




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

static Triangle* dev_triangles = NULL;
// Maximum number of textures (adjust as needed)
#define MAX_TEXTURES 16

// Texture object arrays
cudaTextureObject_t* dev_textures = NULL;

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}




// Predicate to identify terminated paths (paths with no remaining bounces)
struct PathSegmentTerminated
{
    __host__ __device__
        bool operator()(const PathSegment& segment)
    {
        return (segment.remainingBounces == 0);
    }
};

// Structure to hold material IDs for sorting
struct MaterialSort
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
    {
        return a.materialId < b.materialId;
    }
};

// Kernel that writes the image to the OpenGL PBO directly.
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

        // Each thread writes one pixel location in the texture (texel)
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
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
    guiData->SortByMaterial = false;      // Initialize the variable
    guiData->AntiAliasing = true;        // Initialize antialiasing toggle
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

    // Allocate and copy triangles to device
    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    // Allocate device memory for texture objects
    if (!scene->textures.empty())
    {
        cudaMalloc(&dev_textures, scene->textures.size() * sizeof(cudaTextureObject_t));

        // Create texture objects
        for (size_t i = 0; i < scene->textures.size(); ++i) {
            cudaTextureObject_t texObj;

            // Allocate CUDA array
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
            cudaArray* cuArray;
            cudaMallocArray(&cuArray, &channelDesc, scene->textures[i].width, scene->textures[i].height);

            // Copy texture data to CUDA array
            cudaMemcpy2DToArray(
                cuArray,
                0, 0,
                scene->textures[i].imageData.data(),
                scene->textures[i].width * 4 * sizeof(unsigned char), // 4 bytes per pixel (RGBA)
                scene->textures[i].width * 4 * sizeof(unsigned char),
                scene->textures[i].height,
                cudaMemcpyHostToDevice
            );

            // Set texture resource description
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray;

            // Set texture description
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 1;

            // Create texture object
            cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

            // Copy texture object to device memory
            cudaMemcpy(&(dev_textures[i]), &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
        }
    }

    checkCUDAError("pathtraceInit");
}


void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    // Free triangles
    if (dev_triangles != NULL)
    {
        cudaFree(dev_triangles);
        dev_triangles = NULL;
    }

    // Free texture objects and arrays
    if (dev_textures != NULL)
    {
        // Destroy texture objects (if necessary)
        // Note: You might need to keep track of cudaArray pointers to free them

        cudaFree(dev_textures);
        dev_textures = NULL;
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antiAliasing)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        // Set up RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float x_offset = 0.0f;
        float y_offset = 0.0f;

        if (antiAliasing)
        {
            x_offset = u01(rng) - 0.5f;
            y_offset = u01(rng) - 0.5f;
        }

        // Compute the point on the image plane
        glm::vec3 imagePoint = cam.position + cam.view * cam.focalDistance
            - cam.right * cam.pixelLength.x * ((float)x + x_offset - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + y_offset - (float)cam.resolution.y * 0.5f);

        // Jitter the ray origin within the lens aperture
        float r = cam.lensRadius * sqrtf(u01(rng));
        float theta = 2.0f * PI * u01(rng);
        float lensU = r * cosf(theta);
        float lensV = r * sinf(theta);

        glm::vec3 lensPoint = cam.position + cam.right * lensU + cam.up * lensV;

        // Compute the new ray direction
        glm::vec3 rayDirection = glm::normalize(imagePoint - lensPoint);

        segment.ray.origin = lensPoint;
        segment.ray.direction = rayDirection;
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.color = glm::vec3(1.0f);
    }
}



__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        int tmp_triangleIndex = -1;
        int tmp_materialId = -1;
        glm::vec3 tmp_barycentricCoords;
        bool outside = true;

        // Iterate over all geometries
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            float t = -1.0f;
            glm::vec3 intersect_point;
            glm::vec3 normal;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, intersect_point, normal, outside);

                // Update closest intersection for CUBE
                if (t > 0.0f && t < t_min)
                {
                    t_min = t;
                    hit_geom_index = i;
                    tmp_intersect = intersect_point;
                    tmp_normal = normal;
                    tmp_materialId = geom.materialid;
                    tmp_triangleIndex = -1;
                    tmp_barycentricCoords = glm::vec3(0.0f);
                }
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, intersect_point, normal, outside);

                // Update closest intersection for SPHERE
                if (t > 0.0f && t < t_min)
                {
                    t_min = t;
                    hit_geom_index = i;
                    tmp_intersect = intersect_point;
                    tmp_normal = normal;
                    tmp_materialId = geom.materialid;
                    tmp_triangleIndex = -1;
                    tmp_barycentricCoords = glm::vec3(0.0f);
                }
            }
            else if (geom.type == CUSTOM_MESH)
            {
                // Transform ray to object space
                Ray rt;
                rt.origin = multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
                rt.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

                // First, check if the ray intersects the mesh's bounding box
                float t_aabb;
                if (aabbIntersectionTest(rt, geom.bboxMin, geom.bboxMax, t_aabb))
                {
                    // If it does, proceed to test triangles
                    for (int j = 0; j < geom.triangleCount; ++j)
                    {
                        Triangle triangle = triangles[geom.triangleStartIndex + j];

                        float t_temp;
                        glm::vec3 baryPosition;
                        bool hitTriangle = triangleIntersectionTest(rt, triangle, t_temp, baryPosition);
                        if (hitTriangle && t_temp > 0.0f && t_temp < t_min)
                        {
                            t = t_temp;

                            // Compute intersection point in world space
                            glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
                            intersect_point = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.0f));

                            // Interpolate normals
                            glm::vec3 normalObjSpace = glm::normalize(
                                triangle.n0 * (1.0f - baryPosition.x - baryPosition.y) +
                                triangle.n1 * baryPosition.x +
                                triangle.n2 * baryPosition.y
                            );
                            normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normalObjSpace, 0.0f)));

                            // Update closest intersection
                            if (t > 0.0f && t < t_min)
                            {
                                t_min = t;
                                hit_geom_index = i;
                                tmp_intersect = intersect_point;
                                tmp_normal = normal;
                                tmp_materialId = geom.materialid;
                                tmp_triangleIndex = geom.triangleStartIndex + j;
                                tmp_barycentricCoords = baryPosition;
                            }
                        }
                    }
                }
            }
        }

        // Fill in the intersection record
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surfaceNormal = tmp_normal;
            intersections[path_index].materialId = tmp_materialId;
            intersections[path_index].triangleIndex = tmp_triangleIndex;
            intersections[path_index].barycentricCoords = tmp_barycentricCoords;
            intersections[path_index].hitGeomType = geoms[hit_geom_index].type;
        }
    }
}



/**
 * Shading kernel that applies material properties and generates new rays
 * by scattering the ray according to the material's BSDF.
 * It also accumulates the color contribution to the image buffer when a path terminates.
 */


#if !DEBUG_SHADER
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* textures,
    Geom* geoms,
    Triangle* triangles,
    glm::vec3* dev_image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& segment = pathSegments[idx];
        ShadeableIntersection& intersection = shadeableIntersections[idx];

        if (segment.remainingBounces > 0 && intersection.t > 0.0f)
        {
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, segment.pixelIndex, segment.remainingBounces);

            // Make a local copy of the material
            Material material = materials[intersection.materialId];

            // Get the intersection point and normal
            glm::vec3 intersect = segment.ray.origin + segment.ray.direction * intersection.t;
            glm::vec3 normal = glm::normalize(intersection.surfaceNormal);

            // If the material is emissive, accumulate the emitted light and terminate the path
            if (material.emittance > 0.0f)
            {
                // Multiply the color by the material's emission
                segment.color *= (material.color * material.emittance);
                // Accumulate the color into the image
                dev_image[segment.pixelIndex] += segment.color;
                // Terminate the path
                segment.remainingBounces = 0;
                // Set the color to zero to prevent further accumulation
                segment.color = glm::vec3(0.0f);
            }
            else
            {
                // Initialize baseColor with the material's color
                glm::vec3 baseColor = material.color;

                if (intersection.hitGeomType == CUSTOM_MESH)
                {
                    // Triangle-specific code
                    glm::vec2 uv(0.0f);

                    // Retrieve the triangle and barycentric coordinates
                    Triangle triangle = triangles[intersection.triangleIndex];
                    glm::vec3 baryPosition = intersection.barycentricCoords;

                    // Interpolate UV coordinates
                    uv = barycentricInterpolation(
                        triangle.uv0, triangle.uv1, triangle.uv2,
                        baryPosition
                    );

                    float u = uv.x;
                    float v = uv.y;

                    // Sample base color texture if available
                    if (material.baseColorTextureIndex >= 0)
                    {
                        cudaTextureObject_t texObj = textures[material.baseColorTextureIndex];

                        // Sample the texture
                        float4 texColor = tex2D<float4>(texObj, u, v);
                        baseColor *= glm::vec3(texColor.x, texColor.y, texColor.z);
                    }

                    // Apply normal map if available
                    if (material.normalTextureIndex >= 0)
                    {
                        // Sample the normal map
                        cudaTextureObject_t normalTexObj = textures[material.normalTextureIndex];
                        float4 normalSample = tex2D<float4>(normalTexObj, u, v);

                        // Convert from [0,1] to [-1,1]
                        glm::vec3 normalMap = glm::normalize(glm::vec3(
                            normalSample.x * 2.0f - 1.0f,
                            normalSample.y * 2.0f - 1.0f,
                            normalSample.z * 2.0f - 1.0f
                        ));

                        // Compute tangent, bitangent, and normal vectors
                        glm::vec3 tangent, bitangent;

                        computeTangentSpace(
                            triangle.v0, triangle.v1, triangle.v2,
                            triangle.uv0, triangle.uv1, triangle.uv2,
                            triangle.n0, triangle.n1, triangle.n2,
                            baryPosition,
                            tangent, bitangent, normal
                        );

                        // Transform normal from tangent space to world space
                        glm::mat3 TBN = glm::mat3(tangent, bitangent, normal);
                        normal = glm::normalize(TBN * normalMap);
                    }
                }
                else
                {
                    // do after semester ends
                }

                // Update material color with texture in local copy
                material.color = baseColor;

                // Scatter the ray using the BSDF with the updated normal
                scatterRay(segment, intersect, normal, material, rng);

                // Decrement remaining bounces
                segment.remainingBounces--;

                // If no remaining bounces, accumulate the color
                if (segment.remainingBounces == 0)
                {
                    dev_image[segment.pixelIndex] += segment.color;
                    segment.color = glm::vec3(0.0f);
                }
            }
        }
        else
        {
            // No background light; accumulate black
            dev_image[segment.pixelIndex] += glm::vec3(0.0f);

            // Terminate the path
            segment.remainingBounces = 0;
            segment.color = glm::vec3(0.0f);
        }
    }
}


#endif


// debug version of shadeMaterial, to visualize UVs + flat colors
#if DEBUG_SHADER
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* textures,
    Geom* geoms,
    Triangle* triangles,
    glm::vec3* dev_image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& segment = pathSegments[idx];
        ShadeableIntersection& intersection = shadeableIntersections[idx];

        // Check if the path hit something
        if (intersection.t > 0.0f)
        {
            // Get the material
            Material& material = materials[intersection.materialId];

            // Calculate the intersection point (not really needed here, but keeping it for clarity)
            glm::vec3 intersect = segment.ray.origin + segment.ray.direction * intersection.t;

            // Initialize the color as the material base color
            glm::vec3 baseColor = material.color;

            // Initialize UV coordinates
            glm::vec2 uv(0.0f);

            // Retrieve the triangle and barycentric coordinates for UV interpolation
            Triangle triangle = triangles[intersection.triangleIndex];
            glm::vec3 baryPosition = intersection.barycentricCoords;

            // Interpolate UV coordinates
            uv = barycentricInterpolation(
                triangle.uv0, triangle.uv1, triangle.uv2,
                baryPosition
            );

            // Wrap UVs into [0, 1] range
            float u = uv.x;
            float v = uv.y;

            // Sample base color texture if available
            if (material.baseColorTextureIndex >= 0)
            {
                cudaTextureObject_t texObj = textures[material.baseColorTextureIndex];

                // Sample the texture using the UV coordinates
                float4 texColor = tex2D<float4>(texObj, u, v);
                baseColor = glm::vec3(texColor.x, texColor.y, texColor.z); // Use texture color as base color
            }

            // Directly assign the base color to the pixel in the image buffer
            dev_image[segment.pixelIndex] += baseColor;
            segment.remainingBounces = 0;
            segment.color = glm::vec3(0.0f);

        }
        else
        {
            // If no hit, assign background color (black) to the pixel
            dev_image[segment.pixelIndex] += glm::vec3(0.0f);
        }
    }
}
#endif



// Helper functions

__device__ glm::vec2 barycentricInterpolation(
    const glm::vec2& uv0,
    const glm::vec2& uv1,
    const glm::vec2& uv2,
    const glm::vec3& baryPosition)
{
    float u = baryPosition.x;
    float v = baryPosition.y;
    float w = 1.0f - u - v;
    return uv0 * w + uv1 * u + uv2 * v;
}

__device__ void computeTangentSpace(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2,
    const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
    const glm::vec3& baryPosition,
    glm::vec3& tangent, glm::vec3& bitangent, glm::vec3& normal)
{
    // Compute edges and delta UVs
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec2 deltaUV1 = uv1 - uv0;
    glm::vec2 deltaUV2 = uv2 - uv0;

    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

    tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    bitangent = f * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);

    tangent = glm::normalize(tangent);
    bitangent = glm::normalize(bitangent);

    // Interpolate normals
    normal = glm::normalize(
        n0 * (1.0f - baryPosition.x - baryPosition.y) +
        n1 * baryPosition.x +
        n2 * baryPosition.y
    );
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

    // Generate initial rays from the camera for each pixel
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, guiData->AntiAliasing);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot rays into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // Clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // Tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // --- Material Sorting ---
        if (guiData != NULL && guiData->SortByMaterial)
        {
            // Sort the intersections and corresponding paths by material ID
            thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
            thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);

            thrust::sort_by_key(
                dev_thrust_intersections,
                dev_thrust_intersections + num_paths,
                dev_thrust_paths,
                MaterialSort()
            );
        }

        // Shading
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF. Accumulate colors of terminated paths into dev_image.
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_geoms,
            dev_triangles,
            dev_image
            );


          

        cudaDeviceSynchronize(); // Ensure the kernel has completed
        checkCUDAError("shadeMaterial");
        // Stream compaction to remove terminated paths (paths with remainingBounces == 0)
        PathSegment* new_end = thrust::remove_if(
            thrust::device, dev_paths, dev_paths + num_paths, PathSegmentTerminated());
        num_paths = new_end - dev_paths;

        // Terminate the loop if all paths have terminated or reached maximum depth
        if (num_paths == 0 || depth >= traceDepth)
        {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Since we have already accumulated the colors into dev_image during shading,
    // we don't need to call finalGather here.

    // Send results to OpenGL buffer for rendering
    sendImageToPBO <<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
