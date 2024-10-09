#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "oidn.hpp"

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool finalIteration)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / (float)iter;

        // Calculate luminance
        float luminance = 0.2126f * pix.x + 0.7152f * pix.y + 0.0722f * pix.z;

        // Apply Reinhard tone mapping to luminance
        float mappedLuminance = luminance / (luminance + 1.0f);

        // Scale the color by the ratio of mapped luminance to original luminance
        if (luminance > 0.0f) {
            pix *= (mappedLuminance / luminance);
        }

		// Gamma correction
		pix = glm::pow(pix, glm::vec3(1.0f / 2.2f));

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        if (finalIteration) {
            image[index] = color;
        }
        else {
            // Each thread writes one pixel location in the texture (textel)
            pbo[index].w = 0;
            pbo[index].x = color.x;
            pbo[index].y = color.y;
            pbo[index].z = color.z;
        }
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_normals = NULL; // Accumulated normals of points hit after each intersection test
static glm::vec3* dev_albedos = NULL;
static Geom* dev_geoms = NULL;
static Mesh* dev_meshes = NULL;
static glm::vec3* dev_meshNormals = NULL; // Normals of vertices in all meshes in scene
static Triangle* dev_triangles = NULL;
static glm::vec3* dev_vertices = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static cudaTextureObject_t* dev_texture_objects = NULL;
static cudaArray_t dev_envMapCuArray = NULL;
static cudaTextureObject_t dev_envMapTexObj = NULL;
static glm::vec2* dev_baseColorUvs = NULL;
static glm::vec2* dev_normalUvs = NULL;
static glm::vec2* dev_emissiveUvs = NULL;
static BvhNode* dev_bvhNodes = NULL;


#ifdef NDEBUG
static oidn::DeviceRef oidnDevice = NULL;
#endif

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

	cudaMalloc(&dev_normals, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_normals, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_albedos, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_albedos, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
	cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_meshNormals, scene->normals.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_meshNormals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_baseColorUvs, scene->baseColorUvs.size() * sizeof(glm::vec2));
	cudaMemcpy(dev_baseColorUvs, scene->baseColorUvs.data(), scene->baseColorUvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_normalUvs, scene->normalUvs.size() * sizeof(glm::vec2));
	cudaMemcpy(dev_normalUvs, scene->normalUvs.data(), scene->normalUvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_emissiveUvs, scene->emissiveUvs.size() * sizeof(glm::vec2));
	cudaMemcpy(dev_emissiveUvs, scene->emissiveUvs.data(), scene->emissiveUvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_texture_objects, scene->textures.size() * sizeof(cudaTextureObject_t));

    // Allocate and copy textures
    for (int i = 0; i < scene->textures.size(); ++i) {
		const Texture& texture = scene->textures[i];
        cudaArray_t dev_texture;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        cudaMallocArray(&dev_texture, &channelDesc, texture.width, texture.height);
		cudaMemcpy2DToArray(dev_texture, 0, 0, texture.data.data(), texture.width * sizeof(float4), texture.width * sizeof(float4), texture.height, cudaMemcpyHostToDevice);

        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_texture;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

		cudaMemcpy(dev_texture_objects + i, &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }

	// Allocate and copy environment map
	if (scene->envMap.data.size() > 0) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cudaMallocArray(&dev_envMapCuArray, &channelDesc, scene->envMap.width, scene->envMap.height);
		cudaMemcpy2DToArray(dev_envMapCuArray, 0, 0, scene->envMap.data.data(), scene->envMap.width * sizeof(float4), scene->envMap.width * sizeof(float4), scene->envMap.height, cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = dev_envMapCuArray;

		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&dev_envMapTexObj, &resDesc, &texDesc, nullptr);
	}

	cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BvhNode));
	cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");

    // Initialize OIDN
#ifdef NDEBUG
	oidnDevice = oidn::newDevice(oidn::DeviceType::CUDA);
	oidnDevice.commit();
#endif
}

void pathtraceFree(Scene* scene)
{
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_normals);
	cudaFree(dev_albedos);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
	cudaFree(dev_meshes);
	cudaFree(dev_meshNormals);
	cudaFree(dev_triangles);
	cudaFree(dev_vertices);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
	cudaFree(dev_baseColorUvs);
	cudaFree(dev_normalUvs);
	cudaFree(dev_emissiveUvs);
	cudaFree(dev_bvhNodes);

    if (dev_texture_objects != NULL) {
        for (int i = 0; i < scene->textures.size(); ++i) {
			cudaTextureObject_t texObj;
			cudaMemcpy(&texObj, dev_texture_objects + i, sizeof(cudaTextureObject_t), cudaMemcpyDeviceToHost);
			cudaDestroyTextureObject(texObj);
        }
    }

	cudaFree(dev_texture_objects);

	if (dev_envMapCuArray != NULL) {
		cudaFreeArray(dev_envMapCuArray);
		cudaDestroyTextureObject(dev_envMapTexObj);
	}

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* TODO: motion blur - jitter rays "in time"
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Antialiasing jitter - randomizes the ray direction within a pixel
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
		thrust::uniform_real_distribution<float> unifDist(-0.5, 0.5);
		float jitterX = unifDist(rng);
		float jitterY = unifDist(rng);

        glm::vec3 rayDirection  = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

		// Depth of field jitter - randomizes the ray origin within the aperture
		// First, calculate point on the lens as a random point within the aperture
        thrust::uniform_real_distribution<float> apertureDist(-1.0, 1.0);
		glm::vec3 lensPoint = cam.position + (cam.apertureRadius * 2.0f * (cam.up * apertureDist(rng) + cam.right * apertureDist(rng)));

		// Next, calculate point of focus on the focal plane
		glm::vec3 focusPoint = cam.position + (cam.focalLength * rayDirection);

		// Finally, calculate the ray direction from the lens point to the focus point
		segment.ray.direction = glm::normalize(focusPoint - lensPoint);
		segment.ray.origin = lensPoint;

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    Material* materials,
    glm::vec3* vertices,
    Mesh* meshes,
    Triangle* triangles,
    glm::vec3* meshNormals,
    glm::vec2* baseColorUvs,
    glm::vec2* normalUvs,
    glm::vec2* emissiveUvs,
    BvhNode* bvhNodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths) return;
    PathSegment pathSegment = pathSegments[path_index];
	extern __shared__ int sharedMemory[];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
	glm::vec2 baryCoords;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    int hit_triangle_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
	glm::vec2 tmp_baryCoords;
	int tmp_hit_triangle_index;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];
		const Mesh& mesh = meshes[geom.meshId];

        t = meshIntersectionTest(geom, triangles, vertices, meshNormals, mesh, mesh.bvhRootIndex, bvhNodes, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_hit_triangle_index, tmp_baryCoords, sharedMemory, t_min);

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            baryCoords = tmp_baryCoords;
			hit_triangle_index = tmp_hit_triangle_index;
        }
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
		const Mesh& mesh = meshes[geoms[hit_geom_index].meshId];
		const Triangle& triangle = triangles[hit_triangle_index];

        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
        

		// Compute UVs for the hit point using barycentric coordinates
        if (materials[geoms[hit_geom_index].materialid].baseColorTextureId != -1) {
            const glm::vec2& uv0 = baseColorUvs[mesh.baseColorUvIndex + triangle.attributeIndex[0]];
            const glm::vec2& uv1 = baseColorUvs[mesh.baseColorUvIndex + triangle.attributeIndex[1]];
            const glm::vec2& uv2 = baseColorUvs[mesh.baseColorUvIndex + triangle.attributeIndex[2]];
            intersections[path_index].baseColorUvs = uv0 * (1.0f - baryCoords.x - baryCoords.y) + uv1 * baryCoords.x + uv2 * baryCoords.y;
        }
		if (materials[geoms[hit_geom_index].materialid].normalTextureId != -1) {
			const glm::vec2& uv0 = normalUvs[mesh.normalUvIndex + triangle.attributeIndex[0]];
			const glm::vec2& uv1 = normalUvs[mesh.normalUvIndex + triangle.attributeIndex[1]];
			const glm::vec2& uv2 = normalUvs[mesh.normalUvIndex + triangle.attributeIndex[2]];
            intersections[path_index].normalUvs = uv0 * (1.0f - baryCoords.x - baryCoords.y) + uv1 * baryCoords.x + uv2 * baryCoords.y;

            // Compute bitangents and tangents at each vertex and interpolate
			glm::vec3 edge0 = vertices[mesh.vertStartIndex + triangle.attributeIndex[1]] - vertices[mesh.vertStartIndex + triangle.attributeIndex[0]];
			glm::vec3 edge1 = vertices[mesh.vertStartIndex + triangle.attributeIndex[2]] - vertices[mesh.vertStartIndex + triangle.attributeIndex[1]];
            glm::vec3 edge2 = vertices[mesh.vertStartIndex + triangle.attributeIndex[0]] - vertices[mesh.vertStartIndex + triangle.attributeIndex[2]];

			glm::vec2 deltaUV0 = normalUvs[mesh.normalUvIndex + triangle.attributeIndex[1]] - normalUvs[mesh.normalUvIndex + triangle.attributeIndex[0]];
			glm::vec2 deltaUV1 = normalUvs[mesh.normalUvIndex + triangle.attributeIndex[2]] - normalUvs[mesh.normalUvIndex + triangle.attributeIndex[1]];
			glm::vec2 deltaUV2 = normalUvs[mesh.normalUvIndex + triangle.attributeIndex[0]] - normalUvs[mesh.normalUvIndex + triangle.attributeIndex[2]];

			float f0 = 1.0f / (deltaUV0.x * deltaUV1.y - deltaUV1.x * deltaUV0.y);
			float f1 = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
			float f2 = 1.0f / (deltaUV2.x * deltaUV0.y - deltaUV0.x * deltaUV2.y);

			glm::vec3 t0 = f0 * (deltaUV1.y * edge0 - deltaUV0.y * edge1);
			glm::vec3 t1 = f1 * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
			glm::vec3 t2 = f2 * (deltaUV0.y * edge2 - deltaUV2.y * edge0);

			glm::vec3 b0 = f0 * (-deltaUV1.x * edge0 + deltaUV0.x * edge1);
			glm::vec3 b1 = f1 * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);
			glm::vec3 b2 = f2 * (-deltaUV0.x * edge2 + deltaUV2.x * edge0);

			intersections[path_index].surfaceTangent = t0 * (1.0f - baryCoords.x - baryCoords.y) + t1 * baryCoords.x + t2 * baryCoords.y;
			intersections[path_index].surfaceBitangent = b0 * (1.0f - baryCoords.x - baryCoords.y) + b1 * baryCoords.x + b2 * baryCoords.y;
		}
        if (materials[geoms[hit_geom_index].materialid].emissiveTextureId != -1) {
			const glm::vec2& uv0 = emissiveUvs[mesh.emissiveUvIndex + triangle.attributeIndex[0]];
			const glm::vec2& uv1 = emissiveUvs[mesh.emissiveUvIndex + triangle.attributeIndex[1]];
			const glm::vec2& uv2 = emissiveUvs[mesh.emissiveUvIndex + triangle.attributeIndex[2]];
			intersections[path_index].emissiveUvs = uv0 * (1.0f - baryCoords.x - baryCoords.y) + uv1 * baryCoords.x + uv2 * baryCoords.y;
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
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* textObjs,
	cudaTextureObject_t envMapTexObj,
	glm::vec3* normals,
	glm::vec3* albedos
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
	PathSegment& pathSegment = pathSegments[idx];

    if (intersection.t <= 0.0f) {
		glm::vec3 envColor(0.0f);
        if (envMapTexObj != 0) {
            // Calculate normalized texture coordinates (u, v) based on ray direction
            float u = 0.5f + atan2(pathSegment.ray.direction.z, pathSegment.ray.direction.x) / (2.0f * M_PI);
            float v = 0.5f - asin(pathSegment.ray.direction.y) / M_PI;

            // Sample the environment map texture
            float4 texColor = tex2D<float4>(envMapTexObj, u, v);
            envColor = glm::vec3(texColor.x, texColor.y, texColor.z);
        }

        // Set the path segment color to the sampled environment color
        pathSegment.color *= envColor;
        pathSegment.remainingBounces = 0;
        if (depth == 0) {
            albedos[pathSegment.pixelIndex] = envColor;
			normals[pathSegment.pixelIndex] = -pathSegment.ray.direction;
        }
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.color;

	if (material.baseColorTextureId != -1) {
		float4 texColor = tex2D<float4>(textObjs[material.baseColorTextureId], intersection.baseColorUvs.x, intersection.baseColorUvs.y);
		materialColor.x = texColor.x;
		materialColor.y = texColor.y;
		materialColor.z = texColor.z;
	}

    if (material.normalTextureId != -1) {
		float4 texNormal = tex2D<float4>(textObjs[material.normalTextureId], intersection.normalUvs.x, intersection.normalUvs.y);
		
        // Convert from [0, 1] space to [-1, 1]
        glm::vec3 normal = glm::normalize(2.0f * glm::vec3(texNormal.x, texNormal.y, texNormal.z) - 1.0f);
		glm::mat3 TBN = glm::mat3(intersection.surfaceTangent, intersection.surfaceBitangent, intersection.surfaceNormal);

		// Transform normal from tangent space to world space.
        intersection.surfaceNormal = glm::normalize(TBN * normal);
    }

    // If the material indicates that the object was a light, "light" the ray
    if (material.emittance > 0.0f) {
        materialColor = material.emissiveFactor;
        if (material.emissiveTextureId != -1) {
            float4 texColor = tex2D<float4>(textObjs[material.emissiveTextureId], intersection.emissiveUvs.x, intersection.emissiveUvs.y);
            materialColor.x = texColor.x;
            materialColor.y = texColor.y;
            materialColor.z = texColor.z;
        }

        materialColor *= material.emittance;

        if (depth == 0) {
			materialColor *= glm::dot(-pathSegment.ray.direction, intersection.surfaceNormal);
			albedos[pathSegment.pixelIndex] = materialColor;
			normals[pathSegment.pixelIndex] = intersection.surfaceNormal;
        }

		pathSegment.color *= materialColor;
		pathSegment.remainingBounces = 0;
        return;
    }

	glm::vec3 intersect = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
	scatterRay(pathSegment, intersect, intersection.surfaceNormal, material, rng);

    pathSegment.color *= materialColor;

	if (depth == 0) {
		normals[pathSegment.pixelIndex] = intersection.surfaceNormal;
		albedos[pathSegment.pixelIndex] = materialColor;
	}

    // Russian roulette termination (based on luminance)
	float p = 0.2126f * materialColor.x + 0.7152f * materialColor.y + 0.0722f * materialColor.z;
    p /= (p + 1);
	thrust::uniform_real_distribution<float> u01(0, 1);
    if (depth > 2) {
        if (u01(rng) > p) {
		    pathSegment.remainingBounces = 0;
            return;
	    }
        // Redistribute the "lost" energy by terminating paths to unterminated paths.
        pathSegment.color /= p;
    }
	
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, const PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= nPaths) return;
 
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
}

struct MaterialIdComparator
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const
    {
        return a.materialId < b.materialId;
    }
};

#ifdef NDEBUG
void applyOIDNDenoise(glm::vec3* output, glm::vec3* colors, glm::vec3* normals, glm::vec3* albedos, glm::ivec2 resolution)
{
    // Create OIDN filter
    oidn::FilterRef filter = oidnDevice.newFilter("RT"); // Generic ray tracing filter

	// Prefilter normals and albedos
    filter.setImage("color", normals, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", normals, oidn::Format::Float3, resolution.x, resolution.y);
    filter.commit();
    filter.execute();

    filter.set("hdr", true); // Assuming the image is in HDR 
    filter.setImage("color", albedos, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", albedos, oidn::Format::Float3, resolution.x, resolution.y);
    filter.commit();
    filter.execute();

    filter.setImage("color", colors, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", output, oidn::Format::Float3, resolution.x, resolution.y);
	filter.setImage("normal", normals, oidn::Format::Float3, resolution.x, resolution.y);
	filter.setImage("albedo", albedos, oidn::Format::Float3, resolution.x, resolution.y);
    filter.commit();
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (oidnDevice.getError(errorMessage) != oidn::Error::None)
    {
        fprintf(stderr, "Error: %s\n", errorMessage);
    }
}
#endif

__global__ void renormalizeNormals(int n, glm::vec3* normals)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= n) return;

	normals[index] = glm::normalize(normals[index]);
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, int maxIterations)
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    int depth = 0;
	while (num_paths > 0 && depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		// Shared memory will be used to maintain the stack of nodes to traverse in the BVH
		// Careful - shared memory is limited and with a deep enough BVH, this could fail to launch.
		// With 48KB of memory and a maximum of 1024 threads per block, each thread can get 48 bytes / 4 bytes per int = 12 integers
        // in the stack. A depth of 12 isn't great, but if we limit threads per block to 512 or 256, we get way more depth than we ever need.
        // (We could increase this by using a smaller int datatype for the stack, since we don't need to index very high).
		int sharedMemorySize = blockSize1d * (MAX_BVH_DEPTH + 1) * sizeof(int);
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d, sharedMemorySize >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_materials,
            dev_vertices,
            dev_meshes,
			dev_triangles,
            dev_meshNormals,
            dev_baseColorUvs,
            dev_normalUvs,
			dev_emissiveUvs,
			dev_bvhNodes
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // Before shading, sort by material to increase memory coherency and reduce warp divergence.
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
			MaterialIdComparator()
        );

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
			dev_texture_objects,
			dev_envMapTexObj,
            dev_normals,
            dev_albedos
        );

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

		// Compact streams with thrust partition (reorders paths so that terminated paths are at the end)
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end,
            [] __device__(const PathSegment& segment) {
                return segment.remainingBounces != 0;
            }
        );

		checkCUDAError("remove terminated paths");
        cudaDeviceSynchronize();

		num_paths = dev_path_end - dev_paths;
        ++depth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

#ifdef NDEBUG
    // Apply OIDN denoising
    if (iter == maxIterations) {
        renormalizeNormals<<<numBlocksPixels, blockSize1d>>> (pixelcount, dev_normals);
        applyOIDNDenoise(dev_image, dev_image, dev_normals, dev_albedos, cam.resolution);
	}
#endif

	// Send image to OpenGL for display
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, (iter == maxIterations));

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
