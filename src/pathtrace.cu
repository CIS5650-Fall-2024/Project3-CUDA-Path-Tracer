#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <OpenImageDenoise/oidn.hpp>

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

__constant__ float cie_1964_dev_data[471][3];

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

 //Function to help with the conversion from wavelength to RGB
__host__ __device__ glm::vec3 wl_rgb(int wavelength) {
	wavelength -= 360;
    glm::vec3 xyz = (wavelength < 0 || wavelength > 470) ? glm::vec3(0.f) : glm::vec3(cie_1964_dev_data[wavelength][0], cie_1964_dev_data[wavelength][1], cie_1964_dev_data[wavelength][2]);
    float x = xyz.x;
    float y = xyz.y;
    float z = xyz.z;

    glm::vec3 rgb;
    rgb.r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    rgb.g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    rgb.b = (0.0556434 * x - 0.2040259 * y + 1.0572252 * z) * 3.9f;
    return glm::clamp(rgb, 0.f, 1.f);
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
        // Reinhardt Operator
        pix /= 1.f + pix;

        // Gamma Correction
        float gamma = 1.9f;
        pix = glm::pow(pix, glm::vec3(1.0f / gamma));

        glm::ivec3 color;
#if !DENOISE
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#else
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);
#endif

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
// ...
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
static glm::vec3* dev_texture_data = NULL;
static BVHNode* dev_bvhNodes = NULL;

#if DENOISE
// OIDN Stuff
static oidn::DeviceRef oidn_device;
static glm::vec3* dev_oidn_color = NULL;
static glm::vec3* dev_oidn_color_normalized = NULL;
static glm::vec3* dev_oidn_albedo = NULL;
static glm::vec3* dev_oidn_normal = NULL;
static glm::vec3* dev_oidn_albedo_normalized = NULL;
static glm::vec3* dev_oidn_normal_normalized = NULL;
static glm::vec3* dev_oidn_output = NULL;
#endif

static glm::vec4* dev_environmentMap = NULL;
static glm::vec2* dev_environmentMapSize = NULL;


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

	// Constant Memory for CIE 1964 data for Wavelength Dispersion
	cudaMemcpyToSymbol(cie_1964_dev_data, cie_1964_host_data, 471 * sizeof(glm::vec3));

#if DENOISE
    // OIDN Memory Allocation

    oidn_device = oidnNewDevice(OIDN_DEVICE_TYPE_CUDA);
	oidn_device.commit();

	cudaMalloc(&dev_oidn_color, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_color, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_oidn_color_normalized, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_color_normalized, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_oidn_albedo, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_albedo, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_oidn_normal, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_normal, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_oidn_albedo_normalized, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_albedo_normalized, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_oidn_normal_normalized, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_normal_normalized, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_oidn_output, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_oidn_output, 0, pixelcount * sizeof(glm::vec3));
#endif


    checkCUDAError("pathtraceInit");

    // For Meshes

    if (scene->triangles.size() > 0) {

        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
        cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_texture_data, scene->textureData.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_texture_data, scene->textureData.data(), scene->textureData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

        int num_nodes = scene->bvhNodes.size();
        cudaMalloc(&dev_bvhNodes, num_nodes * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), num_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

    }

    if (scene->envMap != NULL) {

        glm::vec2 size = glm::vec2(scene->envMap->width, scene->envMap->height);

        cudaMalloc(&dev_environmentMapSize, sizeof(glm::vec2));
        cudaMemcpy(dev_environmentMapSize, &(size), sizeof(glm::vec2), cudaMemcpyHostToDevice);
        checkCUDAError("pathtraceInit");
        if (size != glm::vec2(0, 0)) {
            cudaMalloc(&dev_environmentMap, scene->envMapData.size() * sizeof(glm::vec4));
            cudaMemcpy(dev_environmentMap, scene->envMapData.data(), scene->envMapData.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);
        }
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
    // TODO: clean up any extra device memory you created
	cudaFree(dev_triangles);
	cudaFree(dev_textures);
	cudaFree(dev_texture_data);
    cudaFree(dev_bvhNodes);

#if DENOISE
	cudaFree(dev_oidn_color);
	cudaFree(dev_oidn_color_normalized);
	cudaFree(dev_oidn_albedo);
	cudaFree(dev_oidn_normal);
	cudaFree(dev_oidn_albedo_normalized);
	cudaFree(dev_oidn_normal_normalized);
	cudaFree(dev_oidn_output);
#endif

	cudaFree(dev_environmentMap);
	cudaFree(dev_environmentMapSize);

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

        // TODO: Depth Of Field

#if ANTIALIASING
        // antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u1_5(-0.5, 0.5);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u1_5(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u1_5(rng) - (float)cam.resolution.y * 0.5f)
        );
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
        // wavelength setting
#if DISPERSION
        thrust::uniform_real_distribution<float> u01(0, 1);
        segment.waveLength = u01(rng) * 470 + 360;
		segment.color = 3.0f * wl_rgb(segment.waveLength);
#else
		segment.color = glm::vec3(1.0f);
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
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
    Triangle* tris,
    BVHNode* bvhnodes,
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
        int meshId = -1;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        int temp_meshId = -1;

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
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == MESH)
            {
#if BVH
                t = bvhMeshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tris, bvhnodes, temp_meshId);
#else 
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tris, geom.triangleStart, geom.triangleEnd);
#endif
            }
            else if (geom.type == SDF) {
                // Insert Scene SDF Function Here w/ Procedural Texture
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
                meshId = temp_meshId;
            }
        }

        if (hit_geom_index == -1)
        {   
            // TODO: ENV Mappig
            intersections[path_index].t = -1.0f;
			pathSegment.remainingBounces = 0;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
            // If we hit a mesh:
			if (meshId != -1) {
				intersections[path_index].texid = meshId;
                intersections[path_index].hasUV = geoms[meshId].usesUVs;
                if (geoms[meshId].usesUVs) {
                    intersections[path_index].texid = geoms[meshId].textureStart;
                }
                intersections[path_index].materialId = geoms[meshId].materialid;
            }
            else {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            }
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
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
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

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    int depth,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* textureMaps,
	glm::vec2* envMapSize,
	glm::vec4* envMap,
    glm::vec3* textureColors,
    glm::vec3* oidn_albedo_buffer,
    glm::vec3* oidn_normal_buffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
		// Check if the ray is terminated
		if (pathSegments[idx].remainingBounces <= 0)
        {
			return;
		}

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.f) // if the intersection exists...
        {
            glm::vec3 textureCol = glm::vec3(-1.0f);
            // Get the texture color
#if TEXTURING
            if (intersection.hasUV) {
                int x = glm::min(textureMaps[intersection.texid].width * intersection.uv.x, textureMaps[intersection.texid].width - 1.0f);
                int y = glm::min(textureMaps[intersection.texid].height * intersection.uv.y, textureMaps[intersection.texid].height - 1.0f);
                int idx = textureMaps[intersection.texid].width * y + x + textureMaps[intersection.texid].startIndex;
                textureCol = textureColors[idx];
            }
#endif
#if DENOISE
            if (depth == 1) {
                oidn_albedo_buffer[pathSegments[idx].pixelIndex] += intersection.hasUV ? textureCol : materials[intersection.materialId].color;
                oidn_normal_buffer[pathSegments[idx].pixelIndex] += 0.5f * (intersection.surfaceNormal + 1.0f);
            }
#endif

            // Set up the RNG
            Material material = materials[intersection.materialId];

            // If the material indicates that the object was a light, "light" the ray and terminate the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (material.color * material.emittance);
				pathSegments[idx].remainingBounces = 0;
            }
			// Otherwise, bounce the ray
            else {

                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
                thrust::uniform_real_distribution<float> u01(0, 1);

                // Get the ray
				Ray& ray = pathSegments[idx].ray;

				// Get the intersection point
				glm::vec3 intersect = ray.origin + ray.direction * intersection.t;

                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng, textureCol);

            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            //pathSegments[idx].color = glm::vec3(0.0f);

			//Environment Mapping
            if (envMapSize != NULL && envMap != NULL) {
                glm::vec3 dir = pathSegments[idx].ray.direction;
                float theta = acosf(dir.y), phi = atan2f(dir.z, dir.x);

                float u = (phi + PI) * 1.f / (2 * PI);
                float v = theta / PI;
                int tex_x_idx = glm::fract(u) * envMapSize[0].x;
                int tex_y_idx = glm::fract(v) * envMapSize[0].y;
                int tex_1d_idx = tex_y_idx * envMapSize[0].x + tex_x_idx;
                pathSegments[idx].color *= glm::vec3(envMap[tex_1d_idx]);
#if DENOISE
                if (depth == 1) {
					oidn_albedo_buffer[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
                }
#endif
            }
            else {
				pathSegments[idx].color = glm::vec3(0.0f);
            }

			pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void normalizeImages(int iteration, int pixel_count, glm::vec3* color_buffer, glm::vec3* normalized_color_buffer, glm::vec3* albedo_buffer, glm::vec3* normal_buffer, glm::vec3* albedo_normalized_buffer, glm::vec3* normal_normalized_buffer)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < pixel_count)
	{
		normalized_color_buffer[index] = color_buffer[index] / (float)iteration;
		albedo_normalized_buffer[index] = albedo_buffer[index] / (float)iteration;
		normal_normalized_buffer[index] = normal_buffer[index] / (float)iteration;
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

__global__ void blendImages(glm::vec3* image, glm::vec3* image2, glm::vec3* image3, int pixelCount, float fract, int iter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelCount) {
        image[idx] = image[idx] * fract + image2[idx] * (1.f - fract);
		if (image3 != NULL) {
            image3[idx] = image[idx] * (float)iter;
		}
    }
}

#if DENOISE
void denoise(const glm::vec2 resolution) {
    oidn::FilterRef filter = oidn_device.newFilter("RT");
    filter.setImage("color", dev_oidn_color_normalized, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("albedo", dev_oidn_albedo_normalized, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("normal", dev_oidn_normal_normalized, oidn::Format::Float3, resolution.x, resolution.y);
    filter.setImage("output", dev_oidn_output, oidn::Format::Float3, resolution.x, resolution.y);
    filter.set("hdr", true);
    filter.commit();

	oidn::FilterRef albedo_filter = oidn_device.newFilter("RT");
	albedo_filter.setImage("color", dev_oidn_albedo_normalized, oidn::Format::Float3, resolution.x, resolution.y);
	albedo_filter.setImage("output", dev_oidn_albedo_normalized, oidn::Format::Float3, resolution.x, resolution.y);
	albedo_filter.commit();

	oidn::FilterRef normal_filter = oidn_device.newFilter("RT");
	normal_filter.setImage("normal", dev_oidn_normal_normalized, oidn::Format::Float3, resolution.x, resolution.y);
	normal_filter.setImage("output", dev_oidn_normal_normalized, oidn::Format::Float3, resolution.x, resolution.y);
	normal_filter.commit();

	albedo_filter.execute();
	normal_filter.execute();

    filter.execute();
}
#endif

// Predicate for stream compaction
struct is_path_terminated
{
	__host__ __device__ bool operator()(const PathSegment& path)
	{
        return path.remainingBounces > 0;
	}
};

// Predicate for sorting by material
struct compare_material
{
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
	{
		return a.materialId < b.materialId;
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

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
			dev_triangles,
            dev_bvhNodes,
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

        /*shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> >(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );*/

#if MATERIAL_SORTING
		// Sort the paths by material
		thrust::device_ptr<ShadeableIntersection> dev_intersections_ptr(dev_intersections);
		thrust::device_ptr<PathSegment> dev_paths_ptr(dev_paths);
		thrust::stable_sort_by_key(dev_intersections_ptr, dev_intersections_ptr + num_paths, dev_paths_ptr, compare_material());
#endif

#if DENOISE
       shadeMaterial <<<numblocksPathSegmentTracing, blockSize1d >>> (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_environmentMapSize,
		    dev_environmentMap,
            dev_texture_data,
            dev_oidn_albedo,
            dev_oidn_normal);
#else
       shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
		    dev_environmentMapSize,
		    dev_environmentMap,
            dev_texture_data,
            nullptr,
            nullptr);
#endif

#if STREAM_COMPACTION
		// Stream compaction using thrust
        thrust::device_ptr<PathSegment> dev_compaction_paths(dev_paths);
		
		thrust::device_ptr<PathSegment> new_paths_end = thrust::stable_partition(thrust::device, dev_compaction_paths, dev_compaction_paths + num_paths, is_path_terminated());
		num_paths = new_paths_end.get() - dev_paths;
#endif

		iterationComplete = (depth >= traceDepth || num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
#if DENOISE
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_oidn_color, dev_paths);
#else
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);
#endif

    ///////////////////////////////////////////////////////////////////////////

#if DENOISE
	int denoise_iter = 1000;
	// Denoise the image
    if (iter < denoise_iter) {
        normalizeImages << <numBlocksPixels, blockSize1d >> > (iter, pixelcount, dev_oidn_color, dev_oidn_color_normalized, dev_oidn_albedo, dev_oidn_normal, dev_oidn_albedo_normalized, dev_oidn_normal_normalized);
        blendImages << <numBlocksPixels, blockSize1d >> > (dev_image, dev_oidn_color_normalized, NULL, pixelcount, 0.f, iter);
    }
    if (iter % denoise_iter == 0) {
        normalizeImages << <numBlocksPixels, blockSize1d >> > (iter, pixelcount, dev_oidn_color, dev_oidn_color_normalized, dev_oidn_albedo, dev_oidn_normal, dev_oidn_albedo_normalized, dev_oidn_normal_normalized);
        denoise(cam.resolution);
        blendImages << <numBlocksPixels, blockSize1d >> > (dev_image, dev_oidn_output, dev_oidn_color, pixelcount, 0.3f, iter);
    }
    else {
        normalizeImages << <numBlocksPixels, blockSize1d >> > (iter, pixelcount, dev_oidn_color, dev_oidn_color_normalized, dev_oidn_albedo, dev_oidn_normal, dev_oidn_albedo_normalized, dev_oidn_normal_normalized);
        blendImages << <numBlocksPixels, blockSize1d >> > (dev_image, dev_oidn_color_normalized, NULL, pixelcount, 0.5f, iter);
    }

#endif

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
}
