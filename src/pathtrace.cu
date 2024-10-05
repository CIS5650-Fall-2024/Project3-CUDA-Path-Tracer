#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define SORT 0
#define AABB 1
#define SAA 1
#define DOF 0
#define RANDTEXTURE 0

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
static bool* dev_paths_terminated = NULL;
static PathSegment* dev_buffer = NULL;
static int* dev_intersection_key = NULL;
static int* dev_path_key = NULL;
static Vertex* dev_vertices = NULL;
static Texture* dev_textures = NULL;
static Texture* dev_norm_textures = NULL;

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
	cudaMalloc(&dev_paths_terminated, pixelcount * sizeof(bool));
	cudaMalloc(&dev_buffer, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_intersection_key, pixelcount * sizeof(int));
	cudaMalloc(&dev_path_key, pixelcount * sizeof(int));

	if (scene->vertices.size() > 0)
	{
		cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
		cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
	}
	if (scene->textures.size() > 0)
	{
		cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
		cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
	}
	if (scene->norm_textures.size() > 0)
	{
		cudaMalloc(&dev_norm_textures, scene->norm_textures.size() * sizeof(Texture));
		cudaMemcpy(dev_norm_textures, scene->norm_textures.data(), scene->norm_textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
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
	cudaFree(dev_paths_terminated);
	cudaFree(dev_buffer);
	cudaFree(dev_intersection_key);
	cudaFree(dev_path_key);
	cudaFree(dev_vertices);
	cudaFree(dev_textures);
	cudaFree(dev_norm_textures);

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

		// TODO: implement antialiasing by jittering the ray
				// create a random number generator
		thrust::default_random_engine rng{
			makeSeededRandomEngine(iter, index, traceDepth)
		};

		// use uniform distribution to generate new points
		thrust::normal_distribution<float> n0(-0.1f, 0.1f);
		thrust::uniform_real_distribution<float> u0(-0.3f, 0.3f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));

#if SAA
		// anti aliasing
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u0(rng))
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u0(rng)));
#endif

#if DOF
		//DOF
		float pLensx = n0(rng) * cam.lensRadius;
		float pLensy = n0(rng) * cam.lensRadius;

		segment.ray.origin = cam.position + glm::vec3(pLensx, pLensy, 0);
		float d_z = glm::dot(cam.view, segment.ray.direction);
		// ray(ft) should also consider cam.position
		segment.ray.direction = glm::normalize(cam.f / d_z * segment.ray.direction + cam.position - segment.ray.origin);

#endif
		segment.pixelIndex = index;
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
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
	int geoms_size,
	ShadeableIntersection* intersections,
	Vertex* vertices)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)

	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv(-1.f);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal(0.0f);
		glm::vec2 tmp_uv(0.0f);

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

			// add mesh geom intersect check
			else if (geom.type == MESH)
			{
				//perform culling
#if AABB		if (aabbIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside))
#endif
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, vertices);

				//if (aabbIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside))
				/*{
					t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, vertices);
				}*/
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
			// The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
			intersections[path_index].textureOffset = geoms[hit_geom_index].texture_offset;
			intersections[path_index].textureCount = geoms[hit_geom_index].texture_count;
			intersections[path_index].normTextureOffset = geoms[hit_geom_index].norm_texture_offset;
			intersections[path_index].normTextureCount = geoms[hit_geom_index].norm_texture_count;
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
	Material* materials,
	Texture* textures,
	Texture* norm_textures)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		//this is for no stream compaction
		if (pathSegments[idx].remainingBounces == 0) return;

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) // if the intersection exists...
		{
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];

			// texture mapping
			glm::vec2 segmentUV = intersection.uv;
#if !RANDTEXTURE			
			if (intersection.textureCount > 0 && segmentUV.x > 0 && segmentUV.y > 0)
			{
				int textureOffset = intersection.textureOffset;
				int height = textures[textureOffset].height;
				int width = textures[textureOffset].width;

				const float x = segmentUV.x * width;
				const float y = (1.0f - segmentUV.y) * height;

				int floor_y = static_cast<int>(glm::floor(x));
				int floor_x = static_cast<int>(glm::floor(y));

				//material.color *= textures[textureOffset + floor_y + width * floor_x].color;

				material.color = textures[textureOffset + floor_y + width * floor_x].color;

				float gamma = 1.8f;
				material.color.r = glm::pow(material.color.r, 1.0f / gamma);
				material.color.g = glm::pow(material.color.g, 1.0f / gamma);
				material.color.b = glm::pow(material.color.b, 1.0f / gamma);
			}
#endif
#if RANDTEXTURE
			if (intersection.textureCount > 0 && segmentUV.x > 0 && segmentUV.y > 0)
				material.color = glm::vec3(u01(rng), u01(rng), u01(rng));
#endif

			////normal mapping
			if (intersection.normTextureCount > 0 && segmentUV.x > 0 && segmentUV.y > 0)
			{
				int normTextureOffset = intersection.normTextureOffset;
				int height = norm_textures[normTextureOffset].height;
				int width = norm_textures[normTextureOffset].width;

				const float x = segmentUV.x * width;
				const float y = (1.0f - segmentUV.y) * height;

				int floor_y = static_cast<int>(glm::floor(x));
				int floor_x = static_cast<int>(glm::floor(y));

				glm::vec3 tex_normal = norm_textures[normTextureOffset + floor_y + width * floor_x].color * 2.0f - 1.0f;

				glm::vec3 up = glm::abs(intersection.surfaceNormal.y) < 0.999f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
				glm::vec3 T = glm::normalize(glm::cross(up, intersection.surfaceNormal));
				glm::vec3 B = glm::cross(intersection.surfaceNormal, T);
				glm::mat3 TBN = glm::mat3(T, B, intersection.surfaceNormal);
				intersection.surfaceNormal = TBN * tex_normal;
			}


			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				// no more reflect after light
				pathSegments[idx].remainingBounces = 0;

			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
				//ray(t)=origin+t×direction
				glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
				scatterRay(
					pathSegments[idx],
					intersect,
					intersection.surfaceNormal,
					material,
					rng);

				//float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				//pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			//also terminate this path!
			pathSegments[idx].remainingBounces = 0;
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

// get mask for terminated paths
__global__ void getTerminated(int nPaths, const PathSegment* paths, bool* paths_terminated)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		//if no bounces, flag as terminated
		PathSegment iterationPath = paths[index];
		paths_terminated[index] = (iterationPath.remainingBounces == 0);
	}
}

// load ternimated paths to buffer
__global__ void loadBuffer(int pixels, const PathSegment* paths, PathSegment* buffer)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < pixels)
	{
		PathSegment iterationPath = paths[index];
		if (iterationPath.remainingBounces == 0)
			buffer[iterationPath.pixelIndex] = paths[index];
	}
}

//
__global__ void getKeyByMaterial(int nPaths, const ShadeableIntersection* intersections, int* intersection_key, int* path_key)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		int key = intersections[index].materialId;
		intersection_key[index] = key;
		path_key[index] = key;
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
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
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections,
			dev_vertices);
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

		thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
#if SORT
		getKeyByMaterial << <numblocksPathSegmentTracing, blockSize1d >> >
			(num_paths, dev_intersections, dev_intersection_key, dev_path_key);
		cudaDeviceSynchronize();

		//sort by material id
		thrust::device_ptr<ShadeableIntersection> thrust_intersections(dev_intersections);

		thrust::device_ptr<int> thrust_intersection_key(dev_intersection_key);
		thrust::device_ptr<int> thrust_path_key(dev_path_key);
		thrust::sort_by_key(thrust_intersection_key, thrust_intersection_key + num_paths, thrust_intersections);
		thrust::sort_by_key(thrust_path_key, thrust_path_key + num_paths, thrust_paths);
#endif
		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_textures,
			dev_norm_textures);
		cudaDeviceSynchronize();

		// load terminated paths to buffer before stream compaction, or next step they will be erased in remove_if()!
		loadBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths, dev_buffer);
		cudaDeviceSynchronize();

		// get mask for terminated paths
		getTerminated << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths, dev_paths_terminated);
		cudaDeviceSynchronize();

		//cout << num_paths << endl;
		//stream compaction: remove PathSegments with no more bounces
		thrust::device_ptr<bool>thrust_paths_terminated(dev_paths_terminated);
		auto thrust_paths_end = thrust::remove_if(thrust_paths, thrust_paths + num_paths, thrust_paths_terminated, thrust::identity<bool>());
		num_paths = thrust_paths_end - thrust_paths;

		iterationComplete = (num_paths == 0) || (depth == traceDepth);
		//if (iterationComplete) cout << "iteration complete!" << endl;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_buffer);
	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
