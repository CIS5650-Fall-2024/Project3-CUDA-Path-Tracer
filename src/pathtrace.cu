#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "light.h"

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

// post process the image
__device__ inline glm::vec3 postProcess(glm::vec3 x)
{
	
#if TONE_MAPPING_ACES
	x = ACESFilm(x);
#elif TONE_MAPPING_REINHARD
	x = Reinhard(x); 
#endif
    return x;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool isPostProcess)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / static_cast<float>(iter);

		if (isPostProcess)
		{
			pix = postProcess(pix);
		}

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

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
static glm::vec3* dev_image_post = NULL;
static PathSegment* dev_paths = NULL;
static PathSegment* dev_terminated_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static thrust::device_ptr<PathSegment> dev_thrust_paths;
static thrust::device_ptr<PathSegment> dev_thrust_terminated_paths;
static cudaTextureObject_t envMap = NULL;

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
	cudaMalloc(&dev_image_post, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_post, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_terminated_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	dev_thrust_paths = thrust::device_ptr<PathSegment>(dev_paths);
	dev_thrust_terminated_paths = thrust::device_ptr<PathSegment>(dev_terminated_paths);
	if (scene->envMap != NULL)
	    envMap = scene->envMap->texObj;

	//cudaMalloc(&dev_materials, hst_scene->materials.size() * sizeof(Material));
	//cudaMemcpy(dev_materials, hst_scene->materials.data(), hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	//cudaMalloc(&dev_geoms, hst_scene->geoms.size() * sizeof(Geom));
	//cudaMemcpy(dev_geoms, hst_scene->geoms.data(), hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	//cudaMalloc(&dev_triangles, hst_scene->triangles.size() * sizeof(Triangle));
	//cudaMemcpy(dev_triangles, hst_scene->triangles.data(), hst_scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	//BVHAccel::LinearBVHNode* nodes = hst_scene->getLBVHRoot();
 //   if (nodes)
 //   {
	//	cudaMalloc(&dev_nodes, hst_scene->bvh->bvhNodes * sizeof(LinearBVHNode));
	//	cudaMemcpy(dev_nodes, nodes, hst_scene->bvh->bvhNodes * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
	//}


	checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_intersections);
	cudaFree(dev_terminated_paths);
	cudaFree(dev_image_post);
	//cudaFree(dev_materials);
	//cudaFree(dev_geoms);
	//cudaFree(dev_triangles);
	//cudaFree(dev_nodes);
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
        segment.color = glm::vec3(0.f);

        // TODO: implement antialiasing by jittering the ray
		float pixelX = float(x);
		float pixelY = float(y);
        
#ifdef JITTER
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-JITTER, JITTER);
		pixelX += u01(rng);
		pixelY += u01(rng);
#endif
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (pixelX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (pixelY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
		segment.throughput = glm::vec3(1.0f);
		segment.accumLight = glm::vec3(0.0f);
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
    Triangle* dev_triangles,
	int triangles_size,
	LinearBVHNode* dev_nodes,
    ShadeableIntersection* intersections,
    int num_lights,
    int iter)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
		ShadeableIntersection& intersection = intersections[path_index];


        float t_min = FLT_MAX;
#ifdef USE_BVH
        // bvh intersection
		ShadeableIntersection bvhIntersection;
		bvhIntersection.t = -1.0f;
        bvhIntersection.hitBVH = 0;
        if (BVHIntersect(pathSegment.ray, dev_nodes, dev_triangles, &bvhIntersection) && bvhIntersection.t > 0.0f && bvhIntersection.t < t_min)
            intersection = bvhIntersection;
#ifdef DEBUG_BVH
        else intersection = bvhIntersection;
#endif

#else
        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        int hit_geom_index = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        for (int i = 0; i < triangles_size; i++)
        {
            Triangle triangle = dev_triangles[i];
            float t = triangle.intersect(pathSegment.ray);
            /*if (t > 0 && t < t_min)
            {
                t_min = t;
                normal = triangle.getNormal();
            }*/
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = triangle.getNormal();
            }
        }
        if (hit_geom_index == -1)
        {
            intersection.t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersection.t = t_min;
            intersection.materialId = dev_triangles[hit_geom_index].materialid;
            intersection.surfaceNormal = normal;
        }
#endif
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
        intersection.directLightId = num_lights == 1 ? 0 : thrust::uniform_int_distribution<int>(0, num_lights - 1)(rng);
    }
}


__global__ void shadeMaterialNaive(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
    cudaTextureObject_t envMap,
    int num_lights,
    LinearBVHNode* dev_nodes,
    Triangle* dev_triangles,
    Light* dev_lights,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment pathSegment = pathSegments[idx];
#ifdef DEBUG_BVH
        //scatterRay(pathSegment, getPointOnRay(pathSegment.ray, intersection.t), intersection.t, intersection.surfaceNormal, intersection.uv, material, rng);
        pathSegment.color += glm::vec3(intersection.hitBVH);
        pathSegment.throughput = glm::vec3(1.0);
        pathSegment.remainingBounces = 0;
#else
        //thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
        //thrust::uniform_real_distribution<float> u01(0, 1);

        //Material material = materials[intersection.materialId];
        //glm::vec3 materialColor = material.color;

        //MIS(pathSegment, intersection, getPointOnRay(pathSegment.ray, intersection.t), material, rng, num_lights, dev_nodes, dev_triangles, dev_lights, envMap);

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
                //pathSegment.color += (materialColor * material.emittance);
				pathSegment.accumLight += pathSegment.throughput * (materialColor * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            else
            {
				//scatterRay(pathSegment, intersection, getPointOnRay(pathSegment.ray, intersection.t), material, rng, num_lights, dev_nodes, dev_triangles, dev_lights, envMap);
				MIS(pathSegment, intersection, getPointOnRay(pathSegment.ray, intersection.t), material, rng, num_lights, dev_nodes, dev_triangles, dev_lights, envMap, depth);
            }

        }
        else {
			//pathSegment.color += getEnvironmentalRadiance(pathSegment.ray.direction, envMap);
            pathSegment.accumLight += pathSegment.throughput * getEnvironmentalRadiance(pathSegment.ray.direction, envMap);
            pathSegment.remainingBounces = 0;
        }
#endif

		pathSegments[idx] = pathSegment;
    }
}

 
// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
		glm::vec3 col = iterationPath.color;

#ifdef DEBUG_THROUGHPUT
        image[iterationPath.pixelIndex] += glm::length(iterationPath.throughput) / 1.732;
#elif defined DEBUG_RADIANCE
		image[iterationPath.pixelIndex] += glm::length(iterationPath.color) / 1.732;
#else
        image[iterationPath.pixelIndex] += iterationPath.accumLight;
        //image[iterationPath.pixelIndex] += iterationPath.color * iterationPath.throughput;
#endif


    }
}



struct isValid
{
    __host__ __device__ bool operator() (const PathSegment& segment) {
        return segment.isTerminated();
    }

};

// first look at intersection, then direct lighting idex, then material id
struct sortByIsectDIMat
{
	__host__ __device__ bool operator() (const ShadeableIntersection& a, const ShadeableIntersection& b) const {
		if (a.directLightId < b.directLightId) return true;
		else if (a.directLightId == b.directLightId && a.materialId < b.materialId) return true;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, uchar4* pbo_post, int frame, int iter)
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

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
	int curr_paths = pixelcount;
	thrust::device_ptr<PathSegment> dev_thrust_terminated_paths_end = dev_thrust_terminated_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;


	float totalElapsedTime = 0.0f;
	int iteration = 0;
    float totalPaths = 0;
    while (!iterationComplete)
    {
		totalPaths += curr_paths;
        depth++;
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (curr_paths + blockSize1d - 1) / blockSize1d;

        iteration++;
        
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            curr_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            hst_scene->triangles.size(),
            dev_nodes,
            dev_intersections,
			hst_scene->lights.size(),
			iter
        );
        

       
		// sort by intersection, then direct lighting index, then material id
		//thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + curr_paths, dev_paths, sortByIsectDIMat());
        

        cudaEventRecord(gpuInfo->start);
		shadeMaterialNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			curr_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			envMap,
			envMap == NULL ? hst_scene->lights.size() : hst_scene->lights.size() + 1,
			dev_nodes,
			dev_triangles,
			dev_lights,
            depth
            );
        cudaEventRecord(gpuInfo->stop);
        cudaEventSynchronize(gpuInfo->stop);
        float elapsedTime = 0.0f;
        cudaEventElapsedTime(&elapsedTime, gpuInfo->start, gpuInfo->stop);
        totalElapsedTime += elapsedTime;

        // Implement thrust stream compaction
		dev_thrust_terminated_paths_end = thrust::copy_if(dev_thrust_paths, dev_thrust_paths + curr_paths, dev_thrust_terminated_paths_end, isValid()); // copy terminated paths to the terminated paths array
		auto paths_end = thrust::remove_if(dev_thrust_paths, dev_thrust_paths + curr_paths, isValid());

        curr_paths = paths_end - dev_thrust_paths;
        iterationComplete = (curr_paths <= 0 || depth >= traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
	totalElapsedTime /= iteration;
	gpuInfo->elapsedTime = totalElapsedTime;
	gpuInfo->averagePathPerBounce = depth;

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	int num_terminated_paths = dev_thrust_terminated_paths_end - dev_thrust_terminated_paths;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_terminated_paths, dev_image, dev_terminated_paths);
    checkCUDAError("trace one bounce");
#ifdef POSTPROCESS
	cudaMemcpy(dev_image_post, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo_post, cam.resolution, iter, dev_image_post, true);
	cudaMemcpy(hst_scene->state.image.data(), dev_image_post,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else 
    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, false);
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif

    checkCUDAError("pathtrace");
}
