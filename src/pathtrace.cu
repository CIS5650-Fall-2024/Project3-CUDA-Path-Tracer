#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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
//For with obj mesh only
#define OBJ 1
#define BVH 0

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


__host__ __device__ glm::vec3 sampleTexture(const Texture& texture, const glm::vec2& uv) {
    int u = uv.x * texture.width;
    int v = (1.0f - uv.y) * texture.height;

    u = glm::clamp(u, 0, texture.width - 1);
    v = glm::clamp(v, 0, texture.height - 1);

    int index = (v * texture.width + u) * texture.channels;

    if (texture.channels == 3) {
        return glm::vec3(
            texture.data[index] / 255.0f,
            texture.data[index + 1] / 255.0f,
            texture.data[index + 2] / 255.0f);
    }

    if (texture.channels == 4) {
        return glm::vec3(
            texture.data[index] / 255.0f,
            texture.data[index + 1] / 255.0f,
            texture.data[index + 2] / 255.0f);
    }

    return glm::vec3(1.0f);
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
// ...
//Add for mesh
//static Triangle** dev_triangle_ptrs = NULL;
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
unsigned char* dev_texture_data = NULL;
static Texture* dev_normals = NULL;
static Texture* dev_normals_data = NULL;
//BVH
static BVHNode* dev_bvhNodes = NULL;

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
    checkCUDAError("pathtraceInit");
    // TODO: initialize any extra device memeory you need
    // Allocate memory for all the triangles on the device
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    // cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < scene->textures.size(); ++i) {
        Texture& texture = scene->textures[i];
        unsigned char* dev_texture_data;

        // Allocate memory for the texture's pixel data
        cudaMalloc(&dev_texture_data, texture.width * texture.height * texture.channels * sizeof(unsigned char));

        // Copy pixel data to the device
        cudaMemcpy(dev_texture_data, texture.data, texture.width * texture.height * texture.channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Update the device pointer in the texture struct
        texture.data = dev_texture_data;

        // Copy the updated texture to device memory
        cudaMemcpy(&dev_textures[i], &texture, sizeof(Texture), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_normals, scene->normals.size() * sizeof(Texture));
    for (size_t i = 0; i < scene->normals.size(); ++i) {
		Texture& normal = scene->normals[i];
		unsigned char* dev_normals_data;

		// Allocate memory for the texture's pixel data
		cudaMalloc(&dev_normals_data, normal.width * normal.height * normal.channels * sizeof(unsigned char));

		// Copy pixel data to the device
		cudaMemcpy(dev_normals_data, normal.data, normal.width * normal.height * normal.channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Update the device pointer in the texture struct
		normal.data = dev_normals_data;

		// Copy the updated texture to device memory
		cudaMemcpy(&dev_normals[i], &normal, sizeof(Texture), cudaMemcpyHostToDevice);
	}

    cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInitmesh");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    // Free all the triangle data on the device
    cudaFree(dev_triangles);
    cudaFree(dev_textures);
    cudaFree(dev_normals);
   // cudaFree(dev_bvhNodes);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, float aperture, float focal)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::uniform_real_distribution<float> u02(0, 1);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)(x + u01(rng)) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + u02(rng)) - (float)cam.resolution.y * 0.5f)
        );

        //Depth of field
        // Reference from: https://blog.demofox.org/2018/07/04/pathtraced-depth-of-field-bokeh/
        if (aperture > 0.0f) {
			// Generate a random point on the lens
            float angle = u01(rng) * 2 * PI;
            float radius = sqrt(u01(rng));
            glm::vec2 offset = glm::vec2(radius * cos(angle), radius * sin(angle)) * aperture;
            glm::vec3 sensorPlanePosition = cam.position + cam.right * offset[0] + cam.up * offset[1];
            segment.ray.origin = sensorPlanePosition;
            glm::vec3 focalPoint = segment.ray.origin + focal * segment.ray.direction;
            segment.ray.direction = glm::normalize(focalPoint - sensorPlanePosition);
		}

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
    // Pass in the triangles
    Triangle* triangles)
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
        glm::vec2 tmp_uv = glm::vec2(0.0);
        glm::vec2 uv;
        glm::vec3 tmp_tant = glm::vec3(0.0f);
        glm::vec3 tmp_bitant = glm::vec3(0.0f);
        glm::vec3 tangent;
        glm::vec3 bitangent;


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
            else if (geom.type == MESH) {

                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, triangles, tmp_uv, tmp_tant, tmp_bitant);
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
                tangent = tmp_tant;
                bitangent = tmp_bitant;
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
            //deault is -1, if changed then it is with texture
            intersections[path_index].textureid = geoms[hit_geom_index].textureid;
            intersections[path_index].outside = outside;
            intersections[path_index].normalid = geoms[hit_geom_index].normalid;
            intersections[path_index].tangent = tangent;
            intersections[path_index].bitangent = bitangent;
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

#if OBJ
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* textures,
    Texture* normalMaps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];

    if (idx >= num_paths) {
        return;
    }

    if (idx < num_paths) {
        // if the intersection exists...
        if (intersection.t > 0.0f) {
            // Set up RNG for random number generation
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Retrieve the material properties at the intersection
            Material material = materials[intersection.materialId];
            glm::vec3 surfaceColor = material.color;

            // Pass outside to the scatterRay function, to check if the ray is inside or outside the object
            bool outside = intersection.outside;

            // Check if the intersection has a texture, if so, sample the texture
            glm::vec3 texCol = glm::vec3(-1.0f);
            bool hasTexture = false;
            if (intersection.textureid != -1) {
                Texture texture = textures[intersection.textureid];
                texCol = sampleTexture(texture, intersection.uv);
                hasTexture = true;
            }

            // Check if the intersection has a normal map, if so, sample the normal map
            glm::vec3 tangentNorm = glm::vec3(-1.0f);
            if (intersection.normalid != -1) {
                Texture normalMap = normalMaps[intersection.normalid];
                tangentNorm = sampleTexture(normalMap, intersection.uv);
                tangentNorm = glm::normalize(tangentNorm * 2.0f - 1.0f);
                glm::mat3 TBN = glm::mat3(intersection.tangent, intersection.bitangent, intersection.surfaceNormal);
                glm::vec3 worldNormal = glm::normalize(TBN * tangentNorm);
                intersection.surfaceNormal = worldNormal;
            }

            // If the material is light (emssive) then stop the path
            if (material.emittance > 0.0f) {
                pathSegment.color *= (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
                return;
            }
            else {
                // Russian Roulette
                float prob = glm::max(material.color.r, glm::max(material.color.g, material.color.b));
                // Make sure no ray terminate too early
                prob = glm::max(prob, 0.3f);
                if (u01(rng) >= prob) {
                    pathSegment.color = glm::vec3(0.0f);
                    pathSegment.remainingBounces = 0;
                }
                else {
                    // Calculate the intersection point
                    glm::vec3 origin = getPointOnRay(pathSegment.ray, intersection.t);
                    // Scatter the ray and update the pathSegment color
                    scatterRay(pathSegment, origin, intersection.surfaceNormal, material, rng, outside, texCol, hasTexture);
                    pathSegment.color /= prob;
                }
            }
        }
        else {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}
#else
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];

    if (idx >= num_paths) {
        return;
    }

    // Check for a valid intersection
    if (idx < num_paths) {
        if (intersection.t > 0.0f) {
            // Set up RNG for random number generation
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Retrieve the material properties at the intersection
            Material material = materials[intersection.materialId];
            bool outside = intersection.outside;
            glm::vec3 surfaceColor = material.color;

            glm::vec3 texCol = glm::vec3(-1.0f);

            // If the material is emissive (i.e., a light source), light the ray
            if (material.emittance > 0.0f) {
                pathSegment.color *= (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
                return;
            }
            // Otherwise, handle reflection or diffuse lighting
            else {
                glm::vec3 bsdf = glm::vec3(0.0f);
                // Calculate the intersection point
                glm::vec3 origin = getPointOnRay(pathSegment.ray, intersection.t);
                scatterRay(pathSegment, origin, intersection.surfaceNormal, material, rng, outside, texCol, false);
                //pathSegment.color *= material.color;
            }
        }
        else {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}
#endif
//test
__global__ void getCumulativeColor(int num_paths, PathSegment* pathSegments, glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];
        if (pathSegment.remainingBounces <= 0)
        {
            atomicAdd(&image[pathSegment.pixelIndex].x, pathSegment.color.x);
            atomicAdd(&image[pathSegment.pixelIndex].y, pathSegment.color.y);
            atomicAdd(&image[pathSegment.pixelIndex].z, pathSegment.color.z);
        }
    }
}


__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}


struct is_path_terminated
{
    __host__ __device__
        bool operator()(const PathSegment& path)
    {
        return path.remainingBounces <= 0;
    } 
};

struct is_ray_hit
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& intersection)
    {
        return intersection.t > 0.0f;
    }
};

struct sort_by_material_id
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};


struct get_material_id {
    __host__ __device__
        int operator()(const ShadeableIntersection& intersection) const {
        return intersection.materialId;
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
    float aperture;
    float focal;
    if(guiData != NULL)
	{
		aperture = guiData->gAperture;
		focal = guiData->gFocal;
	}
	else
	{
        // not use depth of field
		aperture = -1.0f;
		focal = -1.0f;
	}

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, aperture, focal);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    // without stream compaction test
   // while (depth < 7)
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
            dev_triangles
            );
        checkCUDAError("computeIntersections error");
        cudaDeviceSynchronize();

        depth++;
#if 1
        // Compact paths and intersections together using zip_iterator
        thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
        thrust::device_ptr<ShadeableIntersection> thrust_intersections(dev_intersections);

        // Define a zip iterator to combine paths and intersections
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(thrust_paths, thrust_intersections));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(thrust_paths + num_paths, thrust_intersections + num_paths));

        // Compact both `dev_paths` and `dev_intersections` based on `is_ray_hit` predicate
        auto zip_new_end = thrust::copy_if(
            thrust::device,
            begin, end,  // Input range (combined paths and intersections)
            thrust_intersections, // Check `dev_intersections` for hit
            begin,            // Output for filtered pairs
            is_ray_hit()          // Predicate checking if ray hit anything
        );

        num_paths = thrust::get<0>(zip_new_end.get_iterator_tuple()) - thrust_paths;
        dev_path_end = dev_paths + num_paths;

        if (num_paths <= 0 || depth > traceDepth) {
            iterationComplete = true;
        }
#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        //sort by material id
        thrust::device_vector<int> materialIds(num_paths);
        thrust::transform(dev_intersections, dev_intersections + num_paths, materialIds.begin(), get_material_id());

        thrust::sort_by_key(
            materialIds.begin(), materialIds.end(),
            thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections))
        );

        cudaDeviceSynchronize();
#if OBJ   
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_normals
            );
        cudaDeviceSynchronize();
#else
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );
        cudaDeviceSynchronize();
#endif
#if 1
        // Add color to the image before stream compaction
        getCumulativeColor << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths,
            dev_paths,
            dev_image
            );
        cudaDeviceSynchronize();

        //iterationComplete = true; // TODO: should be based off stream compaction results.

        // Stream compaction
        auto* new_end2 = thrust::remove_if(thrust::device, dev_paths, dev_path_end, is_path_terminated());
        cudaDeviceSynchronize();
        dev_path_end = new_end2;
        num_paths = dev_path_end - dev_paths;
        // printf("Depth: %d, Number of active paths: %d\n", depth, num_paths);

        if (num_paths <= 0 || depth > traceDepth) {
            iterationComplete = true;
        }
#endif
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }


    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
