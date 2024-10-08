#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "obj_utils.h"

#define ERRORCHECK 1

// whether to sort the materials
#define SORTMATERIAL 1
#define RUSSIANROULETTE 0
#define RUSSIANROULETTE_THRESHOLD 0.6f

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define OCTREESEARCH 1
#define APPROXOCTREELEAFSIZE 1

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
static Tri* dev_triangle_set = NULL;
static OctTreeRegion* dev_octRegions = NULL;
static int numOctRegions = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

    //initialized triangle set for mesh rendering
    int num_triangles = scene->triangles.size();
    cudaMalloc(&dev_triangle_set, num_triangles * sizeof(Tri));
    cudaMemcpy(dev_triangle_set, scene->triangles.data(), scene->triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice);

    #if OCTREESEARCH == 1
    float numleafrough = ((float)(num_triangles)) / APPROXOCTREELEAFSIZE;
    int tree_depth = floor((int)(log2(numleafrough) / 3.0f));
    if(tree_depth < 0){
        tree_depth = 0;
    }
    int tree_size = ((1 << 3 * (tree_depth + 1)) - 1) / 7;
    numOctRegions = tree_size;
    OctTreeRegion* octRegions = new OctTreeRegion[tree_size];
    std::cout << "before octree tri array, with tree depth: " << tree_depth << " and tree size " << tree_size << endl;
    cudaDeviceSynchronize();
    //formOctreeTrianglesArray(dev_triangle_set, num_triangles, octRegions, tree_size, 0, tree_depth, 0, 1, 0);
    formOctreeTrianglesArray(dev_triangle_set, num_triangles, octRegions, 0, tree_depth, 0, 0);
    cudaMalloc(&dev_octRegions, tree_size * sizeof(OctTreeRegion));
    cudaMemcpy(dev_octRegions, octRegions, tree_size * sizeof(OctTreeRegion), cudaMemcpyHostToDevice);
    for(int i = 0; i < tree_size; i++){
        OctTreeRegion region = octRegions[i];
        Geom bbox = region.boundingBox;
        std::cout << "oct region " << i << endl;
        std::cout << "    scale: " << bbox.scale.x << " " << bbox.scale.y << " " <<  bbox.scale.z << endl;
        std::cout << "    trans: " << bbox.translation.x << " " << bbox.translation.y << " " <<  bbox.translation.z << endl;
        std::cout << "    depth: " << region.depth << endl;
        std::cout << "    length: " << region.length << endl;
    }
    #endif
    cout << "Num Triangles: " << num_triangles << " sizeof triangle set: " << num_triangles * sizeof(Tri) << endl;

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

    cudaFree(dev_triangle_set);
    cudaFree(dev_octRegions);
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
    int idx = x + y * (blockDim.x * gridDim.x);



    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
        thrust::uniform_real_distribution<float> rnggen(0.0f, 1.0f);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        float random_scale = 0.006f;
        //float random_scale = 0.0f;
        glm::vec3 jitter_vec = cam.view;
        glm::vec3 view_right = glm::normalize(glm::cross(cam.view, cam.up));
        glm::vec3 view_up = glm::normalize(glm::cross(view_right, cam.view));
        float rand_right = rnggen(rng) - 0.5;
        float rand_up = rnggen(rng) - 0.5;
        jitter_vec += - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f);
        jitter_vec += - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f);
        jitter_vec += rand_right * view_right * random_scale;
        jitter_vec += rand_up * view_up * random_scale;
        segment.ray.direction = glm::normalize(jitter_vec);

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
    Tri* triangle_set,
    int num_triangles,
    OctTreeRegion* octRegions,
    int num_octs)
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

        //naive parse through mesh
        #if OCTREESEARCH == 1
        FloatInt mesh_intersect = getTriIntersectionOctTree(pathSegment.ray, triangle_set, num_triangles, octRegions, num_octs,t_min);
        #else
        FloatInt mesh_intersect = getTriIntersectionLinear(pathSegment.ray, triangle_set, num_triangles,t_min);
        #endif
        if(mesh_intersect.i != -1){
            intersections[path_index].t = mesh_intersect.f;
            intersections[path_index].materialId = -1;
            intersections[path_index].triangleId = mesh_intersect.i;
            intersections[path_index].surfaceNormal = triangle_set[mesh_intersect.i].normal;
        } else if (hit_geom_index != -1){
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].triangleId = -1;
        } else {
            intersections[path_index].t = -1.0f;
        }
    }
}
__global__ void shadeBSDFMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Tri* triangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        //if(pathSegments[idx].remainingBounces > 0){
        if (intersection.t > 0) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

            Material material; 
            if(intersection.triangleId != -1){
                Tri tri = triangles[intersection.triangleId];
                material.color = tri.color;
                material.emittance = tri.emittance;
                material.hasReflective = tri.reflectivity;
            } else {
                material = materials[intersection.materialId];
            }
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            // Otherwise, do some lighting computation. 
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= materialColor * material.emittance;
                pathSegments[idx].remainingBounces = 0;
            } else {

                glm::vec3 intersectOrigin = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
                intersectOrigin += intersection.surfaceNormal * 0.00001f;
                scatterRay(pathSegments[idx], intersectOrigin, intersection.surfaceNormal, material, rng);
                pathSegments[idx].remainingBounces -= 1;
                //Reflective
                //pathSegments[idx].ray.direction -= 2.0f * intersection.surfaceNormal * glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal);

                if (pathSegments[idx].remainingBounces == 0){
                    pathSegments[idx].color = glm::vec3(0.0f);
                }

                #if RUSSIANROULETTE
                float color_norm = glm::length(pathSegments[idx].color);
                if(color_norm < RUSSIANROULETTE_THRESHOLD){
                    float prob_keep = color_norm / RUSSIANROULETTE_THRESHOLD;
                    thrust::uniform_real_distribution<float> rnggen(0.0f, 1.0f);
                    if(rnggen(rng) < prob_keep){
                        pathSegments[idx].color /= prob_keep;
                    } else {
                        pathSegments[idx].remainingBounces = 0;
                        pathSegments[idx].color *= 0;
                    }
                }

                #endif
            }
        } else {
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
        //} 
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



struct my_predicate
{
    __host__ __device__
    bool operator()(const PathSegment& pathSegment) {
        return pathSegment.remainingBounces != 0;
    }
};

struct mat_sort_predicate
{
    __host__ __device__ 
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) 
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
    int num_paths = pixelcount;
    int num_paths_remaining = num_paths;
    thrust::device_ptr<PathSegment> dev_paths_device_ptr(dev_paths);

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    int num_triangles = hst_scene->triangles.size();
    bool iterationComplete = false;
    //std::cout << "start loop" << endl;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths_remaining * sizeof(ShadeableIntersection));
        cudaDeviceSynchronize();

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths_remaining + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
        //computeIntersections<<<1, 1>>> (
            depth,
            num_paths_remaining,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_triangle_set,
            num_triangles,
            dev_octRegions,
            numOctRegions
        );
        cudaDeviceSynchronize();
        checkCUDAError("intersection check");

        //Sort the materials
        #if SORTMATERIAL == 1
        thrust::sort_by_key(dev_intersections, dev_intersections + num_paths_remaining, dev_paths_device_ptr, mat_sort_predicate());
        #endif

        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeBSDFMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths_remaining,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_triangle_set
        );
        cudaDeviceSynchronize();

        //Stream Compation with thrust::remove_if
        num_paths_remaining = thrust::partition(dev_paths_device_ptr, dev_paths_device_ptr + num_paths_remaining, my_predicate()) - dev_paths_device_ptr; 
        //std::cout << "num paths remaining: " << num_paths_remaining << endl;
        //std::cout << "num paths " << num_paths << endl;
        cudaDeviceSynchronize();

        iterationComplete = num_paths_remaining == 0; 

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
    checkCUDAError("check");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
