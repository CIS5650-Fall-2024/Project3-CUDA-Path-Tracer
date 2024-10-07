#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
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
static Geom* dev_geoms = NULL;
static Light* dev_lights = NULL;
static glm::vec3* dev_image = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
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
    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

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
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
        thrust::uniform_real_distribution<float> u01_2(-1, 1);
        // segment.ray.origin = cam.position;
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)(x + u01(rng)) - (float)cam.resolution.x * 0.5f)
            - cam.up * (cam.pixelLength.y) * ((float)(y + u01(rng)) - (float)cam.resolution.y * 0.5f)
        );
        auto p = glm::vec3(0, u01_2(rng), u01_2(rng));
        segment.ray.origin = (cam.defocus_angle <= 0) ? cam.position : (cam.position + p[1] * cam.defocus_disk_up + p[2] * cam.defocus_disk_right);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.endPath = false;
        segment.path_index = index;
        segment.IOR = 1.0f;
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
            intersections[path_index].intersect_point = intersect_point;
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


__global__ void shadeMaterial(
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
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].indirectColor += pathSegments[idx].throughput * (material.emittance);
                pathSegments[idx].remainingBounces = 0;
                return;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner

            // Russian Roulette for path termination
            float continueProbability = glm::min(0.99f, glm::max(materialColor.r, glm::max(materialColor.g, materialColor.b)));
            if (u01(rng) > continueProbability) {

                pathSegments[idx].remainingBounces = 0;
                pathSegments[idx].indirectColor = pathSegments[idx].throughput * glm::vec3(0.0f);
                return;
            }

            scatterRay(pathSegments[idx], intersection.intersect_point, intersection.surfaceNormal, material, rng);
            pathSegments[idx].throughput /= continueProbability;
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].indirectColor = pathSegments[idx].throughput * glm::vec3(0.0f);
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
        image[iterationPath.pixelIndex] += (iterationPath.directColor + iterationPath.indirectColor);
    }
}




struct invalid_intersection {

    __host__ __device__
        bool operator()(thrust::tuple<ShadeableIntersection, PathSegment> const& t) {
        return thrust::get<0>(t).t > 0.0f;
    }
};

struct MaterialCmp {
    __host__ __device__
        bool operator()(const ShadeableIntersection& m1, const ShadeableIntersection& m2) {
        return m1.materialId < m2.materialId;
    }
};


struct custom_predicate {

    __host__ __device__
        bool operator()(PathSegment const& t) {
        return t.remainingBounces >= 0;
    }
};

__global__ void DirectLighting(int iter, int num_paths, PathSegment* dev_paths,
    ShadeableIntersection* dev_intersections,
    Geom* dev_geoms, int geom_size, Material* dev_materials, Light* dev_light, int lightSource_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
    thrust::uniform_real_distribution<int> u01(0, lightSource_count - 1);
    thrust::uniform_real_distribution<float> u01_2(0, 1);
    float randomValue = u01_2(rng);
    if (idx >= num_paths)
        return;
    PathSegment& pathSegment = dev_paths[idx];
    glm::vec3 incident_vector = glm::normalize(pathSegment.ray.direction);
    ShadeableIntersection& intersection = dev_intersections[idx];
    float r;
    if (intersection.t >= 0.0)
    {
        Material material = dev_materials[intersection.materialId];
        int light_index = u01(rng);

        float t;
        Geom light = dev_geoms[dev_light[light_index].geom_id];
        float area = light.scale.x * light.scale.y;

        glm::vec3 intersect_point = intersection.intersect_point;
        glm::vec3 normal = intersection.surfaceNormal;
        glm::vec3 light_pos;
        glm::vec3 view_point = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
        glm::vec3 light_dir;
        float pdf;
        if (area <= 0.1)
        {
            light_pos = light.translation - glm::vec3(0, 0.01, 0);
            r = length(view_point - light_pos);
            light_dir = glm::normalize(light_pos - view_point);
            float NdotL = glm::max(glm::dot(normal, light_dir), 0.0f);
            glm::vec3 diffuse = material.color * dev_light[light_index].intensity * NdotL;
            pdf = 1.0f;
        }
        else
        {
            light_pos = light.translation - 0.5f * glm::vec3(light.scale.x, light.scale.y, 0);
            light_pos += light.scale.x * u01_2(rng) * glm::vec3(1.0f, 0, 0) + light.scale.y * u01_2(rng) * glm::vec3(0, 1, 0);
            r = length(light_pos - view_point);
            light_dir = normalize(light_pos - view_point);
            glm::vec3 nor = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(0.0f)));
            pdf = (r * r) / (area * glm::dot(light_dir, nor));
        }

        Ray shadow_ray;
        shadow_ray.origin = view_point + 0.001f * normal;
        shadow_ray.direction = light_dir;


        float t_min = FLT_MAX;
        Geom geom;
        if (material.emittance > 0)
        {
            pathSegment.directColor = material.color * dev_light[light_index].intensity;
            return;
        }
        bool isShadowed = false;
        bool outside = false;
        bool sphere = false;
        for (int i = 0; i < geom_size; i++)
        {
            if (i == dev_light[light_index].geom_id) continue;
            t = -1.0f;
            geom = dev_geoms[i];
            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;
            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, shadow_ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, shadow_ray, tmp_intersect, tmp_normal, outside);
            }

            if ((t < r - 0.001f) && (t > 0.0f))
            {
                isShadowed = true;
                break;
            }

        }
        if (!isShadowed)
        {
            float NdotL = glm::max(glm::dot(normal, light_dir), 0.0f);
            glm::vec3 diffuse = material.color * dev_light[light_index].intensity * NdotL;

            if (material.hasRefractive > 0)
                pathSegment.directColor = glm::vec3(0);
            else
                pathSegment.directColor = diffuse * dev_light[light_index].intensity / (r * r);

        }
        else {
            pathSegment.directColor = glm::vec3(0.0f, 0.0f, 0.0f);
        }
    }
    else
    {
        pathSegment.directColor = glm::vec3(0.0f, 0.0f, 0.0f);
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
    int actual_num_paths = dev_path_end - dev_paths;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

    while (!iterationComplete)
    {
        numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
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

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
        thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
        thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths,MaterialCmp());

         if(depth == 1)
             DirectLighting<<<numblocksPathSegmentTracing, blockSize1d>>>(iter,num_paths, dev_paths, dev_intersections, dev_geoms, hst_scene->geoms.size(), dev_materials,dev_lights,hst_scene->lights.size());
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials);
        
        // Stream compaction

        //dev_path_end = thrust::partition(thrust::device,dev_paths, dev_paths + num_paths, custom_predicate());
        num_paths = dev_path_end - dev_paths;
        // Stream compaction to remove invalid intersections and corresponding path segments
        // Update the number of paths                  
        if (num_paths == 0 || depth >= traceDepth)
            iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

    }
    // Assemble this iteration and apply it to the image
    finalGather << <numBlocksPixels, blockSize1d >> > (actual_num_paths, dev_image, dev_paths);


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
