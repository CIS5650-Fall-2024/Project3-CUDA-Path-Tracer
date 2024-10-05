#pragma once

#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <stack>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <OpenImageDenoise/oidn.hpp>
#if LOG_PERF
#include <fstream>
#endif

#include "bvh.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "samplers.h"
#include "flags.h"

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
static BVH::Node* dev_nodes = NULL;
static Material* dev_materials = NULL;
static glm::vec4* dev_textures = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// OIDN
oidn::DeviceRef device;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static glm::vec3* dev_albedo_norm = NULL;
static glm::vec3* dev_normal_norm = NULL;
static glm::vec3* dev_output = NULL;

std::ofstream streamCompactionLogFile;

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
    if (hst_scene->restart) {
        cudaMemcpy(dev_image, scene->state.image.data(), scene->state.image.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        hst_scene->restart = false;
    }
    else
        cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_nodes, scene->nodes.size() * sizeof(BVH::Node));
    cudaMemcpy(dev_nodes, scene->nodes.data(), scene->nodes.size() * sizeof(BVH::Node), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(glm::vec4));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");

    device = oidn::newDevice(oidn::DeviceType::CUDA);
    device.commit();

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo_norm, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo_norm, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal_norm, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal_norm, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_output, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_output, 0, pixelcount * sizeof(glm::vec3));

#if LOG_PERF
    streamCompactionLogFile.open("streamcompactionlog.txt");
#endif
}

void pathtraceFree()
{
#if LOG_PERF
    streamCompactionLogFile.close();
#endif

    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_nodes);
    cudaFree(dev_materials);
    cudaFree(dev_textures);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    cudaFree(dev_albedo);
    cudaFree(dev_normal);
    cudaFree(dev_albedo_norm);
    cudaFree(dev_normal_norm);
    cudaFree(dev_output);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int iterModSamplesX, int iterModSamplesY, float invSampleWidth, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        PathSegment& segment = pathSegments[index];

        // Antialiasing: jitter rays by [0,1] to generate uniformly random direction distribution per pixel
        // Stratified sampling, shoot jittered ray at (i % (sampleWidth * sampleWidth))-th grid
        glm::vec2 jitter = glm::vec2((iterModSamplesX + u01(rng)) * invSampleWidth, (iterModSamplesY + u01(rng)) * invSampleWidth);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - 0.5f + jitter[0] - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - 0.5f + jitter[1] - (float)cam.resolution.y * 0.5f)
        );

        // Depth of Field, construct a new direction pointing the same direction but from new origin AND at focal length away
        glm::vec2 apertureOrigin = cam.apertureSize * randomOnUnitCircle(rng);
        segment.ray.origin = cam.position + cam.right * apertureOrigin[0] + cam.up * apertureOrigin[1];
        segment.ray.direction = glm::normalize(segment.ray.direction * cam.focalLength + cam.position - segment.ray.origin);

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
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
    int numPaths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geomsSize,
    BVH::Node* nodes,
    int nodesSize,
    int rootIdx,
    glm::vec4* textures,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= numPaths) return;

    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    glm::vec2 texCoord;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_texCoord;
    
    // Early terminate if no intersection with the root node
    glm::vec2 times;
    if (bboxIntersectionTest(nodes[rootIdx].bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times) < 0.f) {
        intersections[path_index].t = -1.0f;
        intersections[path_index].materialId = -1;
        return;
    }

    // BVH intersection hierarchy
    // Don't render details / far away objects beyond 1024 hierarchical levels
    // 1024 is an arbitrary depth limit since dynamic array sizing is bad
    int nodeStack[1024];
    memset(nodeStack, 0, 1024);
    int nodeStackFinger = 0;

    nodeStack[nodeStackFinger] = rootIdx;
    nodeStackFinger++;

    while (nodeStackFinger > 0 && nodeStackFinger < 1024) {
        int currIdx = nodeStack[nodeStackFinger - 1];
        const BVH::Node& node = nodes[currIdx];
        nodeStackFinger--;

        bool hit = bboxIntersectionTest(node.bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times) > 0.f;
        if (!hit || hit && times[0] > t_min) continue;

        if (node.l == node.r) {
            for (int i = node.start; i < node.start + node.size; i++) {
                Geom& geom = geoms[i];
                if (geom.type == CUBE)
                {
                    t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_texCoord, outside);
                }
                else if (geom.type == SPHERE)
                {
                    t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_texCoord, outside);
                }
                else if (geom.type == TRIANGLE)
                {
                    t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_texCoord, textures, outside);
                }

                // Compute the minimum t from the intersection tests to determine what
                // scene geometry object was hit first.
                if (t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                    texCoord = tmp_texCoord;
                }
            }
            continue;
        }
      
        // Check intersection with left and right children
        bool hitL = bboxIntersectionTest(nodes[node.l].bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times) > 0.f;
        bool hitR = bboxIntersectionTest(nodes[node.r].bbox, pathSegment.ray, tmp_intersect, tmp_normal, outside, times) > 0.f;

        if (hitL && hitR) {
            // Both hit
            nodeStack[nodeStackFinger] = node.l;
            nodeStackFinger++;
            nodeStack[nodeStackFinger] = node.r;
            nodeStackFinger++;
        } else if (hitR) {
            nodeStack[nodeStackFinger] = node.r;
            nodeStackFinger++;
        } else if (hitL) {
            nodeStack[nodeStackFinger] = node.l;
            nodeStackFinger++;
        }
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
        intersections[path_index].materialId = -1;
    }
    else
    {
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
        intersections[path_index].texCoord = texCoord;
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
    int numPaths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec4* textures,
    ImageTextureInfo bgTextureInfo,
    glm::vec3* dev_img,
    glm::vec3* albedos,
    glm::vec3* normals,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;
    if (pathSegments[idx].remainingBounces < 0) return;

    // Set up the RNG
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

    // Scatter the ray at intersecting point and perform bsdf evaluation
    scatterRay(pathSegments[idx], 
        pathSegments[idx].ray.origin + 
        pathSegments[idx].ray.direction * shadeableIntersections[idx].t, 
        shadeableIntersections[idx],
        materials[shadeableIntersections[idx].materialId],
        textures,
        bgTextureInfo,
        rng,
        dev_img,
        albedos,
        normals,
        depth);
}

__global__ void averageOIDNArrays(
    int iter,
    int numPaths,
    glm::vec3* albedos,
    glm::vec3* normals,
    glm::vec3* albedos_norm,
    glm::vec3* normals_norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    albedos_norm[idx] = albedos[idx] / (float)iter;
    normals_norm[idx] = glm::normalize(normals[idx]);
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
    const int sampleWidth = hst_scene->state.sampleWidth;
    const float invSampleWidth = 1.0f / (float)sampleWidth;

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

    int iterModSamples = iter % (sampleWidth * sampleWidth);
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, 
                                                            iterModSamples % sampleWidth, 
                                                            iterModSamples / sampleWidth, 
                                                            invSampleWidth, 
                                                            traceDepth, 
                                                            dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    ShadeableIntersection* dev_intersections_end = dev_intersections + pixelcount;
    int remaining_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // Terminate iteration once maximum trace depth has been reached or if no valid rays remain
    while (remaining_paths && depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, remaining_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (remaining_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            remaining_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_nodes,
            hst_scene->nodes.size(),
            hst_scene->bvhRootIdx,
            dev_textures,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if PATHTRACE_CONTIGUOUS_MATERIALID
        dev_intersections_end = dev_intersections + remaining_paths;
        // Sort arrays in decreasing materialId order and contiguous memory
        // We do not terminate paths here as we need to apply environmet mapping
        thrust::stable_sort_by_key(
            thrust::device, 
            dev_intersections, 
            dev_intersections_end, 
            dev_paths,
            [] __device__(const ShadeableIntersection & si1, const ShadeableIntersection & si2) { return si1.materialId > si2.materialId; });
#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            remaining_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            hst_scene->bgTextureInfo,
            dev_image,
            dev_albedo,
            dev_normal,
            depth
        );
        checkCUDAError("shade material error");

#if STREAM_COMPACTION
        // Compaction : Terminate paths with no more remaining bounces
        dev_path_end = thrust::stable_partition(
            thrust::device, 
            dev_paths, 
            dev_path_end, 
            [] __device__ (const PathSegment& ps) { return ps.remainingBounces > -1; });

        remaining_paths = dev_path_end - dev_paths;
#endif

#if LOG_PERF
        streamCompactionLogFile << remaining_paths << "\n";
#endif

        if (guiData)
        {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

#if USE_OIDN
    // Perform basic denoising for the real time renders for performance
    // Based on https://github.com/RenderKit/oidn?tab=readme-ov-file#basic-denoising-c11-api
    
    // Normalize albedo and normal arrays (currently summed up iter times)
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    averageOIDNArrays << <numBlocksPixels, blockSize1d >> > (iter, pixelcount, dev_albedo, dev_normal, dev_albedo_norm, dev_normal_norm);

    // Create a filter for denoising a beauty (color) image using optional auxiliary images too
    // This can be an expensive operation, so try no to create a new filter for every image!
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  dev_image,  oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // beauty
    filter.setImage("albedo", dev_albedo_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // auxiliary
    filter.setImage("normal", dev_normal_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // auxiliary
    filter.setImage("output", dev_output, oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // denoised beauty
    filter.set("hdr", true); // beauty image is HDR
    filter.commit();

    // Filter the beauty image
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_output);
#else
    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
#endif
}

void retrieveRenderBuffer() {
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

#if USE_OIDN
    // Perform denoising with prefiltering for the saved images
    // Based on https://github.com/RenderKit/oidn?tab=readme-ov-file#denoising-with-prefiltering-c11-api
    
    // Create a filter for denoising a beauty (color) image using prefiltered auxiliary images too
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  dev_image,  oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // beauty
    filter.setImage("albedo", dev_albedo_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // auxiliary
    filter.setImage("normal", dev_normal_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // auxiliary
    filter.setImage("output", dev_output, oidn::Format::Float3, cam.resolution.x, cam.resolution.y); // denoised beauty
    filter.set("hdr", true); // beauty image is HDR
    filter.set("cleanAux", true); // auxiliary images will be prefiltered
    filter.commit();

    // Create a separate filter for denoising an auxiliary albedo image (in-place)
    oidn::FilterRef albedoFilter = device.newFilter("RT"); // same filter type as for beauty
    albedoFilter.setImage("albedo", dev_albedo_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    albedoFilter.setImage("output", dev_albedo_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    albedoFilter.commit();

    // Create a separate filter for denoising an auxiliary normal image (in-place)
    oidn::FilterRef normalFilter = device.newFilter("RT"); // same filter type as for beauty
    normalFilter.setImage("normal", dev_normal_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    normalFilter.setImage("output", dev_normal_norm, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    normalFilter.commit();

    // Prefilter the auxiliary images
    albedoFilter.execute();
    normalFilter.execute();

    // Filter the beauty image
    filter.execute();

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_output,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif

    checkCUDAError("pathtrace");
}

void resetRenderBuffer() {
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
}