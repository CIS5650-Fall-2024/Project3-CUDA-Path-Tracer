#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <OpenImageDenoise/oidn.hpp>
#include <thrust/transform.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define MATERIAL_SORT 0
#define DENOISE 1

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
// ...
static int* dev_materialIds = NULL;
static int* dev_pixelIds = NULL;
thrust::device_ptr<int> thrust_materialIds;
thrust::device_ptr<PathSegment> thrust_pathSegments;
thrust::device_ptr<ShadeableIntersection> thrust_intersections;

static glm::vec3* dev_image_postProcess = NULL;
static Vertex* dev_vertices = NULL;
static Texture* dev_textures = NULL;
static EnvTexture* dev_envMap = NULL;
static std::vector<Texture> tmp_textures;
static EnvTexture tmpEnvText;
static glm::vec3* dev_surfaceNormals = NULL;
static glm::vec3* dev_surfaceAlbedo = NULL;

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

    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
    cudaMemset(dev_materialIds, -1, pixelcount * sizeof(int));

    cudaMalloc(&dev_pixelIds, pixelcount * sizeof(int));
    cudaMemset(dev_pixelIds, 0, pixelcount * sizeof(int));

    cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

    tmp_textures.resize(hst_scene->textures.size());
    for (size_t i = 0; i < hst_scene->textures.size(); i++) {
        cudaMalloc(&tmp_textures[i].imgData, hst_scene->textures[i].width * hst_scene->textures[i].height * hst_scene->textures[i].channel * sizeof(unsigned char));
        cudaMemcpy(tmp_textures[i].imgData, hst_scene->textures[i].imgData, hst_scene->textures[i].width * hst_scene->textures[i].height * hst_scene->textures[i].channel * sizeof(unsigned char), cudaMemcpyHostToDevice);
        tmp_textures[i].width = hst_scene->textures[i].width;
        tmp_textures[i].height = hst_scene->textures[i].height;
        tmp_textures[i].channel = hst_scene->textures[i].channel;
    }
    cudaMalloc(&dev_textures, tmp_textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, tmp_textures.data(), tmp_textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    cudaMalloc(&tmpEnvText.imgData, hst_scene->environmentMap.width * hst_scene->environmentMap.height * hst_scene->environmentMap.channel * sizeof(float));
    cudaMemcpy(tmpEnvText.imgData, hst_scene->environmentMap.imgData, hst_scene->environmentMap.width * hst_scene->environmentMap.height * hst_scene->environmentMap.channel * sizeof(float), cudaMemcpyHostToDevice);
    tmpEnvText.width = hst_scene->environmentMap.width;
    tmpEnvText.height = hst_scene->environmentMap.height;
    tmpEnvText.channel = hst_scene->environmentMap.channel;
    tmpEnvText.isTextureValid = hst_scene->environmentMap.isTextureValid;
    tmpEnvText.color = hst_scene->environmentMap.color;
    tmpEnvText.brightness = hst_scene->environmentMap.brightness;
    cudaMalloc(&dev_envMap, sizeof(EnvTexture));
    cudaMemcpy(dev_envMap, &tmpEnvText, sizeof(EnvTexture), cudaMemcpyHostToDevice);


    cudaMalloc(&dev_image_postProcess, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_postProcess, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_surfaceNormals, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_surfaceNormals, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_surfaceAlbedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_surfaceAlbedo, 0, pixelcount * sizeof(glm::vec3));

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
    cudaFree(dev_materialIds);
    cudaFree(dev_pixelIds);
    cudaFree(dev_vertices);
    for (size_t i = 0; i < tmp_textures.size(); i++) {
        cudaFree(tmp_textures[i].imgData);
    }
    cudaFree(tmpEnvText.imgData);
    cudaFree(dev_envMap);
    cudaFree(dev_textures);
    cudaFree(dev_image_postProcess);
    cudaFree(dev_surfaceNormals);
    cudaFree(dev_surfaceAlbedo);
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
        int pixelIdx = y * cam.resolution.x + x;
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, pixelIdx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float jitter_x = u01(rng) - 0.5f;
        float jitter_y = u01(rng) - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter_x)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter_y)
        );

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
    glm::vec3* surfaceNormals,
    int vertex_size,
    Vertex* vertices,
    Material* materials,
    Texture* textures
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths && pathSegments[path_index].remainingBounces > 0)
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
        glm::vec2 tmp_uv = glm::vec2(0.0);

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
            else if (geom.type == CUSTOM) 
            {
                t = customMeshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, vertex_size, vertices, materials, textures, outside);
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
            intersections[path_index].outside = outside;
            intersections[path_index].uv = uv;
            if (depth == 0) {
                surfaceNormals[path_index] = glm::normalize(normal);
            }
        }
    }
}

__global__ void getAlbedos(int num_paths, glm::vec3* albedos, ShadeableIntersection* intersections, Material* materials, Texture* textures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = intersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            Material material = materials[intersection.materialId];
            albedos[idx] = material.baseColorTextIdx == -1 ? material.color : getColorFromTexture(intersection.uv, textures[material.baseColorTextIdx]);
        }
    }
}

__global__ void getMaterialIds(int num_paths, ShadeableIntersection* intersections, int* materialIds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        materialIds[idx] = intersections[idx].materialId;
    }
}

__global__ void getPixelId(int num_paths, PathSegment* pathSegments, int* pixelIds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        pixelIds[idx] = pathSegments[idx].pixelIndex;
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
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* textures,
    EnvTexture* envMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    PathSegment& pathSegment = pathSegments[idx];
    if (idx >= num_paths) {
        return;
    }

    if (pathSegment.remainingBounces <= 0)
    {
        return;
    }
    if (idx < num_paths)
    {
        ShadeableIntersection  intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.baseColorTextIdx == -1 ? material.color : getColorFromTexture(intersection.uv, textures[material.baseColorTextIdx]);

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                return;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, intersection.uv,intersection.outside, material, materialColor, rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            if (envMap->isTextureValid == 1) {
                glm::vec3 direction = glm::normalize(pathSegment.ray.direction);
                float u = 0.5f + (atan2(direction.z, direction.x) / (2.0f * PI));
                float v = 0.5f - (asin(direction.y) / PI);

                int x = static_cast<int>(u * envMap->width);
                int y = static_cast<int>(v * envMap->height);

                pathSegments[idx].color *= glm::clamp(glm::vec3(envMap->imgData[3 * (y * envMap->width + x) + 0],
                envMap->imgData[3 * (y * envMap->width + x) + 1],
                envMap->imgData[3 * (y * envMap->width + x) + 2]), glm::vec3(0.0), glm::vec3(1.0)) * envMap->brightness;
            }
            else {
                pathSegments[idx].color *= envMap->color;
            }

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

__global__ void updateColor(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces <= 0) {
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
    }
}

struct isTerminated {
    __host__ __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces != 0;  // Remove if the path is terminated
    }
};

void applyDenoising(std::vector<glm::vec3>& image, std::vector<glm::vec3>& normals, std::vector<glm::vec3>& albedos, int width, int height) {
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a filter for denoising
    oidn::FilterRef filter = device.newFilter("RT");

    // Create an OIDN buffer for the input image (which must be Float3 format for RGB)
    oidn::BufferRef colorBuffer = device.newBuffer(width * height * 3 * sizeof(float));
    oidn::BufferRef albedoBuffer = device.newBuffer(width * height * 3 * sizeof(float));
    oidn::BufferRef normalBuffer = device.newBuffer(width * height * 3 * sizeof(float));
    filter.setImage("color", colorBuffer, oidn::Format::Float3, width, height);
    //filter.setImage("albedo", albedoBuffer, oidn::Format::Float3, width, height);
    //filter.setImage("normal", normalBuffer, oidn::Format::Float3, width, height);
    filter.setImage("output", colorBuffer, oidn::Format::Float3, width, height);
    filter.set("hdr", true);
    filter.commit();

    float* colorPtr = (float*)colorBuffer.getData();
    float* albedoPtr = (float*)albedoBuffer.getData();
    float* normalPtr = (float*)normalBuffer.getData();
    for (int i = 0; i < width * height; i++) {
        colorPtr[i * 3] = image[i].r;
        colorPtr[i * 3 + 1] = image[i].g;
        colorPtr[i * 3 + 2] = image[i].b;
        albedoPtr[i * 3] = albedos[i].r;
        albedoPtr[i * 3 + 1] = albedos[i].g;
        albedoPtr[i * 3 + 2] = albedos[i].b;
        normalPtr[i * 3] = normals[i].x;
        normalPtr[i * 3 + 1] = normals[i].y;
        normalPtr[i * 3 + 2] = normals[i].z;
    }
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "OIDN Error: " << errorMessage << std::endl;
    }

    for (int i = 0; i < width * height; i++) {
        image[i] = glm::vec3(colorPtr[i * 3], colorPtr[i * 3 + 1], colorPtr[i * 3 + 2]);
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

    int vertex_size = hst_scene->vertices.size();
    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

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
            dev_surfaceNormals,
            vertex_size,
            dev_vertices,
            dev_materials,
            dev_textures
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        
        if (depth == 0) {
            getAlbedos << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_surfaceAlbedo, dev_intersections, dev_materials, dev_textures);
        }

        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if MATERIAL_SORT
        getMaterialIds << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_materialIds);
        cudaDeviceSynchronize();
        checkCUDAError("material sort");

        thrust::device_ptr<int> thrust_materalIds(dev_materialIds);
        thrust::device_ptr<PathSegment> thrust_pathSegments(dev_paths);
        thrust::device_ptr<ShadeableIntersection> thrust_intersections(dev_intersections);

        thrust::sort_by_key(thrust_materalIds, thrust_materalIds + num_paths, thrust::make_zip_iterator(thrust::make_tuple(thrust_pathSegments, thrust_intersections)));
        cudaDeviceSynchronize();
#endif


        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_envMap
            );

        cudaDeviceSynchronize();


        PathSegment* new_end = thrust::stable_partition(
            thrust::device,         // Execution policy for device-side computation
            dev_paths,              // Start of the array (device pointer)
            dev_paths + num_paths,  // End of the array (device pointer)
            isTerminated()          // Predicate to remove terminated paths
        );
        cudaDeviceSynchronize();


        num_paths = new_end - dev_paths;


        if (num_paths <= 0 || depth >= traceDepth) {
            iterationComplete = true;   // All paths have been terminated
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    cudaMemcpy(dev_image_postProcess, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

#if DENOISE
    std::vector<glm::vec3> host_image(pixelcount);
    std::vector<glm::vec3> host_normals(pixelcount);
    std::vector<glm::vec3> host_albedos(pixelcount);
    cudaMemcpy(host_image.data(), dev_image_postProcess, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy for denoising");
    cudaDeviceSynchronize();

    cudaMemcpy(host_normals.data(), dev_surfaceNormals, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy for denoising");
    cudaDeviceSynchronize();

    cudaMemcpy(host_albedos.data(), dev_surfaceAlbedo, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy for denoising");
    cudaDeviceSynchronize();

    applyDenoising(host_image, host_normals, host_albedos, cam.resolution.x, cam.resolution.y);

    cudaMemcpy(dev_image_postProcess, host_image.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy after denoising");
    cudaDeviceSynchronize();
#endif // DENOISE

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image_postProcess);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image_postProcess,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}