#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define ANTIALIAS 1
#define DOF 0

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
static Vertex* dev_vertices = NULL;
static Texture* dev_textures = NULL;
static glm::vec4* dev_pixels = NULL;
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

    cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_pixels, scene->texturePixels.size() * sizeof(glm::vec4));
    cudaMemcpy(dev_pixels, scene->texturePixels.data(), scene->texturePixels.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_vertices);
    cudaFree(dev_textures);
    cudaFree(dev_pixels);

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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);

        float x_rng = 0.0f;
        float y_rng = 0.0f;

        if (ANTIALIAS)
        {
            // implement antialiasing by jittering the ray
            thrust::uniform_real_distribution<float> u05(-0.5, 0.5);

            x_rng = u05(rng);
            y_rng = u05(rng);
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + x_rng - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + y_rng - (float)cam.resolution.y * 0.5f)
        );
        
        if (DOF)
        {
            thrust::uniform_real_distribution<float> u01(0, 1);

            float angle = u01(rng) * TWO_PI;
            float r = u01(rng) * cam.lens_radius;
            glm::vec3 offset = r * (cos(angle) * cam.right + sin(angle) * cam.up);

            segment.ray.origin = cam.position + offset;
            segment.ray.direction = glm::normalize(cam.focal_dis / glm::dot(segment.ray.direction, cam.view) * segment.ray.direction - offset);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// Möller–Trumbore intersection
__host__ __device__ float triangleIntersect(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2,
    glm::vec3 rayOrigin, glm::vec3 rayDirection) {
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = glm::cross(rayDirection, edge2);
    a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return INFINITY;    // This ray is parallel to this triangle.
    }
    f = 1.0 / a;
    s = rayOrigin - p0;
    u = f * glm::dot(s, h);
    if (u < 0.0 || u > 1.0)
        return INFINITY;
    q = glm::cross(s, edge1);
    v = f * glm::dot(rayDirection, q);
    if (v < 0.0 || u + v > 1.0) {
        return INFINITY;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * glm::dot(edge2, q);
    if (t > EPSILON) {
        return t;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return INFINITY;
}

__host__ __device__ glm::vec3 barycentric(glm::vec3 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {
    glm::vec3 edge1 = t2 - t1;
    glm::vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return glm::vec3(S1 / S, S2 / S, S3 / S);
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
    ShadeableIntersection* intersections,
    Vertex* vertices,
    int vert_size)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int mat_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_tangent;

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
                mat_index = geoms[i].materialid;
                normal = tmp_normal;
            }
        }

        // check if intersects with model
        for (int vertIdx = 0; vertIdx < vert_size; vertIdx += 3)
        {
            Vertex& v1 = vertices[vertIdx];
            Vertex& v2 = vertices[vertIdx + 1];
            Vertex& v3 = vertices[vertIdx + 2];
            t = triangleIntersect(v1.pos, v2.pos, v3.pos, pathSegment.ray.origin, pathSegment.ray.direction);

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                mat_index = v1.materialid;
                glm::vec3 p = pathSegment.ray.origin + (pathSegment.ray.direction * t);
                glm::vec3 bary = barycentric(p, v1.pos, v2.pos, v3.pos); // Calculate barycentric coordinates
                normal = glm::normalize(bary.x * v1.nor + bary.y * v2.nor + bary.z * v3.nor); // Interpolate normals
                uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v3.uv; // Interpolate uv
                tmp_tangent = bary.x * v1.tangent + bary.y * v2.tangent + bary.z * v3.tangent;
            }
        }

        if (mat_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = mat_index;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].surfaceUV = uv;
            intersections[path_index].tangent = tmp_tangent;
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
    glm::vec4* pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces == 0) return;

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

            // overwrite color if using material texture
            if (material.diffuse_textureId != -1)
            {
                Texture t = textures[material.diffuse_textureId];
                glm::vec2 uv = intersection.surfaceUV;
                int x = (int)(glm::fract(uv.x) * (float)t.width);
                int y = (int)(glm::fract(1.0 - uv.y) * (float)t.width);

                int pixIdx = t.startIdx + y * t.width + x;

                materialColor = glm::vec3(pixels[pixIdx]);
            }

            glm::vec3 normal = intersection.surfaceNormal;

            // overwrite normal if using normal texture
            if (material.normal_textureId != -1)
            {
                Texture t = textures[material.normal_textureId];
                glm::vec2 uv = intersection.surfaceUV;
                int x = (int)(glm::fract(uv.x) * (float)t.width);
                int y = (int)(glm::fract(1.0 - uv.y) * (float)t.width);

                int pixIdx = t.startIdx + y * t.width + x;

                glm::vec3 pix_nor = glm::vec3(pixels[pixIdx]);

                glm::vec3 tangent = intersection.tangent;
                glm::vec3 bitangent = glm::cross(normal, tangent);

                glm::mat3 mat{ tangent, bitangent, normal };

                normal = mat * pix_nor;
            }

            material.color = materialColor;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                // Light source
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;  // Terminate path
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                Ray ray = pathSegments[idx].ray;
                scatterRay(pathSegments[idx], ray.origin + ray.direction * intersection.t, normal, material, rng);
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
            dev_intersections,
            dev_vertices,
            (int)hst_scene->vertices.size()
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;
        //std::cout << depth << " : " << num_paths << std::endl;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        thrust::sort_by_key(thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths, compareMatID());

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_pixels
        );

        PathSegment* new_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isActive());
        num_paths = new_end - dev_paths;

        if (num_paths <= 0) {
            iterationComplete = true;
        }

        if (depth > traceDepth)
        {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
