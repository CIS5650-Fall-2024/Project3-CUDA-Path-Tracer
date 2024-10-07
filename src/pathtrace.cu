#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include "./thirdparty/oidn-2.3.0.x64.windows/include/OpenImageDenoise/oidn.hpp"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#ifndef M_PI
#define M_PI 3.1415926f
#endif
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
//switch between material
const bool SORT_BY_MATERIAL = false;
//toggleable BVH
const bool BVH = true;

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_denoised_image = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static std::vector<Triangle*> dev_mesh_triangles;
static std::vector<BVHNode*> dev_mesh_BVHNodes;

//testing
//static glm::vec3* dev_denoised_albedo = NULL;


int* dev_materialIds;

static oidn::DeviceRef device;
static oidn::BufferRef colorBuf;
static oidn::BufferRef normalBuf;
static oidn::BufferRef albedoBuf;
static oidn::FilterRef filter;
static oidn::FilterRef normalFilter;
static oidn::FilterRef albedoFilter;


thrust::device_ptr<int> dev_thrust_materialIds;
thrust::device_ptr<PathSegment> dev_thrust_paths;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

//only for load the enironment texture. As it mostly only be handle once ever.
//while texturedata will be created for multiple times if there are multiple objects,
//so I created a sepereted, extremly similar structure to hold floating point data of environment hdr
bool createCudaTexture_hdr(EnvData_hdr& textureData, Texture& textureObj) {
    if (textureData.h_data == nullptr) {
        textureObj.texObj = 0;
        textureObj.cuArray = nullptr;
        return true;
    }
    cudaError_t err; 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    err = cudaMallocArray(&textureObj.cuArray, &channelDesc, textureData.width, textureData.height);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA array: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    
    err = cudaMemcpy2DToArray(
        textureObj.cuArray,      
        0, 0,                    
        textureData.h_data,      
        textureData.width * 4 * sizeof(float), 
        textureData.width * 4 * sizeof(float), 
        textureData.height,      
        cudaMemcpyHostToDevice   
    );
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data to CUDA array: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(textureObj.cuArray);
        textureObj.cuArray = nullptr;
        return false;
    }
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureObj.cuArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;    
    texDesc.addressMode[1] = cudaAddressModeWrap;    
    texDesc.filterMode = cudaFilterModeLinear;      
    texDesc.readMode = cudaReadModeElementType;     
    texDesc.normalizedCoords = 1;                  
    err = cudaCreateTextureObject(&textureObj.texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA texture object: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(textureObj.cuArray);
        textureObj.cuArray = nullptr;
        return false;
    }
    delete[] textureData.h_data;
    textureData.h_data = nullptr;

    return true;
}

//modulize the creating process of cuda texture obj
bool createCudaTexture(TextureData& textureData, Texture& textureObj) {
    if (textureData.h_data == nullptr) {
        textureObj.texObj = 0;
        textureObj.cuArray = nullptr;
        return true;
    }

    cudaError_t err;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    err = cudaMallocArray(&textureObj.cuArray, &channelDesc, textureData.width, textureData.height);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA array: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMemcpy2DToArray(
        textureObj.cuArray,      
        0, 0,                     // Offset 
        textureData.h_data,       // host data
        textureData.width * 4 * sizeof(unsigned char), // bytes per row
        textureData.width * 4 * sizeof(unsigned char), // Width in bytes
        textureData.height,       // rows
        cudaMemcpyHostToDevice   
    );
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data to CUDA array: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(textureObj.cuArray);
        textureObj.cuArray = nullptr;
        return false;
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureObj.cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;     // Wrap  x
    texDesc.addressMode[1] = cudaAddressModeWrap;     // Wrap y
    texDesc.filterMode = cudaFilterModeLinear;        // Linear filtering
    texDesc.readMode = cudaReadModeNormalizedFloat;   // normalized float
    texDesc.normalizedCoords = 1;                      // normalized coordinates

    err = cudaCreateTextureObject(&textureObj.texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA texture object: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(textureObj.cuArray);
        textureObj.cuArray = nullptr;
        return false;
    }

    delete[] textureData.h_data;
    textureData.h_data = nullptr;

    return true;
}


void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

    /*cudaMalloc(&dev_denoised_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_albedo, 0, pixelcount * sizeof(glm::vec3));*/

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    Geom* host_geoms = new Geom[scene->geoms.size()];
    memcpy(host_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom));
    for (int i = 0; i < scene->geoms.size(); i++) {
        if (host_geoms[i].type == MESH) {
            int numTriangles = host_geoms[i].numTriangles;
            Triangle* dev_triangles;
            cudaMalloc(&dev_triangles, numTriangles * sizeof(Triangle));
            cudaMemcpy(dev_triangles, host_geoms[i].triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
            host_geoms[i].triangles = dev_triangles;
            dev_mesh_triangles.push_back(dev_triangles);

            int numBVHNodes = host_geoms[i].numBVHNodes;
            BVHNode* dev_BVH;
            cudaMalloc(&dev_BVH, numBVHNodes * sizeof(BVHNode));
            cudaMemcpy(dev_BVH, host_geoms[i].bvhNodes, numBVHNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);
            host_geoms[i].bvhNodes = dev_BVH;
            dev_mesh_BVHNodes.push_back(dev_BVH);
        }
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, host_geoms, scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    delete[] host_geoms;

    for (Material& material : scene->materials)
    {
        //for debugging
        if (material.albedoMapData.h_data != nullptr)
        {
            if (!createCudaTexture(material.albedoMapData, material.albedoMapTex)) {
                std::cerr << "Failed to create CUDA texture for albedo map." << std::endl;
                
                exit(EXIT_FAILURE);
            }
        }
        //for debugging
        if (material.normalMapData.h_data != nullptr)
        {
            if (!createCudaTexture(material.normalMapData, material.normalMapTex)) {
                std::cerr << "Failed to create CUDA texture for normal map." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        if (material.envMapData.h_data != nullptr)
        {
            if (!createCudaTexture_hdr(material.envMapData, material.envMap)) {
                std::cerr << "Failed to create CUDA texture for env map." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    }

    for (Material& material : scene->materials)
    {
        material.albedoMapData.h_data = nullptr;
        material.normalMapData.h_data = nullptr;
    }

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);



    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
    cudaMemset(dev_materialIds, 0, pixelcount * sizeof(int));

    /*cudaMalloc(&dev_color, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_color, (0.0f, 0.0f, 0.0f), pixelcount * sizeof(glm::vec3));*/

    device = oidn::newDevice(oidn::DeviceType::CUDA);
    device.commit();

    colorBuf = device.newBuffer(pixelcount * sizeof(glm::vec3));
    normalBuf = device.newBuffer(pixelcount * sizeof(glm::vec3));
    albedoBuf = device.newBuffer(pixelcount * sizeof(glm::vec3));

    filter = device.newFilter("RT");

    albedoFilter = device.newFilter("RT");
    albedoFilter.setImage("albedo", albedoBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    albedoFilter.setImage("output", albedoBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    albedoFilter.commit();

    normalFilter = device.newFilter("RT");
    normalFilter.setImage("normal", normalBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    normalFilter.setImage("output", normalBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    normalFilter.commit();



    filter.setImage("color", colorBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.setImage("normal", normalBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.setImage("output", colorBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.setImage("albedo", albedoBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.set("hdr", true);
    filter.set("cleanAux", true);
    filter.commit();

    


    checkCUDAError("pathtraceInit");
}


void pathtraceFree(Scene* scene)
{
    
    //all used for texture debugging, remember to allocate the texture obj after cuda initialization, 
    // so holding texture data using a host pointer is correct way
    cudaError_t err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing device: " << cudaGetErrorString(err) << std::endl;
    }


    for (Triangle* dev_triangles : dev_mesh_triangles) {
        err = cudaFree(dev_triangles);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_triangles: " << cudaGetErrorString(err) << std::endl;
        }
    }
    for (BVHNode* dev_bvh : dev_mesh_BVHNodes) {
        err = cudaFree(dev_bvh);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_bvh: " << cudaGetErrorString(err) << std::endl;
        }
    }
    dev_mesh_triangles.clear();
    dev_mesh_BVHNodes.clear();

    
    


    

    if (dev_image)
    {
        err = cudaFree(dev_image);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_image: " << cudaGetErrorString(err) << std::endl;
        }
        dev_image = nullptr;
    }  

    if (dev_paths)
    {
        err = cudaFree(dev_paths);;
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_paths: " << cudaGetErrorString(err) << std::endl;
        }
        dev_paths = nullptr;
    }
    if (dev_geoms)
    {
        err = cudaFree(dev_geoms);;
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_geoms: " << cudaGetErrorString(err) << std::endl;
        }
        dev_geoms = nullptr;
    }

    
    if (dev_materials)
    {
        err = cudaFree(dev_materials);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_materials: " << cudaGetErrorString(err) << std::endl;
        }
        dev_materials = nullptr;
    }
    if (dev_intersections)
    {
        err = cudaFree(dev_intersections);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_intersections: " << cudaGetErrorString(err) << std::endl;
        }
        dev_intersections = nullptr;
    }

    
    if (dev_denoised_image)
    {
        err = cudaFree(dev_denoised_image);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_denoised_image: " << cudaGetErrorString(err) << std::endl;
        }
        dev_denoised_image = nullptr;
    }

    // Free dev_albedo
    if (dev_albedo)
    {
        err = cudaFree(dev_albedo);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_albedo: " << cudaGetErrorString(err) << std::endl;
        }
        dev_albedo = nullptr;
    }

    // Free dev_normal
    if (dev_normal)
    {
        err = cudaFree(dev_normal);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_normal: " << cudaGetErrorString(err) << std::endl;
        }
        dev_normal = nullptr;
    }
    
    
    //cudaFree(dev_denoised_albedo);
    

    // TODO: clean up any extra device memory you created
    if(dev_materialIds)
    {
        err = cudaFree(dev_materialIds);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing dev_intersections: " << cudaGetErrorString(err) << std::endl;
        }
        dev_materialIds = nullptr;
    }
    


    colorBuf.release();
    normalBuf.release();
    albedoBuf.release();
    normalFilter.release();
    albedoFilter.release();
    filter.release();
    device.release();
    checkCUDAError("pathtraceFree");
}

//code from PBRT
__device__ glm::vec2 SampleUniformDiskConcentric(glm::vec2 u) {
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1, 1);

    if (uOffset.x == 0 && uOffset.y == 0) {
        return glm::bvec2(0, 0);
    }

    float r, theta;

    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = glm::pi<float>() / 4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = glm::pi<float>() / 2 - glm::pi<float>() / 4 * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(std::cos(theta), std::sin(theta));
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
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> uniforma1(0, 1);

        float jitterx = uniforma1(rng) - 0.5f;
        float jittery = uniforma1(rng) - 0.5f;

        
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jittery - (float)cam.resolution.y * 0.5f)
        );
        glm::vec3 rayDirection = segment.ray.direction;

        //code from PBRT

        float apertureRadius = cam.aperture;
        float focalDist = cam.focalDistance;

        if (apertureRadius > 0.0f) {
            
            glm::vec2 pLens = apertureRadius * SampleUniformDiskConcentric(glm::vec2(uniforma1(rng) - 0.5f, uniforma1(rng) - 0.5f));

            
            glm::vec3 pFocus = cam.position + rayDirection * focalDist;

            // Set the new ray origin on the lens
            segment.ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;

            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
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
    ShadeableIntersection* intersections
    )
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
        glm::vec2 uv;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tempUV;

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
            else if (geom.type == MESH) {
                if (BVH) {
                    t = meshIntersectionTest_BVH(geom, pathSegment.ray, tmp_intersect, tmp_normal, tempUV, outside);
                }
                else {
                    t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tempUV, outside);
                }
                
            }


            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tempUV;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].uv = glm::vec2(0.0f, 0.0f);
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;

            
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
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth,
    glm::vec3* normals,
    glm::vec3* albedo,
    cudaTextureObject_t envMap,
    float envMapIntensity)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[index];
        PathSegment& segment = pathSegments[index]; 

        if (intersection.t > 0.0f) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            glm::vec2 uv = intersection.uv;



            if (material.albedoMapTex.texObj != 0)
            {
                // Ensure UVs are within [0, 1]
                float u = glm::clamp(uv.x, 0.0f, 1.0f);
                float v = glm::clamp(uv.y, 0.0f, 1.0f);

                // Sample the texture
                float4 texColor = tex2D<float4>(material.albedoMapTex.texObj, u, v);
                glm::vec3 texColorVec(texColor.x, texColor.y, texColor.z);
                if (material.hasReflective > 0.0f) {
                    material.specular.color = texColorVec;
                }
                materialColor *= texColorVec;
            }


            if (material.normalMapTex.texObj != 0)
            {
                float u = glm::clamp(uv.x, 0.0f, 1.0f);
                float v = glm::clamp(uv.y, 0.0f, 1.0f);

                // Sample the normal map
                float4 texNormal = tex2D<float4>(material.normalMapTex.texObj, u, v);
                glm::vec3 sampledNormal(texNormal.x * 2.0f - 1.0f, texNormal.y * 2.0f - 1.0f, texNormal.z * 2.0f - 1.0f);
                sampledNormal = glm::normalize(sampledNormal);
                intersection.surfaceNormal = sampledNormal;
            }

            

            segment.ray.origin += segment.ray.direction * intersection.t;
            

           


            if (material.emittance > 0.0f) {
                segment.color *= (materialColor * material.emittance);
                segment.remainingBounces = 0;
                if (depth == 1) {
                    normals[index] += glm::vec3(0.0f);
                    albedo[index] += (materialColor * material.emittance);
                }
                
            }
            else {
                segment.remainingBounces--;
                glm::vec3 normal = glm::normalize(intersection.surfaceNormal);
                
                
                if (material.hasRefractive > 0.0f) {
                    glm::vec3 incident_direction = glm::normalize(segment.ray.direction);
                    

                    


                    float eta_i = 1.0f; // index of refraction of air
                    float eta_t = material.indexOfRefraction;
                    float cosTheta_i = -glm::dot(incident_direction, normal);

                    
                    if (cosTheta_i < 0.0f) {
                        //inside material
                        normal = -normal;
                        eta_i = material.indexOfRefraction;
                        eta_t = 1.0f;
                        cosTheta_i = -cosTheta_i;
                    }

                    ////Schlick's approximation
                    float R0 = (eta_i - eta_t) / (eta_i + eta_t);
                    R0 = R0 * R0;
                    float reflectance = R0 + (1.0f - R0) * powf(1.0f - cosTheta_i, 5.0f);
                    reflectance = glm::clamp(reflectance, 0.0f, 1.0f);
                  

                    ////Monte Carlo Sampling
                    float rand = u01(rng);

                    if (rand < reflectance) {
                        glm::vec3 reflectedDir = glm::reflect(incident_direction, normal);
                        segment.ray.direction = glm::normalize(reflectedDir);
                        segment.color *= material.specular.color;
                        if (depth == 1) {
                            normals[index] += normal;
                            albedo[index] += material.specular.color;
                        }
                        
                    }
                    else
                    {
                        //refraction
                        float eta = eta_i / eta_t;
                        glm::vec3 refractionDir = glm::refract(incident_direction, normal, eta);

                        
                        
                        //total internal reflaction
                        
                        if (glm::length(refractionDir) == 0.0f) {
                            glm::vec3 reflectedDir = glm::reflect(incident_direction, normal);
                            segment.ray.direction = glm::normalize(reflectedDir);
                            segment.color *= material.specular.color;
                            if (depth == 1) {
                                normals[index] += normal;
                                albedo[index] += material.specular.color;
                            }
                            
                        }
                        else 
                        {    
                            segment.ray.direction = glm::normalize(refractionDir);
                            segment.color *= materialColor;
                            if (depth == 1) {
                                normals[index] += normal;
                                albedo[index] += materialColor;
                            }
                           
                        }


                    }
                }
                else if (material.hasReflective > 0.0f) {
                    glm::vec3 reflectiveDirection = glm::reflect(segment.ray.direction,intersection.surfaceNormal);
                    segment.ray.direction = glm::normalize(reflectiveDirection);
                    

                    //used for imperfect specular surface

                    float roughness = material.roughness;
                    
                    if (roughness > 0.0f) {
                        //generate a random directio in hemisphere based on roughness
                        glm::vec3 randomDir = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);

                        randomDir = glm::normalize(glm::mix(reflectiveDirection, randomDir, roughness));
                        segment.ray.direction = randomDir;
                    }
                    segment.color *= material.specular.color;

                    if (depth == 1) {
                        normals[index] += normal;
                        albedo[index] += material.specular.color;
                    }
                   
                    
                }
                else {
                    // Diffuse reflection using random hemisphere sampling
                    glm::vec3 randomDir = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
                    
                    segment.ray.direction = glm::normalize(randomDir);
                    segment.color *= materialColor;
                    if (depth == 1) {
                        normals[index] += normal;
                        albedo[index] += materialColor;
                    }
                    
                }
            }
        }
        else {
            
            glm::vec3 rayDirection = glm::normalize(segment.ray.direction);
            if (envMap == 0) {
                segment.color = glm::vec3(0.0f);
                if (depth == 1) {
                    normals[index] += glm::vec3(0.0f);
                    albedo[index] += glm::vec3(0.0f);

                }
            }
            else {
                //convert into spherical coordinates
                float theta = acosf(rayDirection.y);
                float phi = atan2f(rayDirection.z, rayDirection.x);

                float u = (phi + M_PI) / (2.0f * M_PI);
                float v = theta / M_PI;

                float4 envColor = tex2D<float4>(envMap, u, v);

                glm::vec3 environmentLighting = glm::vec3(envColor.x, envColor.y, envColor.z) * envMapIntensity;


                //map some degree of the color into the module color
                //segment.color *= environmentLighting;

                //do not let any color of the env light module the texture color

                if ((segment.color.x < 1.0f) || (segment.color.y < 1.0f) || (segment.color.z < 1.0f)) {
                    segment.color *= environmentLighting;
                }
                else {
                    segment.color = environmentLighting;
                }

                if (depth == 1) {
                    normals[index] += glm::vec3(0.0f);
                    albedo[index] += environmentLighting;

                }
            }
            
            segment.remainingBounces = 0;

            
            
        }

        segment.ray.origin += 0.01f * segment.ray.direction;
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

__global__ void DenoiseGather(int nPaths, glm::vec3* image, glm::vec3* albedo ,glm::vec3* normals, glm::vec3* colorPtr, 
     glm::vec3* normal, glm::vec3* albedoPtr)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        
        normal[index] = glm::normalize(normals[index]);
        albedoPtr[index] = glm::normalize(albedo[index]);
        colorPtr[index] = image[index];
    }
}

__global__ void normalization(int nPaths, glm::vec3* normal, glm::vec3* albedo, glm::vec3* normalPtr, glm::vec3* albedoPtr)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
       
        normalPtr[index] = glm::normalize(normal[index]);
        albedoPtr[index] = glm::normalize(albedo[index]);
        
        
    }
}

__global__ void DenoiseReverse(int nPaths, glm::vec3* image, glm::vec3* colorPtr)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        
        image[index] = colorPtr[index];
        
    }
}

__global__ void extractMaterialIds(int num_paths, ShadeableIntersection* intersections, int* materialIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        materialIds[idx] = intersections[idx].materialId;
    }
}

struct IsPathTerminated
{
    __device__ bool operator()(const PathSegment& segment)
    {
        return segment.remainingBounces != 0;
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

    cudaTextureObject_t envMapTex = 0;
    float intensity;

    // Find the environment material
    for (int i = 0; i < hst_scene->materials.size(); ++i) {
        Material& material = hst_scene->materials[i];

        if (material.isEnvironment) {
            envMapTex = material.envMap.texObj;
            intensity = material.env_intensity;
            break;
        }
    }

    
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
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
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
        if (SORT_BY_MATERIAL) {
            extractMaterialIds << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_materialIds);
            cudaDeviceSynchronize();
            checkCUDAError("extractMaterialIds");

            thrust::device_ptr<int> dev_thrust_materialIds(dev_materialIds);
            thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
            thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
            thrust::sort_by_key(dev_thrust_materialIds, dev_thrust_materialIds + num_paths, thrust::make_zip_iterator(
                thrust::make_tuple(dev_thrust_paths, dev_thrust_intersections)
            ));


            cudaDeviceSynchronize();
        }
        

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth,
            dev_normal,
            dev_albedo,
            envMapTex,
            intensity
        );
        checkCUDAError("shading");
        cudaDeviceSynchronize();

        PathSegment* new_end = thrust::stable_partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            IsPathTerminated() 
        );
        cudaDeviceSynchronize();

        int n_num_paths = new_end - dev_paths;
        num_paths = n_num_paths; 
        iterationComplete = (num_paths == 0) || (depth >= traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
    


    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);
    if (iter % 3 == 0)
    {
          glm::vec3* colorPtr = (glm::vec3*)colorBuf.getData();
          glm::vec3* normalPtr = (glm::vec3*)normalBuf.getData();
          glm::vec3* albedoPtr = (glm::vec3*)albedoBuf.getData();
          normalization << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_normal, dev_albedo, normalPtr, albedoPtr);

          DenoiseGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_albedo, dev_normal, colorPtr, normalPtr, albedoPtr);
     
        
          // Prefilter the auxiliary images (normal image and albedo image)
          albedoFilter.execute();
          normalFilter.execute();
        

        
        //     Execute OIDN denoiser, Filter the beauty image
           filter.execute();
          const char* errorMessage;
          if (device.getError(errorMessage) != oidn::Error::None)
              std::cout << "Error: " << errorMessage << std::endl;
          DenoiseReverse << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_denoised_image, colorPtr);

    }
    
    //sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_albedo);
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_denoised_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
