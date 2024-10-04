#include "pathtrace.h"

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static GuiDataContainer* guiData = NULL;
static Geom* dev_geoms = NULL;
static Geom* dev_geoms_after_t = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static glm::vec3* dev_vertices = NULL;
static MeshTriangle* dev_meshes = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec2* dev_texcoords = NULL;
static Texture* dev_textures = NULL;
static float* dev_textures_data_1 = NULL;

cudaTextureObject_t albedoTexture = 0;
cudaTextureObject_t metallicTexture = 0;
cudaTextureObject_t normalTexture = 0;
cudaTextureObject_t emissiveTexture = 0;
struct cudaResourceDesc resDesc;
struct cudaTextureDesc texDesc;

#if BVH
static BVHNode* dev_bvh = NULL;
#endif

#if OIDN
#define EMA_ALPHA 0.2f
#define DENOISE_INTERVAL 100

#include <OpenImageDenoise/oidn.hpp>

static glm::vec3* dev_denoised = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;

void denoise()
{
    int width = hst_scene->state.camera.resolution.x,
        height = hst_scene->state.camera.resolution.y;

    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", dev_image, oidn::Format::Float3, width, height); 
    filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    filter.setImage("output", dev_denoised, oidn::Format::Float3, width, height); 
    filter.set("hdr", true); 
    filter.set("cleanAux", true); 
    filter.commit();

    oidn::FilterRef albedoFilter = device.newFilter("RT"); 
    albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    oidn::FilterRef normalFilter = device.newFilter("RT"); 
    normalFilter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.setImage("output", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.commit();

    albedoFilter.execute();
    normalFilter.execute();

    filter.execute();

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;
}

__global__
void copyTraceResult(
    PathSegment* pathSegments, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    glm::vec3* albedo, glm::vec3* normal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        PathSegment pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];

        albedo[pathSegment.pixelIndex] = pathSegment.color;
        normal[pathSegment.pixelIndex] = intersection.surfaceNormal;
    }
}

__global__
void mergeDenoisedAndImage(int pixelcount, glm::vec3* image, glm::vec3* denoised)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelcount) {
        // exponential moving average
        image[idx] = image[idx] * (1 - EMA_ALPHA) + denoised[idx] * EMA_ALPHA;
    }
}
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

struct isRayAlive
{
    __host__ __device__ bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

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

void LoadTextureData(Scene* scene, std::string filename, cudaTextureObject_t& texObj)
{
    //Texture
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (!data) {
        std::cout << "No texture for this scene." << filename << std::endl;
        return;
    }

    float* h_data = (float*)std::malloc(sizeof(float) * width * height * 4);

    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < channels; c++) {
            h_data[i * 4 + c] = data[i * channels + c] / 255.0f;
        }
        if (channels == 3) {
            h_data[i * 4 + 3] = 1.0f;
        }
    }

    //Procedual Texture
    /*for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = (y * width + x) * 4;
            float u = (float)x / (width - 1);
            float v = (float)y / (height - 1);
            h_data[i] = abs(sin(u * 10));
            h_data[i + 1] = abs(cos(v * 30));
            h_data[i + 2] = (u + v) / 2.0f;
            h_data[i + 3] = 1.0f;
        }
    }*/

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    const size_t spitch = width * 4 * sizeof(float);
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * 4 * sizeof(float),
        height, cudaMemcpyHostToDevice);

    // Specify texture
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    std::free(h_data);
    stbi_image_free(data);

}

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

    //Mesh data
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_geoms_after_t, scene->geoms.size() * sizeof(Geom));

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(MeshTriangle));
    cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(MeshTriangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_texcoords, scene->texcoords.size() * sizeof(glm::vec2));
    cudaMemcpy(dev_texcoords, scene->texcoords.data(), scene->texcoords.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    //Loat Texture
    if (scene->texturePaths.size() > 2) {
        std::string filename1 = scene->texturePaths[0];
        LoadTextureData(scene, filename1, albedoTexture);

        std::string filename2 = scene->texturePaths[1];
        LoadTextureData(scene, filename2, metallicTexture);

        std::string filename3 = scene->texturePaths[2];
        LoadTextureData(scene, filename3, normalTexture);
        
        std::string filename4 = scene->texturePaths[3];
        LoadTextureData(scene, filename4, emissiveTexture);
    }
    
#if BVH
    cudaMalloc(&dev_bvh, scene->bvh.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvh, scene->bvh.data(), scene->bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
#endif

#if OIDN
    cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image); 
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_geoms_after_t);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_texcoords);
    cudaFree(dev_normals);
    cudaFree(dev_vertices);
    cudaFree(dev_meshes);
    cudaFree(dev_textures);
    cudaFree(dev_textures_data_1);
    cudaDestroyTextureObject(albedoTexture);

#if BVH
    cudaFree(dev_bvh);
#endif

#if OIDN
    cudaFree(dev_denoised);
    cudaFree(dev_albedo);
    cudaFree(dev_normal);
#endif

    checkCUDAError("pathtraceFree");
}

__device__ glm::vec3 checkerboard(glm::vec2 uv)
{
    if ((int)(uv.x * 10) % 2 == (int)(uv.y * 10) % 2)
        return glm::vec3(.2f);
    else
        return glm::vec3(.8f);
}
__device__ glm::vec3 palettes(glm::vec2 uv)
{
    glm::vec3 a(0.5, 0.5, 0.5), b(0.5, 0.5, 0.5), c(1.0, 1.0, 1.0), d(0.00, 0.33, 0.67);
    return a + b * glm::cos(TWO_PI * (c * glm::length(uv) + d));
}

__host__ __device__
glm::vec2 RingsProcedualTexture(const glm::vec2& u)
{
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f, 1.0f);

    if (uOffset.x == 0.0f && uOffset.y == 0.0f)
    {
        return glm::vec2(0.0f, 0.0f);
    }

    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PI_OVER_FOUR * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

__device__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

__device__ float grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y,
        v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__device__ const int p[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__device__ float perlin(float x, float y, float z) {
    int X = (int)floor(x) & 255,
        Y = (int)floor(y) & 255,
        Z = (int)floor(z) & 255;
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    float u = fade(x),
        v = fade(y),
        w = fade(z);
    int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z,
        B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;

    return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
        grad(p[BA], x - 1, y, z)),
        lerp(u, grad(p[AB], x, y - 1, z),
            grad(p[BB], x - 1, y - 1, z))),
        lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
            grad(p[BA + 1], x - 1, y, z - 1)),
            lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                grad(p[BB + 1], x - 1, y - 1, z - 1))));
}

__device__ glm::vec3 fireball(glm::vec2 uv) {
    float phi = uv.x * 2.0f * PI;
    float theta = uv.y * PI;
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);

    float noise = perlin(x * 4.0f, y * 4.0f, z * 4.0f );
    noise = (noise + 1.0f) * 0.5f;

    glm::vec3 baseColor = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 glowColor = glm::vec3(1.0f, 1.0f, 0.0f); 

    float t = pow(noise, 1.5f);
    glm::vec3 color = glm::mix(baseColor, glowColor, t);

    float glow = pow(1.0f - length(uv - glm::vec2(0.5f, 0.5f)) * 2.0f, 2.0f);
    color += glowColor * glow * 0.5f;

    return color;
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

#if SSAA
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

        // Depth of field automatically enabled for camera with LENSRADIUS and FOCALDIS
        if (cam.lensRadius > 0)
        {
            glm::vec2 pLens = cam.lensRadius * RingsProcedualTexture(glm::vec2(u01(rng), u01(rng)));
            float ft = cam.focalDistance / glm::dot(cam.view, segment.ray.direction);
            glm::vec3 pFocus = segment.ray.origin + segment.ray.direction * ft;
            segment.ray.origin += cam.right * pLens.x + cam.up * pLens.y;
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
    ShadeableIntersection* intersections,
#if BVH
    BVHNode* bvh,
#endif
    MeshTriangle* meshes
    , Texture* textures
    , glm::vec3* vertices
    , glm::vec3* normals
    , glm::vec2* texcoords)
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
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        int tmp_material_index;
        glm::vec2 tmp_texcoord;
        bool tmp_outside = true;

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside, tmp_uv);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside,tmp_uv);
            }
            else if (geom.type == MESH) 
            {
#if BVH
                t = meshIntersectionTestBVH(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside, tmp_uv,
                    bvh, meshes, vertices, normals, texcoords, tmp_material_index);

#else
                t = meshIntersectionTestNaive(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside, tmp_uv,
                    meshes, vertices, normals, texcoords, tmp_material_index);
#endif        
            }
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                outside = tmp_outside;
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
            intersections[path_index].outside = outside;
        }
    }
}

__global__ void updateGeomsPosition(float time, Geom* geom, Geom* geom_after_t, int size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= size) return;
    Geom currGeom = geom[index];
    if (glm::length(currGeom.velocity) > 0.001) {
        currGeom.translation += time * currGeom.velocity;

        glm::mat4 translationMat = glm::translate(glm::mat4(), currGeom.translation);
        glm::mat4 rotationMat = glm::rotate(glm::mat4(), currGeom.rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), currGeom.rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), currGeom.rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
        glm::mat4 scaleMat = glm::scale(glm::mat4(), currGeom.scale);
        currGeom.transform = translationMat * rotationMat * scaleMat;

        currGeom.inverseTransform = glm::inverse(currGeom.transform);
        currGeom.invTranspose = glm::inverseTranspose(currGeom.transform);
    }
    geom_after_t[index] = currGeom;
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
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* textures,
    cudaTextureObject_t albedoTexture,
    cudaTextureObject_t normalTexture,
    cudaTextureObject_t metallicTexture,
    cudaTextureObject_t emissiveTexture
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            Material material = materials[intersection.materialId];
            float u = intersection.uv.x;
            float v = intersection.uv.y;

            //Emissive Mapping
            bool isEmissive = material.shadingType == ShadingType::Emitting;
            if (material.emissiveTextureId != -1) {
                float4 emissive = tex2D<float4>(emissiveTexture, u, v);
                isEmissive = isEmissive || (emissive.x > 0.1f);
                material.emittance = 1.0f;
                material.color = glm::vec3(1.0f);
            }
            
            if (isEmissive) {
                // Light
                glm::vec3 materialColor = material.color;
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                //Procedual Texture, only affects the albedo for better visual
                switch (material.procedualTextureID) {
                     case 1: material.color = checkerboard(intersection.uv); break;
                     case 2: material.color = palettes(intersection.uv); break;
                     case 3: material.color = fireball(intersection.uv); break;
                     default: break;
                }

                //Texture mapping
                if (material.baseColorTextureId != -1) {
                    float4 texel = tex2D<float4>(albedoTexture, u, v);
                    material.color = glm::vec3(texel.x, texel.y, texel.z);
                }

                //Normal mapping
                if (material.normalTextureId != -1) {
                    float4 normalSample = tex2D<float4>(normalTexture, u, v);
                    glm::vec3 normalFromMap = glm::vec3(normalSample.x, normalSample.y, normalSample.z) * 2.0f - 1.0f;
                    glm::vec3 T = glm::normalize(glm::cross(intersection.surfaceNormal, glm::vec3(0, 1, 0)));
                    glm::vec3 B = glm::cross(intersection.surfaceNormal, T);
                    glm::mat3 TBN = glm::mat3(T, B, intersection.surfaceNormal);
                    glm::vec3 worldNormal = TBN * normalFromMap;
                    intersection.surfaceNormal = glm::normalize(worldNormal);
                }

                // Metallic-Roughness mapping
                if (material.roughnessMetallicTextureId != -1) {
                    float4 metallicRoughnessSample = tex2D<float4>(metallicTexture, u, v);
                    intersection.metallic = metallicRoughnessSample.z;  
                    intersection.roughness = metallicRoughnessSample.y;
                }

                //Ray scatter
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
                thrust::uniform_real_distribution<float> u01(0, 1);
                //glm::vec3 intersect = intersection.t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
                scatterRay(pathSegments[idx], intersection, material, rng);
            }
        }
        else {
            //No hit
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }

#if RR
        // Russian roulette
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (depth > 3 && pathSegments[idx].remainingBounces > 0) {
            float y = glm::max(glm::max(pathSegments[idx].color.x, pathSegments[idx].color.y), pathSegments[idx].color.z);
            float q = max(0.05f, 1 - y);
            if (u01(rng) < q) {
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                pathSegments[idx].color /= (1 - q);
            }
        }
#endif

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
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);

    //Update Geom position
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, 0, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float time = min(1.0f,1.05f * glm::sqrt(u01(rng)));
    int numBlocks = (hst_scene->geoms.size() + blockSize1d - 1) / blockSize1d;
    updateGeomsPosition << <numBlocks, blockSize1d >> > (time, dev_geoms, dev_geoms_after_t, hst_scene->geoms.size());

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth, num_paths, dev_paths, dev_geoms_after_t, hst_scene->geoms.size(), dev_intersections
#if BVH
            , dev_bvh
#endif 
            , dev_meshes, dev_textures, dev_vertices, dev_normals, dev_texcoords
        );
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
#ifndef SORT_MATERIAL_ID
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialsCmp());
#endif
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter, depth, num_paths, dev_intersections, dev_paths, dev_materials, dev_textures, albedoTexture, normalTexture, metallicTexture, emissiveTexture
        );

#if OIDN
        if (depth == 1 && (iter % DENOISE_INTERVAL == 0 || iter == hst_scene->state.iterations))
            copyTraceResult << <numblocksPathSegmentTracing, blockSize1d >> > (
                dev_paths, num_paths, dev_intersections, dev_albedo, dev_normal);
#endif

#ifdef STREAM_COMPACTION
        num_paths = thrust::partition(thrust::device,
            dev_paths, dev_paths + num_paths, isRayAlive()) - dev_paths;
#endif

        iterationComplete = depth == traceDepth || num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

#ifdef STREAM_COMPACTION
    num_paths = dev_path_end - dev_paths;
#endif

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

#if OIDN
    if (iter % DENOISE_INTERVAL == 0 && iter != 0)
    {
        denoise();
        mergeDenoisedAndImage << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_denoised);
    }
    else if (iter == hst_scene->state.iterations)
    {
        denoise();
        std::swap(dev_image, dev_denoised);
    }
#endif

    // --- Rendering Stage ---

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
