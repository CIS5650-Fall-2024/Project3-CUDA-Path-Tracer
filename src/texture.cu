#include "texture.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

Texture::Texture(const char* filename){
	// 创建纹理对象
	printf("Creating texture from file: %s\n", filename);
	if (!createHdriFromFile(filename, texObj, cuArray)) {
		// 创建失败，释放资源
		texObj = 0;
		cuArray = nullptr;
	}
}

Texture::~Texture() {
	if (texObj) {
		free();
	}
}


bool Texture::createHdriFromFile(const char* filename, cudaTextureObject_t& texObj, cudaArray_t& cuArray)
{
	//free();
	// load hdri
	int width, height, channels;
	img_data = stbi_loadf(filename, &width, &height, &channels, 4);
	if (img_data == nullptr) {
		fprintf(stderr, "Failed to load HDRI file: %s\n", filename);
		return false;
	}

	// create cuda array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	cudaMallocArray(&cuArray, &channelDesc, width, height);
	cudaMemcpy2DToArray(cuArray, 0, 0, img_data, width * sizeof(float4),
		width * sizeof(float4), height, cudaMemcpyHostToDevice);

	// create cuda object
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
	return true;
}

float* Texture::padToFloat4(const float* data, int width, int height) {
	float* data4 = new float[width * height * 4];
	for (int i = 0; i < width * height; i++) {
		data4[i * 4 + 0] = data[i * 3 + 0];
		data4[i * 4 + 1] = data[i * 3 + 1];
		data4[i * 4 + 2] = data[i * 3 + 2];
		data4[i * 4 + 3] = 1.0f;
	}
	return data4;
}

void Texture::free()
{
	if (!img_data) stbi_image_free(img_data);
	img_data = nullptr;

	if (texObj != 0)
	{
		cudaDestroyTextureObject(texObj);
		texObj = 0;
	}
	// free cuda array
	if (cuArray != nullptr)
	{
		cudaFreeArray(cuArray);
		cuArray = nullptr;
	}
}