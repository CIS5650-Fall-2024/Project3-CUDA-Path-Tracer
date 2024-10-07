#pragma once
//#include <cuda_runtime.h>  
#include <texture_types.h>
#include "utilities.h"

class Texture {
public:
    Texture(const char* filename);
    ~Texture();
    cudaTextureObject_t texObj;
	bool createTextureFromFile(const char* filename, cudaTextureObject_t& texObj, cudaArray_t& cuArray);
    bool createHdriFromFile(const char* filename, cudaTextureObject_t& texObj, cudaArray_t& cuArray);
    float* padToFloat4(const float* data, int width, int height);
	void free();
	//__device__ float4 get(float u, float v);
private:
    cudaArray_t cuArray;  
    float* img_data;
    // 禁止拷贝和赋值
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
};