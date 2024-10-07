#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"
#include "texture.h"
#include "cudaUtilities.h"
#include "bvh.h"

using namespace std;
using namespace tinyobj;

using PrintFunction = void(*)(const char*, ...);
struct GPUInfo {
	cudaDeviceProp prop;
	size_t free_mem, total_mem;
	cudaEvent_t start, stop;
	float elapsedTime;
	int counter;
	int triangleCount;
	float averagePathPerBounce;
	GPUInfo() : counter(0), averagePathPerBounce(0)

	{
		cudaGetDeviceProperties(&prop, 0);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	~GPUInfo()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	void printMemoryInfo(PrintFunction printer)
	{
		cudaDeviceSynchronize();
		size_t free_mem, total_mem;
		cudaMemGetInfo(&free_mem, &total_mem);
		printer("counter: %d\n", counter++);
		printer("Free memory: %zu bytes\n", free_mem);
		printer("Total memory: %zu bytes\n", total_mem);
	}

	void printElapsedTime(PrintFunction printer)
	{
		printer("Elapsed time: %f ms\n", elapsedTime);
	}
};

extern GPUInfo* gpuInfo;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
	std::vector<Light> lights;
    std::vector<Material> materials;
	std::vector<Triangle> triangles;
	Texture* envMap;
    RenderState state;
	BVHAccel* bvh;
	std::string envMapPath;
    void createCube(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
	void createSphere(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale, int latitudeSegments = 40, int longitudeSegments = 20);
	void loadObj(const std::string& filename, uint32_t materialid = 0, glm::vec3 translation = glm::vec3(0), glm::vec3 rotation = glm::vec3(0), glm::vec3 scale = glm::vec3(1.));
	void addMaterial(Material& m);
    void loadEnvMap(const char* filename);
	void loadEnvMap();
    static void updateTransform(Geom& geom, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
	static void updateTriangleTransform(const Geom& geom, std::vector<Triangle>& triangles);
    void createBVH();
	BVHAccel::LinearBVHNode* getLBVHRoot();
};
