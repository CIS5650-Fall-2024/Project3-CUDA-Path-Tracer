#pragma once
#if 0
#include <string>
#include <vector>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "transform.h"
#include <memory>
#include <array>
#include <json.hpp>

struct Material;

class Shape {
public:
	Shape();
	Shape(const Material& m);

	virtual ~Shape() {};

	Transform transform;
	std::unique_ptr<Material> material;
    std::vector<int> msterialids;
};




class Mesh;
class Triangle {
private:
    std::array<glm::vec3, 3> pos;
    std::array<glm::vec3, 3> nor;
    std::array<glm::vec3, 3> uv;
    int index_in_mesh; // What index in the mesh's vector<Triangles> does this sit at?

public:
    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int idx);

    glm::vec3 operator[](unsigned int) const;

    friend class Mesh;
};

class TextureTriangleStorage;

class Mesh : public Shape {
private:
    std::vector<Triangle> triangles;

    // Shader has an array of sampler2Ds
    // to read each Mesh's triangles.
    // Each Mesh knows which sampler2D reads its data
    int triangleSamplerIndex;
    int triangleStorageSideLen;
    std::unique_ptr<TextureTriangleStorage> trianglesAsTexture;

public:
    static unsigned int nextLowestSamplerIndex;

    Mesh(const Material& m);
    Mesh(int triSamplerIndex, int triTexSlot);

    void LoadOBJ(const std::string& filename, const std::string& local_path, int triangle_mesh_id);
    unsigned int numTris() const;
    void computeStorageDimensions(int* w, int* h) const;

    friend class Scene;
    friend class TextureTriangleStorage;
};
#endif
