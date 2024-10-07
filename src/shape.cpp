#include "shape.h"
#include "tinyobj/tiny_obj_loader.h"
#include <cmath>

Shape::Shape(const Material& m)
    : transform(), material(std::make_unique<Material>(m))
{}

Shape::Shape()
    : transform(), material()
{}

#if 0
Mesh::Mesh(const Material& m)
    : Shape(m), triangles(),
    triangleSamplerIndex(-1),
    triangleStorageSideLen(-1),
    trianglesAsTexture(mkU<TextureTriangleStorage>(context, -1, -1, this))
{}

Mesh::Mesh(int triSamplerIndex, int triTexSlot)
    : Shape(), triangles(),
    triangleSamplerIndex(triSamplerIndex),
    triangleStorageSideLen(-1),
    trianglesAsTexture(mkU<TextureTriangleStorage>(context, -1, triTexSlot, this))
{}

Triangle::Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int idx)
    : pos{ p1, p2, p3 }, nor(), uv(), index_in_mesh(idx)
{}
#endif