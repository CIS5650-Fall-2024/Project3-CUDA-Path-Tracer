#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"


#include "tiny_obj_loader.h"



using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {

            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            newMaterial.hasReflective = 1;
            newMaterial.hasRefractive = 0;
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];

            newMaterial.hasReflective = 0;
            newMaterial.hasRefractive = 1;
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];

            newMaterial.hasReflective = 1;
            newMaterial.hasRefractive = 1;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "custom_shape_obj"){
            newGeom.type = OBJ;

            loadShapeFromOBJ(p["PATH"], vertices, triangles);

            newGeom.boundingBox = computeBoundingBox(vertices);
            //newGeom.boundingBox = transformAABB(newGeom.boundingBox, newGeom.transform);
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}



//bool Scene::loadShapeFromOBJ(const std::string& filename, std::vector<Vertex>& vertices, std::vector<Triangle>& triangles) {
//    tinyobj::attrib_t attrib;
//    std::vector<tinyobj::shape_t> shapes;
//    std::vector<tinyobj::material_t> materials;
//    std::string warn, err;
//
//
//    // Load the OBJ file
//    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
//    if (!ret) {
//        std::cerr << "Failed to load OBJ file: " << err << std::endl;
//        return false;
//    }
//
//    // Process vertex positions and normals
//    for (size_t i = 0; i < attrib.vertices.size() / 3; i++) {
//        Vertex vertex;
//
//        vertex.pos = glm::vec3(
//            attrib.vertices[3 * i + 0],
//            attrib.vertices[3 * i + 1],
//            attrib.vertices[3 * i + 2]
//        );
//
//        if (!attrib.normals.empty()) {
//            vertex.nor = glm::vec3(
//                attrib.normals[3 * i + 0],
//                attrib.normals[3 * i + 1],
//                attrib.normals[3 * i + 2]
//            );
//        }
//
//        vertices.push_back(vertex);
//    }
//
//
//    // Process faces (triangles)
//    for (const auto& shape : shapes) {
//        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
//            Triangle tri;
//            tri.idx_v0 = shape.mesh.indices[i + 0].vertex_index;
//            tri.idx_v1 = shape.mesh.indices[i + 1].vertex_index;
//            tri.idx_v2 = shape.mesh.indices[i + 2].vertex_index;
//            triangles.push_back(tri);
//        }
//    }
//
//    return true;
//
//}
//





bool Scene::loadShapeFromOBJ(const std::string& filename, std::vector<Vertex>& vertices, std::vector<Triangle>& triangles) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // Load the OBJ file
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
    if (!ret) {
        std::cerr << "Failed to load OBJ file: " << err << std::endl;
        return false;
    }

    if (!warn.empty()) {
        std::cerr << "TinyObjLoader warning: " << warn << std::endl;
    }

    // Process faces (triangles)
    for (const auto& shape : shapes) {
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            Triangle tri;

            // For each vertex in the triangle
            for (int j = 0; j < 3; ++j) {
                // Get the vertex index and normal index
                int vertex_index = shape.mesh.indices[i + j].vertex_index;
                int normal_index = shape.mesh.indices[i + j].normal_index;

                // Create a new Vertex and copy position
                Vertex vertex;
                vertex.pos = glm::vec3(
                    attrib.vertices[3 * vertex_index],
                    attrib.vertices[3 * vertex_index + 1],
                    attrib.vertices[3 * vertex_index + 2]
                );

                // Assign normal if available
                if (normal_index >= 0 && !attrib.normals.empty()) {
                    vertex.nor = glm::vec3(
                        attrib.normals[3 * normal_index],
                        attrib.normals[3 * normal_index + 1],
                        attrib.normals[3 * normal_index + 2]
                    );
                }
                else {
                    vertex.nor = glm::vec3(0.0f);  // Default to zero vector if no normals are present
                }

                // Add the new vertex to the vertices list and set the triangle indices
                vertices.push_back(vertex);
                if (j == 0) tri.idx_v0 = vertices.size() - 1;
                if (j == 1) tri.idx_v1 = vertices.size() - 1;
                if (j == 2) tri.idx_v2 = vertices.size() - 1;
            }

            triangles.push_back(tri);
        }
    }

    return true;
}




glm::vec2 calculatePixelLength(float xscale, float yscale, glm::ivec2 resolution) {
    float pixelWidth = (2 * xscale) / static_cast<float>(resolution.x);
    float pixelHeight = (2 * yscale) / static_cast<float>(resolution.y);
    return glm::vec2(pixelWidth, pixelHeight);
}



int Scene::constructBVH(int start_index, int end_index) {


    // Create a list of triangle indices
    std::vector<BVH_TriangleAABB_relation> relations(BVH_triangles.size());

    for (int i = 0; i < BVH_triangles.size(); i++) {
        const Triangle& T = BVH_triangles[i];

        const glm::vec3& v1 = T.vertices[0];
        const glm::vec3& v2 = T.vertices[1];
        const glm::vec3& v3 = T.vertices[2];

        relations[i] = BVH_TriangleAABB_relation(i, AABB(v1, v2, v3));

    }

    // In a fully binary tree, the maximum number of nodes is 2n - 1 for n primitives.
    int num_nodes = 2 * (end_index - start_index) - 1;

    std::vector<BVH_Node> nodes(num_nodes);

    BVH_Node& root = nodes[0];
    root.leftChild = 0;
    root.rightChild = 0;
    root.fisrtT_idx = 0;
    root.TCount = end_index - start_index;

    updateBVH(relations, nodes, 0);


    // Start the subdivision process to construct the BVH
    int visited_node = 1; // Start with the root node visited
    int largestLeafSize = 0; // To track the maximum leaf size
    int treeDepth = subdivide(relations, nodes, 0, visited_node, largestLeafSize);

    // Store the constructed nodes in the class's BVH_Nodes vector
    BVH_Nodes.clear(); // Clear any existing BVH nodes
    BVH_Nodes.insert(BVH_Nodes.end(), nodes.begin(), nodes.begin() + visited_node);

    return treeDepth;
}




void Scene::updateBVH(const std::vector<BVH_TriangleAABB_relation>& relation, std::vector<BVH_Node>& nodes, int idx)
{
    // validation check
    if (0 <= idx && idx < nodes.size()) {

        BVH_Node& node = nodes[idx];
        node.aabb = AABB();

        for (int i = node.fisrtT_idx; i < node.fisrtT_idx + node.TCount; i++)
        {
            node.aabb = combine2AABB(node.aabb, relation[i].aabb);
        }
    }
}




// ses Surface Area Heuristic (SAH) to determine the best split point

// relations: A vector containing BVH_TriangleAABB_relation objects, which hold information about triangles and their bounding boxes(AABBs).
// nodes : A vector of BVH_Node objects representing the BVH nodes.
// idx : The index of the current node in the nodes vector.
// visited_node : A reference that tracks the total number of nodes created.
// largestLeafSize : A reference that will store the maximum size of triangle in a node.
int Scene::subdivide(std::vector<BVH_TriangleAABB_relation>& relations, std::vector<BVH_Node>& nodes, int idx, int& visited_node, int& largestLeafSize) {
    
    //validation check
    assert(idx >= 0 && idx < nodes.size());

    BVH_Node& node = nodes[idx];
    node.splitAxis = node.aabb.LongestAxisIndex();
    int axis = node.splitAxis;

    int num_triangles = node.TCount;
    if (num_triangles <= 2) {
        // Create a leaf node if the number of primitives is small
        largestLeafSize = num_triangles;
        return 1;
    }

    // Calculate bounding boxes for SAH
    std::vector<AABB> leftBounds(num_triangles), rightBounds(num_triangles);
    AABB leftBox, rightBox;

    // Sort triangles along the chosen axis
    std::sort(relations.begin() + node.fisrtT_idx, relations.begin() + node.fisrtT_idx + num_triangles,
        [this, axis](const BVH_TriangleAABB_relation& a, const BVH_TriangleAABB_relation& b) {
            auto Ta = BVH_triangles[a.triangle_id];
            auto Tb = BVH_triangles[b.triangle_id];

            const glm::vec3& v1_a = Ta.vertices[0];
            const glm::vec3& v2_a = Ta.vertices[1];
            const glm::vec3& v3_a = Ta.vertices[2];

            const glm::vec3& v1_b = Tb.vertices[0];
            const glm::vec3& v2_b = Tb.vertices[1];
            const glm::vec3& v3_b = Tb.vertices[2];

            glm::vec3 centroid_a = (v1_a + v2_a + v3_a) * (1.0f / 3.0f);
            glm::vec3 centroid_b = (v1_b + v2_b + v3_b) * (1.0f / 3.0f);

            return centroid_a[axis] < centroid_b[axis];
        });
    
    // Compute bounding boxes from left to right
    for (int i = 0; i < num_triangles; i++) {
        leftBox = combine2AABB(leftBox, relations[node.fisrtT_idx + i].aabb);
        leftBounds[i] = leftBox;
    }

    // Compute bounding boxes from right to left
    for (int i = num_triangles - 1; i >= 0; i--) {
        rightBox = combine2AABB(rightBox, relations[node.fisrtT_idx + i].aabb);
        rightBounds[i] = rightBox;
    }

    // Find the best split position using SAH
    float bestCost = std::numeric_limits<float>::infinity();
    int bestSplit = -1;
    float invTotalArea = 1.0f / node.aabb.SurfaceArea();


    // Iterates over all possible split points. The loop goes from 0 to num_triangles - 2 
    // because splitting at the last triangle (num_triangles - 1) would not divide the node into two valid parts.
    for (int i = 0; i < num_triangles - 1; ++i) {
        float leftArea = leftBounds[i].SurfaceArea(); //leftBounds[i]: The cumulative AABB that contains all the triangles from the first triangle up to index i.
        float rightArea = rightBounds[i + 1].SurfaceArea(); //rightBounds[i + 1]: The cumulative AABB that contains all the triangles from index i + 1 to the last triangle.
        float cost = 0.125f + (leftArea * (i + 1) + rightArea * (num_triangles - i - 1)) * invTotalArea;

        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = i;
        }
    }

    if (bestSplit == -1) {
        // If we can't find a valid split, create a leaf node
        largestLeafSize = num_triangles;
        return 1;
    }


    // Partition triangles based on the best split
    int left_triangle_count = bestSplit + 1;
    int right_triangle_count = num_triangles - left_triangle_count;

    int left_index = visited_node++;
    int right_index = visited_node++;

    node.leftChild = left_index;
    node.rightChild = right_index;
    nodes[left_index].fisrtT_idx = node.fisrtT_idx;
    nodes[left_index].TCount = left_triangle_count;
    nodes[right_index].fisrtT_idx = node.fisrtT_idx + left_triangle_count;
    nodes[right_index].TCount = right_triangle_count;
    node.TCount = 0;

    // Update the bounding boxes of the child nodes
    updateBVH(relations, nodes, left_index);
    updateBVH(relations, nodes, right_index);

    int largestLeafSize_L, largestLeafSize_R;
    int leftDepth = subdivide(relations, nodes, left_index, visited_node, largestLeafSize_L);
    int rightDepth = subdivide(relations, nodes, right_index, visited_node, largestLeafSize_R);

    largestLeafSize = glm::max(largestLeafSize_L, largestLeafSize_R);
    return glm::max(leftDepth, rightDepth) + 1;
}



