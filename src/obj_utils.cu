#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include "obj_utils.h"
#include "intersections.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#define BBOX_EPS 0.0001f

void readOBJ(const std::string& filename, std::vector<Tri>& triangles, glm::vec3 scale, glm::vec3 trans) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    bool normalsExist = false;
    std::vector<glm::vec3> vertices;
    std::vector<std::array<int, 3>> tri_inds;
    std::vector<std::array<int, 3>> norm_inds;
    std::vector<glm::vec3> normals;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            glm::vec3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        } else if (prefix == "f") {
            int index;
            int root_index = -1;
            int root_normal_index;
            int prev_index = -1;
            int prev_normal_index = -1;
            std::string vertexInf;
            while (iss >> vertexInf) {
                std::istringstream vertexStream(vertexInf);
                int vertexIndex = 0;
                int normalIndex = -1;
                char slash;
                if (vertexInf.find("/") != std::string::npos) {
                    vertexStream >> vertexIndex >> slash >> slash >> normalIndex;
                } else {
                    vertexStream >> vertexIndex;
                }

                if (root_index == -1) {
                    root_index = vertexIndex;
                    root_normal_index = normalIndex;
                } else if(prev_index != -1) {
                    std::array<int, 3> tri_arr = {root_index - 1, prev_index - 1, vertexIndex - 1};
                    tri_inds.push_back(tri_arr);
                    std::array<int, 3> norm_ind_arr = {root_normal_index - 1, prev_normal_index - 1, normalIndex - 1};
                    norm_inds.push_back(norm_ind_arr);
                    //std::cout << "Root ind: " << root_index << " prev ind " << prev_index << " curr ind " << vertexIndex << std::endl;
                } else {
                    prev_index = vertexIndex;
                    prev_normal_index = normalIndex;
                }
            }
        } else if (prefix == "vn") {
            normalsExist = true;
            glm::vec3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
    }
    for(int i = 0; i < tri_inds.size(); i++) {
        const auto& tri_ind_set = tri_inds[i];
        const auto& norm_ind_set = norm_inds[i];
        Tri newTri;
        glm::vec3 normal_avg(0.0f);
        bool normal_exists = false;
        for(int j = 0; j < 3; j++) {
            newTri.vertices[j] = trans + vertices[tri_ind_set[j]] * scale;
            if(norm_ind_set[j] >= 0){
                normal_exists = true;
                normal_avg += normals[norm_ind_set[j]];
            }
        }
        normal_avg /= 3.0f;
        newTri.midpoint = (newTri.vertices[0] + newTri.vertices[1] + newTri.vertices[2]) / 3.0f;
        //print midpoint 
        //std::cout << newTri.midpoint.x << " " << newTri.midpoint.y << " " << newTri.midpoint.z << std::endl;
        newTri.color = glm::vec3(0.6f, 0.6f, 0.99f);
        //newTri.color = glm::vec3(0.4f, 0.2f, 0.1f);
        newTri.emittance = 0.0f;
        newTri.reflectivity = 0.0f;
        newTri.roughness = 0.0f;
        if(false || normal_exists){
            newTri.normal = glm::normalize(normal_avg);
        } else {
            newTri.normal = glm::normalize(glm::cross(newTri.vertices[1] - newTri.vertices[0], newTri.vertices[2] - newTri.vertices[0]));
        }
        triangles.push_back(newTri);
    }

    file.close();
}

int objTest(std::vector<glm::vec3> vertices, std::vector<Tri> triangles) {
    /*std::cout << "Vertices:" << std::endl;
    for (const auto& vertex : vertices) {
        std::cout << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
    }*/

    /*
    std::cout << "Triangles:" << std::endl;
    for (const auto& tri: triangles) {
        for (const auto& index : tri.vertexIndices) {
            std::cout << index << " ";
        }
        std::cout << std::endl;
    }
    */

    return 0;
}

__device__ FloatInt getTriIntersectionLinear(Ray r, Tri* triangles, int numTriangles, float t_min){
    int min_intersect_index = -1;
    for(int i = 0; i < numTriangles; i++){
        Tri tri = triangles[i];
        glm::vec3 hitPosition;
        float t;
        bool intersected = glm::intersectRayTriangle(r.origin, r.direction, tri.vertices[0], tri.vertices[1], tri.vertices[2], hitPosition);

        if(intersected){
            float t = hitPosition[2];//glm::length(hitPosition - r.origin);
            if(t < t_min && t > 0.0f){
                t_min = t;
                min_intersect_index = i;
            }
        }
    }
    FloatInt result;
    result.f = t_min;
    result.i = min_intersect_index;
    return result;
}


#define STACK_SIZE 100

__device__ FloatInt getTriIntersectionOctTree(Ray r, Tri* triangles, int numTriangles,
                                              OctTreeRegion* octRegions, int numOcts, float t_min) {
    int stack[STACK_SIZE];
    int stack_ind = 0;
    int t_min_index = -1;

    // Push the root node onto the stack
    stack[stack_ind++] = 0;  // Start with root node at index 0

    while (stack_ind > 0) {
        // get the current node and decrement the stack ind
        int curr_index = stack[--stack_ind];
        OctTreeRegion currRegion = octRegions[curr_index];

        // Test bounding box intersection with the ray
        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool outside = true;
        float t = boxIntersectionTest(currRegion.boundingBox, r, tmp_intersect, tmp_normal, outside);

        if (t > 0.0f && t < t_min) {
            // Check if the current node is a leaf node
            int first_child_index = curr_index * 8 + 1;
            if (first_child_index >= numOcts) {
                //leaf, test all triangles with linear
                FloatInt leaf_hit = getTriIntersectionLinear(r, triangles + currRegion.start_tri_ind,
                                                             currRegion.length, t_min);
                /*
                //Render bounding box hack
                int a = currRegion.start_tri_ind % 3;
                int b = currRegion.start_tri_ind % 5;
                int c = currRegion.start_tri_ind % 7;
                FloatInt leaf_hit = {t, currRegion.start_tri_ind};
                triangles[currRegion.start_tri_ind].color = glm::vec3((float)a / 7.0f, (float)b / 11.0f, (float)c / 17.0f);
                //end render hack
                */

                if (leaf_hit.i != -1 && leaf_hit.f < t_min) {
                    t_min = leaf_hit.f;
                    t_min_index = leaf_hit.i + currRegion.start_tri_ind;
                }
            } else {
                //add children to stack
                for (int i = 0; i < 8; i++) {
                    int child_index = first_child_index + i;
                    if (child_index < numOcts) {  
                        // Push child onto the stack
                        if (stack_ind >= STACK_SIZE) {
                            printf("stack overflow\n");
                            return FloatInt{t_min, t_min_index};
                        }
                        stack[stack_ind++] = child_index;
                    }
                }
            }
        }
    }

    return FloatInt{t_min, t_min_index};
}

struct SortTris {
    int component;
    bool min;
    bool midpoint;

    __host__ __device__ SortTris (int comp, bool mn, bool midp) : component(comp), min(mn), midpoint(midp) {}

    __host__ __device__ float get_val(const Tri& tri) const {
        if(midpoint){
            return tri.midpoint[component];
        } else {
            if(min){
                return glm::min(glm::min(tri.vertices[0], tri.vertices[1]), tri.vertices[2])[component];
            } else {
                return glm::max(glm::max(tri.vertices[0], tri.vertices[1]), tri.vertices[2])[component];
            }
        }
    }

    __host__ __device__
    bool operator()(const Tri& a, const Tri& b) const {
        float a_val = get_val(a);
        float b_val = get_val(b);
        return a_val < b_val;
    }
};

Geom boundingBoxFromTris(thrust::device_vector<Tri> curr_tri_vec){
    Geom bbox;
    Tri tmp_tri;
    SortTris curr_sorter(0, true, false);
    auto tri_pt = thrust::min_element(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
    thrust::copy(tri_pt, tri_pt + 1, &tmp_tri);
    float min_x = curr_sorter.get_val(tmp_tri);

    curr_sorter.component = 1;
    tri_pt = thrust::min_element(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
    thrust::copy(tri_pt, tri_pt + 1, &tmp_tri);
    float min_y = curr_sorter.get_val(tmp_tri);

    curr_sorter.component = 2;
    tri_pt = thrust::min_element(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
    thrust::copy(tri_pt, tri_pt + 1, &tmp_tri);
    float min_z = curr_sorter.get_val(tmp_tri);

    curr_sorter.component = 0;
    curr_sorter.min = false;
    tri_pt = thrust::max_element(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
    thrust::copy(tri_pt, tri_pt + 1, &tmp_tri);
    float max_x = curr_sorter.get_val(tmp_tri);

    curr_sorter.component = 1;
    tri_pt = thrust::max_element(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
    thrust::copy(tri_pt, tri_pt + 1, &tmp_tri);
    float max_y = curr_sorter.get_val(tmp_tri);

    curr_sorter.component = 2;
    tri_pt = thrust::max_element(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
    thrust::copy(tri_pt, tri_pt + 1, &tmp_tri);
    float max_z = curr_sorter.get_val(tmp_tri);

    bbox.type = CUBE;
    bbox.materialid = 0;
    bbox.translation = glm::vec3((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f, (min_z + max_z) / 2.0f);
    bbox.rotation = glm::vec3(0.0f);
    bbox.scale = glm::vec3(max_x - min_x, max_y - min_y, max_z - min_z);
    bbox.scale += glm::vec3(BBOX_EPS);
    bbox.transform = utilityCore::buildTransformationMatrix(
        bbox.translation, bbox.rotation, bbox.scale);
    bbox.inverseTransform = glm::inverse(bbox.transform);
    bbox.invTranspose = glm::inverseTranspose(bbox.transform);

    return bbox;
}


void formOctreeTrianglesArray(Tri* triangles, int numTriangles, OctTreeRegion* octrees,
                              int currDepth, int maxDepth, int curr_node_index, int startTriInd) {
    if (currDepth <= maxDepth && numTriangles > 0) {
        std::cout << "curr_node_index: " << curr_node_index << " at depth: " << currDepth << std::endl;

        thrust::device_ptr<Tri> curr_tri_ptr(triangles);
        thrust::device_vector<Tri> curr_tri_vec(curr_tri_ptr, curr_tri_ptr + numTriangles);

        //create and store bbox
        Geom bbox = boundingBoxFromTris(curr_tri_vec);
        OctTreeRegion currRegion;
        currRegion.boundingBox = bbox;
        currRegion.start_tri_ind = startTriInd;
        currRegion.length = numTriangles;
        currRegion.depth = currDepth;
        octrees[curr_node_index] = currRegion;

        if (currDepth < maxDepth && numTriangles > 1) {
            // Sort by x
            SortTris curr_sorter(0, true, true);
            thrust::sort(curr_tri_vec.begin(), curr_tri_vec.end(), curr_sorter);
            int mid_x_index = numTriangles / 2;
            int x_inds[3] = {0, mid_x_index, numTriangles};
            for (int x_side = 0; x_side < 2; x_side++) {
                curr_sorter.component = 1; // Sort by y
                thrust::sort(curr_tri_vec.begin() + x_inds[x_side], curr_tri_vec.begin() + x_inds[x_side + 1], curr_sorter);
                int mid_y_index = (x_inds[x_side] + x_inds[x_side + 1]) / 2;
                int y_inds[3] = {x_inds[x_side], mid_y_index, x_inds[x_side + 1]};
                for (int y_side = 0; y_side < 2; y_side++) {
                    curr_sorter.component = 2; // Sort by z
                    thrust::sort(curr_tri_vec.begin() + y_inds[y_side], curr_tri_vec.begin() + y_inds[y_side + 1], curr_sorter);
                    int mid_z_index = (y_inds[y_side] + y_inds[y_side + 1]) / 2;
                    int z_inds[3] = {y_inds[y_side], mid_z_index, y_inds[y_side + 1]};
                    for (int z_side = 0; z_side < 2; z_side++) {
                        int child_start_ind = z_inds[z_side];
                        int child_length = z_inds[z_side + 1] - z_inds[z_side];
                        int child_offset = (x_side * 4) + (y_side * 2) + z_side;
                        int child_node_index = curr_node_index * 8 + child_offset + 1; 
                        formOctreeTrianglesArray(triangles + child_start_ind, child_length, octrees,
                                                 currDepth + 1, maxDepth, child_node_index,
                                                 startTriInd + child_start_ind);
                    }
                }
            }
        }
    }
}

