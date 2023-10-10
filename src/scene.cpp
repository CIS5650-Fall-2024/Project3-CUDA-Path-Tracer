#include <iostream>
#include "scene.h"
#include <cstring>
#include <cuda.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#define TINYGLTF_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"


static string GetFilePathExtension(const string &fileName) {
	if (fileName.find_last_of(".") != string::npos)
		return fileName.substr(fileName.find_last_of(".") + 1);
	return "";
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;

    std::string ext = GetFilePathExtension(filename);
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
	if (ext.compare("gltf") == 0) {
        cout << "Reading gltf scene file" << endl;
		ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        ret = parseGLTFModel(model);
	}
	else if (ext.compare("glb") == 0) {
        cout << "Reading glb scene file" << endl;
		ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
        ret = parseGLTFModel(model);
	}
    else {
        cout << "Reading generic scene file" << endl;
        ret = parseGenericFile(filename);
    }

    if (!warn.empty()) {
        cout << "Warn: " << warn << endl;
    }
    else if (!err.empty()) {
        cout << "Err: " << err << endl;
    }
    else if (!ret) {
        cout << "Failed to parse gltf file" << endl;
    }
    else {
        cout << "Successfully parsed gltf file" << endl;
    }
}

Scene::~Scene() {
    for (int i = 0; i < meshes.size(); i++) {
        delete[] meshes[i].vertices;
        delete[] meshes[i].indices;
    }
}

int Scene::parseGLTFNode(const int node, const tinygltf::Model &model, glm::mat4& baseTransform) {
    std::cout << "====\n parsing node: " << node << std::endl;
    int status = 1;
    auto& nodeObj = model.nodes[node];
    glm::mat4 transform(1.0f);
    glm::vec3 translation(0.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale(1.0f);
    if (nodeObj.matrix.size() == 16) {
        transform = glm::mat4(nodeObj.matrix[0], nodeObj.matrix[1], nodeObj.matrix[2], nodeObj.matrix[3],
                                nodeObj.matrix[4], nodeObj.matrix[5], nodeObj.matrix[6], nodeObj.matrix[7],
                                nodeObj.matrix[8], nodeObj.matrix[9], nodeObj.matrix[10], nodeObj.matrix[11],
                                nodeObj.matrix[12], nodeObj.matrix[13], nodeObj.matrix[14], nodeObj.matrix[15]);
    }
    else {
        if (nodeObj.translation.size() == 3) {
            translation = glm::vec3(nodeObj.translation[0], nodeObj.translation[1], nodeObj.translation[2]);
        }
        if (nodeObj.rotation.size() == 3) {
            // euler to quaternion
            glm::vec3 euler(nodeObj.rotation[0], nodeObj.rotation[1], nodeObj.rotation[2]);
            rotation = glm::quat(euler);
        }
        if (nodeObj.scale.size() == 3) {
            scale = glm::vec3(nodeObj.scale[0], nodeObj.scale[1], nodeObj.scale[2]);
        }

        transform = glm::translate(glm::mat4(), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
    }
    transform = baseTransform * transform;
    glm::mat4 inverseTransform = glm::inverse(transform);
    glm::mat4 invTranspose = glm::inverseTranspose(transform);
    translation = glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
    //rotation = glm::eulerAngles(glm::quat_cast(transform));
    scale = glm::vec3(glm::length(transform[0]), glm::length(transform[1]), glm::length(transform[2]));
    
    // print transform
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4;j++) {
            std::cout << transform[i][j] << " ";
        }
        std::cout << std::endl;
    }
    // if nodeObj has mesh
    if (nodeObj.mesh != -1) {
        int mesh = nodeObj.mesh;
        auto& meshObj = model.meshes[mesh];
        std::cout << "mesh: " << mesh << std::endl;
        auto& primitives = meshObj.primitives;
        for (auto& primitive : primitives) {
            auto& attributes = primitive.attributes;
            auto it = attributes.find("POSITION");
            if (it == attributes.end()) {
                std::cout << "no position attribute" << std::endl;
                continue;
            }

            auto& posAccessor = model.accessors[it->second];
            auto& posBufferView = model.bufferViews[posAccessor.bufferView];
            auto& posBuffer = model.buffers[posBufferView.buffer].data;
            int numVertices = posAccessor.count;
            float* vertices = new float[posAccessor.count * 3];
            std::memcpy(vertices, posBuffer.data() + posBufferView.byteOffset + posAccessor.byteOffset, posAccessor.count * 3 * sizeof(float));

            auto& indices = primitive.indices;      
            auto& indexAccessor = model.accessors[indices];
            auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            auto& indexBuffer = model.buffers[indexBufferView.buffer].data;
            int numIndices = indexAccessor.count;
            unsigned short* indicesVec = new unsigned short[indexAccessor.count];
            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                unsigned int* indicesVec_ = new unsigned int[indexAccessor.count];
                std::memcpy(indicesVec_, indexBuffer.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexAccessor.count * sizeof(unsigned int));
                for (int i = 0; i < indexAccessor.count; i++) {
                    indicesVec[i] = (unsigned short)indicesVec_[i];
                }
                delete[] indicesVec_;
            }
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                std::memcpy(indicesVec, indexBuffer.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexAccessor.count * sizeof(unsigned short));
            }
            else {
                std::cout << "unsupported index accessor type" << std::endl;
                return -1;
            }

            Material newMaterial;
            auto& material = primitive.material;
            if (material == -1) {
                std::cout << "no material found for primitive, using default material" << std::endl;
                newMaterial.color = glm::vec3(0.5f);
            }
            else {
                auto& materialObj = model.materials[material];
                auto& baseColor = materialObj.pbrMetallicRoughness.baseColorFactor;
                newMaterial.color = glm::vec3(baseColor[0], baseColor[1], baseColor[2]);
                newMaterial.specular.exponent = 0.0f;
                newMaterial.specular.color = glm::vec3(baseColor[0], baseColor[1], baseColor[2]);
                auto& metallicFactor = materialObj.pbrMetallicRoughness.metallicFactor;
                newMaterial.hasReflective = metallicFactor;
                auto& emitFactor = materialObj.emissiveFactor;
                newMaterial.emittance = glm::length(glm::vec3(emitFactor[0], emitFactor[1], emitFactor[2]));
                newMaterial.hasRefractive = 0.0f;

                std::cout << "material: " << material << std::endl;
                std::cout << "baseColor: " << baseColor[0] << " " << baseColor[1] << " " << baseColor[2] << std::endl;
                std::cout << "metallicFactor: " << metallicFactor << std::endl;
                std::cout << "emittance: " << newMaterial.emittance << std::endl;
            }
            materials.push_back(newMaterial);

            Mesh newMesh;
            newMesh.vertices = vertices;
            newMesh.indices = indicesVec;
            newMesh.numVertices = numVertices;
            if (numIndices % 3 != 0) {
                std::cout << "numIndices is not a multiple of 3" << std::endl;
                status = -1;
            }
            newMesh.numIndices = numIndices;
            newMesh.transform = transform;
            newMesh.inverseTransform = inverseTransform;
            newMesh.invTranspose = invTranspose;
            newMesh.translation = translation;
            //newMesh.rotation = rotation;
            newMesh.scale = scale;
            newMesh.materialid = materials.size() - 1;

            // calculate bounding sphere using vertices
            auto boundingVolume = findBoundingVolume(vertices, numVertices);
            boundingVolume.transform = transform * boundingVolume.transform;
            boundingVolume = getAxisAlignedBoundingBox(boundingVolume);
            newMesh.boundingVolume = boundingVolume;
            std::cout << "bounding volume: " << std::endl;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4;j++) {
                    std::cout << boundingVolume.transform[i][j] << " ";
                }
                std::cout << std::endl;
            }

            meshes.push_back(newMesh);

            Octree tree = buildOctree(newMesh);
            octrees.emplace_back(tree);
            std::cout << "******************* octree size: " << tree.nodes.size() << std::endl;
            std::cout << "transform: " << std::endl;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4;j++) {
                    std::cout << transform[i][j] << " ";
                }
                std::cout << std::endl;
            }
            for (int i=0; i<tree.nodes.size(); i++) {
                std::cout << "node: " << i << std::endl;
                for (int j=0; j<8; j++) {
                    std::cout << "  child: " << tree.nodes[i].children[j] << std::endl;
                }
                std::cout << "  dataStart: " << tree.dataStarts[i] << std::endl;
                std::cout << "  centroid: " << tree.boundingBoxes[i].translation.x << " " << tree.boundingBoxes[i].translation.y << " " << tree.boundingBoxes[i].translation.z << std::endl;
                std::cout << "  triangles: " << std::endl;
                for (int j=tree.dataStarts[i]; j<tree.dataStarts[i+1]; j++) {
                    std::cout << "------" << std::endl;
                    std::cout << "    " << tree.triangles[j].vertices[0].x << " " << tree.triangles[j].vertices[0].y << " " << tree.triangles[j].vertices[0].z << std::endl;
                    std::cout << "    " << tree.triangles[j].vertices[1].x << " " << tree.triangles[j].vertices[1].y << " " << tree.triangles[j].vertices[1].z << std::endl;
                    std::cout << "    " << tree.triangles[j].vertices[2].x << " " << tree.triangles[j].vertices[2].y << " " << tree.triangles[j].vertices[2].z << std::endl;
                }
                std::cout << "  bounding box: " << std::endl;
                auto& boundingBox = tree.boundingBoxes[i];
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        std::cout << boundingBox.transform[j][k] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    else if (nodeObj.camera != -1) {
        int camera = nodeObj.camera;
        auto& cameraObj = model.cameras[camera];
        std::cout << "camera: " << camera << std::endl;
        if (cameraObj.type == "perspective") {
            status = loadGLTFPerspectiveCamera(cameraObj, translation) && status;
        }
    }
    else if (nodeObj.name == "Camera") {
        status = addDefaultCamera(transform) && status;
    }

    for (auto& children : nodeObj.children) {
        status = parseGLTFNode(children, model, transform) && status;
    }

    return status;
}

Geom Scene::findBoundingVolume(float* vertices, int numVertices) {
    Geom boundingVolume;
    boundingVolume.type = CUBE;
    boundingVolume.materialid = 0;
    float maxX, maxY, maxZ = -std::numeric_limits<float>::infinity();
    float minX, minY, minZ = std::numeric_limits<float>::infinity();

    for (int i = 0; i < numVertices; i++) {
        std::cout << "vertex: " << vertices[i * 3] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << std::endl;
        maxX = fmax(maxX, vertices[i * 3]);
        maxY = fmax(maxY, vertices[i * 3 + 1]);
        maxZ = fmax(maxZ, vertices[i * 3 + 2]);
        minX = min(minX, vertices[i * 3]);
        minY = fmin(minY, vertices[i * 3 + 1]);
        minZ = fmin(minZ, vertices[i * 3 + 2]);
    }
    float xCenter = (maxX + minX) / 2;
    float yCenter = (maxY + minY) / 2;
    float zCenter = (maxZ + minZ) / 2;
    glm::vec3 translation = glm::vec3(xCenter, yCenter, zCenter);
    boundingVolume.translation = translation;
    glm::vec3 rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    boundingVolume.rotation = rotation;
    std::cout << "min max x: " << minX << " " << maxX << std::endl;
    std::cout << "min max y: " << minY << " " << maxY << std::endl;
    std::cout << "min max z: " << minZ << " " << maxZ << std::endl;
    float xBound = (maxX - minX);
    float yBound = (maxY - minY);
    float zBound = (maxZ - minZ);
    boundingVolume.scale = glm::vec3(xBound, yBound, zBound);

    boundingVolume.transform = utilityCore::buildTransformationMatrix(
        boundingVolume.translation, boundingVolume.rotation, boundingVolume.scale);

    return boundingVolume;
}

Octree Scene::buildOctree(const Mesh& mesh) {
    Octree tree;
    std::vector<Triangle> triangles;
    for (int i = 0; i < mesh.numIndices; i+=3) {
        Triangle triangle;

        triangle.vertices[0] = glm::vec3(mesh.vertices[mesh.indices[i]*3], 
                                         mesh.vertices[mesh.indices[i]*3 + 1], 
                                         mesh.vertices[mesh.indices[i]*3 + 2]);
        triangle.vertices[1] = glm::vec3(mesh.vertices[mesh.indices[i+1]*3], 
                                         mesh.vertices[mesh.indices[i+1]*3 + 1], 
                                         mesh.vertices[mesh.indices[i+1]*3 + 2]);
        triangle.vertices[2] = glm::vec3(mesh.vertices[mesh.indices[i+2]*3], 
                                         mesh.vertices[mesh.indices[i+2]*3 + 1], 
                                         mesh.vertices[mesh.indices[i+2]*3 + 2]);
        glm::vec3 centroid = (triangle.vertices[0] + triangle.vertices[1] + triangle.vertices[2]) / 3.0f;
        centroid = glm::vec3(mesh.transform * glm::vec4(centroid, 1.0f));
        triangle.centroid = centroid;
        triangle.transform = mesh.transform;
        triangles.push_back(triangle);
    }
    tree.transform = mesh.transform;
    tree.inverseTransform = mesh.inverseTransform;
    tree.invTranspose = mesh.invTranspose;

    // Geom boundingBox;
    // boundingBox.type = CUBE;
    // boundingBox.materialid = 1;
    // boundingBox.translation = mesh.translation;

    // boundingBox.scale = mesh.boundingVolume.scale;
    // boundingBox.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    // boundingBox.transform = utilityCore::buildTransformationMatrix(
    //     boundingBox.translation, boundingBox.rotation, boundingBox.scale);
    // boundingBox.inverseTransform = glm::inverse(boundingBox.transform);
    // boundingBox.invTranspose = glm::inverseTranspose(boundingBox.transform);

    tree.root = buildOctreeImpl(tree, mesh.boundingVolume, 0, triangles.begin(), triangles.end());
    tree.dataStarts.push_back(tree.triangles.size());
    return tree;
}

Geom Scene::getAxisAlignedBoundingBox(const Geom& meshBoundingVolume) {
    glm::vec3 vertices[8];
    vertices[0] = glm::vec3(0.5f, 0.5f, 0.5f);
    vertices[1] = glm::vec3(0.5f, 0.5f, -0.5f);
    vertices[2] = glm::vec3(0.5f, -0.5f, 0.5f);
    vertices[3] = glm::vec3(0.5f, -0.5f, -0.5f);
    vertices[4] = glm::vec3(-0.5f, 0.5f, 0.5f);
    vertices[5] = glm::vec3(-0.5f, 0.5f, -0.5f);
    vertices[6] = glm::vec3(-0.5f, -0.5f, 0.5f);
    vertices[7] = glm::vec3(-0.5f, -0.5f, -0.5f);
    float maxX, maxY, maxZ = -std::numeric_limits<float>::infinity();
    float minX, minY, minZ = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 8; i++) {

        vertices[i] = glm::vec3(meshBoundingVolume.transform * glm::vec4(vertices[i], 1.0f));
        std::cout << "TRASFORMED vertex: " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;
        maxX = fmax(maxX, vertices[i].x);
        maxY = fmax(maxY, vertices[i].y);
        maxZ = fmax(maxZ, vertices[i].z);
        minX = min(minX, vertices[i].x);
        minY = fmin(minY, vertices[i].y);
        minZ = fmin(minZ, vertices[i].z);
        std::cout << "min max x: " << minX << " " << maxX << std::endl;
        std::cout << "min max y: " << minY << " " << maxY << std::endl;
        std::cout << "min max z: " << minZ << " " << maxZ << std::endl;
    }
    std::cout << "bounds =========================" << std::endl;
    std::cout << "min max x: " << minX << " " << maxX << std::endl;
    std::cout << "min max y: " << minY << " " << maxY << std::endl;
    std::cout << "min max z: " << minZ << " " << maxZ << std::endl;

    glm::mat4 transform = glm::translate(glm::mat4(), glm::vec3((maxX + minX) / 2, (maxY + minY) / 2, (maxZ + minZ) / 2));
    glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.1f + maxX - minX, 0.1f + maxY - minY, 0.1f + maxZ - minZ));
    transform = transform * scale;
    Geom boundingBox;
    boundingBox.type = CUBE;
    boundingBox.materialid = 0;
    boundingBox.transform = transform;
    boundingBox.inverseTransform = glm::inverse(transform);
    boundingBox.invTranspose = glm::inverseTranspose(transform);
    boundingBox.translation = glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
    boundingBox.rotation = glm::eulerAngles(glm::quat_cast(transform));
    boundingBox.scale = glm::vec3(glm::length(transform[0]), glm::length(transform[1]), glm::length(transform[2]));
    return boundingBox;
}

template <typename Iterator>
int Scene::buildOctreeImpl(Octree& tree, const Geom& boundingBox, int depth, Iterator begin, Iterator end) {
    int newNodeId = tree.nodes.size();
    std::cout << "BUILDOCTTREE: " << newNodeId << " depth: " << depth << std::endl;
    
    if (begin == end) {
        std::cout << "   ===>no triangles " << newNodeId << std::endl;
        return -1;
    }
    if (depth > OCTREE_MAX_DEPTH) {
        std::cout << "   ===>max depth reached" << newNodeId << std::endl;
        return -1;
    }
    
    tree.nodes.emplace_back();
    tree.boundingBoxes.emplace_back(boundingBox);
    tree.dataStarts.emplace_back();
    tree.dataStarts[newNodeId] = tree.triangles.size();
    for (Iterator it = begin; it != end; ++it) {
        tree.triangles.push_back(*it);
    }
    
    if (begin + OCTREE_NUM_PRIMITIVES == end) {
        return newNodeId;
    }
    glm::vec3 center = boundingBox.translation;
    std::cout << "center: " << center.x << " " << center.y << " " << center.z << std::endl;

    auto xComp = [center](const Triangle& triangle) {
        return triangle.centroid.x < center.x;
    };
    auto yComp = [center](const Triangle& triangle) {
        return triangle.centroid.y < center.y;
    };
    auto zComp = [center](const Triangle& triangle) {
        return triangle.centroid.z < center.z;
    };

    Iterator split_x = std::partition(begin, end, xComp);
    Iterator split_y_lower = std::partition(begin, split_x, yComp);
    Iterator split_y_upper = std::partition(split_x, end, yComp);
    Iterator split_y_lower_z_lower = std::partition(begin, split_y_lower, zComp);
    Iterator split_y_lower_z_upper = std::partition(split_y_lower, split_x, zComp);
    Iterator split_y_upper_z_lower = std::partition(split_x, split_y_upper, zComp);
    Iterator split_y_upper_z_upper = std::partition(split_y_upper, end, zComp);

    Geom childBoundingBoxes[8];
    glm::vec3 halfScale = glm::vec3(0.50f);
    glm::vec3 translationScale = boundingBox.scale / 4.00f;
    int idx = 0;
    for (int x = -1; x <=1; x+=2) {
        for (int y = -1; y <=1; y+=2) {
            for (int z = -1; z <=1; z+=2) {
                std::cout << "  OG bounding box: " << std::endl;
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        std::cout << boundingBox.transform[j][k] << " ";
                    }
                    std::cout << std::endl;
                }
                childBoundingBoxes[idx].type = CUBE;
                childBoundingBoxes[idx].materialid = 0;

                glm::vec3 translation = glm::vec3(x * translationScale.x, y * translationScale.y, z * translationScale.z);

                glm::mat4 transform = glm::scale(glm::mat4(), halfScale);
                std::cout << "intended transform: " << std::endl;
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        std::cout << transform[j][k] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "boundngBoxScale: " << boundingBox.scale.x << " " << boundingBox.scale.y << " " << boundingBox.scale.z << std::endl; 
                std::cout << "translationScale: " << translationScale.x << " " << translationScale.y << " " << translationScale.z << std::endl;
                std::cout << "intended translation: " << translation.x << " " << translation.y << " " << translation.z << std::endl;
                childBoundingBoxes[idx].transform = boundingBox.transform * transform;
                childBoundingBoxes[idx].transform = glm::translate(glm::mat4(), translation) * childBoundingBoxes[idx].transform;
                childBoundingBoxes[idx].inverseTransform = glm::inverse(childBoundingBoxes[idx].transform);
                childBoundingBoxes[idx].invTranspose = glm::inverseTranspose(childBoundingBoxes[idx].transform);
                childBoundingBoxes[idx].translation = translation;
                childBoundingBoxes[idx].rotation = glm::eulerAngles(glm::quat_cast(childBoundingBoxes[idx].transform));
                childBoundingBoxes[idx].scale = glm::vec3(glm::length(childBoundingBoxes[idx].transform[0]), 
                                                          glm::length(childBoundingBoxes[idx].transform[1]), 
                                                          glm::length(childBoundingBoxes[idx].transform[2]));

                std::cout << "  NEW bounding box: " << std::endl;
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        std::cout << childBoundingBoxes[idx].transform[j][k] << " ";
                    }
                    std::cout << std::endl;
                }
                idx++;
            }
        }
    }

    std::vector<Triangle> childTriangles[8];
    for (Iterator it = begin; it != end; ++it) {
        Triangle& triangle = *it;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) {
                glm::vec3 transformedVertex = glm::vec3(childBoundingBoxes[i].transform * glm::vec4(triangle.vertices[j], 1.0f));
                if (glm::abs(transformedVertex.x) <= 0.5f && glm::abs(transformedVertex.y) <= 0.5f && glm::abs(transformedVertex.z) <= 0.5f) {
                    childTriangles[i].push_back(triangle);
                }
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        int child_idx = buildOctreeImpl(tree, childBoundingBoxes[i], depth + 1, childTriangles[i].begin(), childTriangles[i].end());
        tree.nodes[newNodeId].children[i] = child_idx;
    }

    // // // -x -y -z
    // int child_idx = buildOctreeImpl(tree, childBoundingBoxes[0], depth + 1, begin, split_y_lower_z_lower);
    // std::cout << "    node: " << newNodeId << " child: " << child_idx << std::endl;
    // tree.nodes[newNodeId].children[0] = child_idx;

    // // -x -y +z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[1], depth + 1, split_y_lower_z_lower, split_y_lower);
    // tree.nodes[newNodeId].children[1] = child_idx;
    // // -x +y -z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[2], depth + 1, split_y_lower, split_y_lower_z_upper);
    // tree.nodes[newNodeId].children[2] = child_idx;
    // // +x -y -z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[3], depth + 1, split_y_lower_z_upper, split_x);
    // tree.nodes[newNodeId].children[3] = child_idx;

    // // +x -y -z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[4], depth + 1, split_x, split_y_upper_z_lower);
    // tree.nodes[newNodeId].children[4] = child_idx;
    // // +x -y +z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[5], depth + 1, split_y_upper_z_lower, split_y_upper);
    // tree.nodes[newNodeId].children[5] = child_idx;
    // // +x +y -z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[6], depth + 1, split_y_upper, split_y_upper_z_upper);
    // tree.nodes[newNodeId].children[6] = child_idx;
    // // +x +y +z
    // child_idx = buildOctreeImpl(tree, childBoundingBoxes[7], depth + 1, split_y_upper_z_upper, end);
    // tree.nodes[newNodeId].children[7] = child_idx;

    for (int i = 0; i < 8; i++) {
        if (tree.nodes[newNodeId].children[i] != -1) {
            tree.nodes[newNodeId].isLeaf = false;
            break;
        }
    }

    for (int i = 0; i < 8; i++) {
        std::cout << "    node: " << newNodeId << std::endl;
        std::cout << "    child: " << tree.nodes[newNodeId].children[i] << std::endl;
    }

    std::cout << "   ===>return node: " << newNodeId << std::endl;
    return newNodeId;
}


int Scene::parseGLTFModel(const tinygltf::Model &model) {
    auto& scenes = model.scenes;
    int status = 1;
    glm::mat4 baseTransform(1.0f);
    for (auto& scene : scenes) {
        auto& nodes = scene.nodes;
        for (int node : nodes) {
            status = parseGLTFNode(node, model, baseTransform) && status;
        }
    }
    addGlobalIllumination();
    //addDefaultCamera();
    return status;
}

int Scene::addGlobalIllumination() {
    Material lightMaterial1;
    lightMaterial1.color = glm::vec3(1.0f, 0.5f, 0.5f);
    lightMaterial1.emittance = 15.0f;
    materials.push_back(lightMaterial1);

    Material lightMaterial2;
    lightMaterial2.color = glm::vec3(0.5f, 0.5f, 1.0f);
    lightMaterial2.emittance = 15.0f;
    materials.push_back(lightMaterial2);

    float x_spacing = 40.0f;
    float z_spacing = 40.0f;
    int num_x = 10;
    int num_z = 10;
    float x_start = -num_x * x_spacing / 2.0f;
    float z_start = -num_z * z_spacing / 2.0f;

    for (int i=0; i<num_x; i++) {
        for (int j=0; j<num_z; j++) {
            for (int k=-1; k<=1; k+=2) {
                Geom light;
                light.type = SPHERE;
                if (k == -1) {
                    light.materialid = materials.size()-2;
                }
                else {
                    light.materialid = materials.size()-1;
                }
                light.translation = glm::vec3(i*x_spacing+x_start, k * 50.0f, j * z_spacing + z_start);
                light.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
                light.scale = glm::vec3(10.0f, 10.0f, 10.0f);
                light.transform = utilityCore::buildTransformationMatrix(
                    light.translation, light.rotation, light.scale);
                light.inverseTransform = glm::inverse(light.transform);
                light.invTranspose = glm::inverseTranspose(light.transform);
                geoms.push_back(light);
            }
        }
    }
    


    Material skyMaterial;
    skyMaterial.color = glm::vec3(0.9f, 0.9f, 1.0f);
    skyMaterial.emittance = 0.5f;
    materials.push_back(skyMaterial);

    // Geom sky;
    // sky.type = SPHERE;
    // sky.materialid = materials.size()-1;
    // sky.translation = glm::vec3(0.0f, 0.1f, 0.0f);
    // sky.rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    // sky.scale = glm::vec3(1000.0f, 1000.0f, 1000.0f);
    // sky.transform = utilityCore::buildTransformationMatrix(
    //     sky.translation, sky.rotation, sky.scale);
    // sky.inverseTransform = glm::inverse(sky.transform);
    // sky.invTranspose = glm::inverseTranspose(sky.transform);
    // geoms.push_back(sky);

    return 1;
}

int Scene::addDefaultCamera(glm::mat4& transform) {
    RenderState &state = this->state;
    Camera & camera = state.camera;
    
    camera.resolution.x = 800;
    camera.resolution.y = 800;
    float aspectRatio = 1.0f;
    float fovy = 60.0f;
    camera.fov = glm::vec2(fovy * aspectRatio, fovy);
    camera.position = glm::vec3(transform[3][0], transform[3][1], transform[3][2]);
    camera.lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);
    camera.view = glm::normalize(camera.lookAt - camera.position);
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    state.iterations = 2000;
    state.traceDepth = 8;
    return 1;
}


int Scene::loadGLTFPerspectiveCamera(const tinygltf::Camera &cameraObj, glm::vec3& translation) {
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy = cameraObj.perspective.yfov * (180 / PI);

    camera.resolution.x = 800; // TODO: derive from cameraObj
    camera.resolution.y = 800;
    float aspectRatio = cameraObj.perspective.aspectRatio;
    camera.fov = glm::vec2(fovy * aspectRatio, fovy);
    camera.position = translation;
    camera.lookAt = glm::vec3(0.0f, 0.0f, 0.0f); // TODO: derive from cameraObj
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f); // TODO: derive from cameraObj
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    state.iterations = 2000; // TODO
    state.traceDepth = 8; // TODO

    std::cout << "camera position: " << glm::to_string(camera.position) << std::endl;
    std::cout << "camera lookAt: " << glm::to_string(camera.lookAt) << std::endl;
    std::cout << "camera up: " << glm::to_string(camera.up) << std::endl;
    std::cout << "camera right: " << glm::to_string(camera.right) << std::endl;
    std::cout << "camera view: " << glm::to_string(camera.view) << std::endl;
    std::cout << "camera pixelLength: " << glm::to_string(camera.pixelLength) << std::endl;
    std::cout << "camera fov: " << glm::to_string(camera.fov) << std::endl;
    std::cout << "camera resolution: " << glm::to_string(camera.resolution) << std::endl;
    std::cout << "camera aspectRatio: " << aspectRatio << std::endl;
    std::cout << "camera xscaled: " << xscaled << std::endl;
    std::cout << "camera yscaled: " << yscaled << std::endl;

    cout << "Loaded gltf camera!" << endl;
    return 1;
}

int Scene::parseGenericFile(string filename) {
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
    return 1;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

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

    cout << "camera view: " << glm::to_string(camera.view) << endl;
    cout << "camera up: " << glm::to_string(camera.up) << endl;
    cout << "camera right: " << glm::to_string(camera.right) << endl;
    cout << "camera lookAt: " << glm::to_string(camera.lookAt) << endl;
    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
