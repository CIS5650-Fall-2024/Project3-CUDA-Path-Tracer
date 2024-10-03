#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_gltf.h"
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
        if(!triangles.empty())
			BuildBVH();
        //printf("Total triangles: %d\n", triangles.size());

        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }

}

#if 0
AABB Scene::calculateAABBTriangles(const Triangle& tri) {
    AABB aabb;
    aabb.max = glm::vec3(-INFINITY);
    aabb.min = glm::vec3(INFINITY);
    for (int i = 0; i < 3; i++) {
		aabb.min = glm::min(aabb.min, tri.verts[i]);
		aabb.max = glm::max(aabb.max, tri.verts[i]);
	}
    return aabb;
}

#endif

AABB Scene::calculateAABBMeshs(Geom& mesh) {
    AABB aabb;
    aabb.max = glm::vec3(-INFINITY);
    aabb.min = glm::vec3(INFINITY);
    for (int i = mesh.triIndexStart; i < mesh.triIndexEnd; ++i) {
        printf("Build aabb box for triangle %d\n", i);
        for (int j = 0; j < 3; ++j) {
			aabb.min = glm::min(aabb.min, triangles[i].verts[j]);
			aabb.max = glm::max(aabb.max, triangles[i].verts[j]);
        }
	}  
    mesh.aabb = aabb;
    return aabb;
}

AABB Scene::calculateAABBSpheres(Geom& sphere) {
    printf("Build aabb box for sphere\n");
	AABB aabb;
	aabb.min = sphere.translation - glm::vec3(sphere.scale.x);
	aabb.max = sphere.translation + glm::vec3(sphere.scale.x);
    sphere.aabb = aabb;
	return aabb;
}

AABB Scene::calculateAABBCubes(Geom& cube) {
    printf("Build aabb box for cube\n");
	AABB aabb;
    glm::vec3 halfSize = cube.scale * 0.5f;
	aabb.min = cube.translation - halfSize;
	aabb.max = cube.translation + halfSize;
    cube.aabb = aabb;
	return aabb;
}


#if 0
AABB mergeAABBs(const AABB& aabb1, const AABB& aabb2) {
    AABB mergedAABB;

    mergedAABB.min = glm::min(aabb1.min, aabb2.min);
    mergedAABB.max = glm::max(aabb1.max, aabb2.max);

    return mergedAABB;
}

int Scene::buildBVH(std::vector<Geom>& geoms, int start, int end) {
    BVHNode node;
    int nodeIndex = bvhNodes.size();
    bvhNodes.push_back(node);
    //printf("Building BVH from %d to %d\n", start, end);
    if (end - start == 1) {
        Geom& currentGeom = geoms[start];
        if(currentGeom.type == CUBE) {
			node.aabb = calculateAABBCubes(currentGeom);
            // No triangles for cubes
			node.triIndexStart = -1;
			node.triIndexEnd = -1;
		}
		else if(currentGeom.type == SPHERE) {
			node.aabb = calculateAABBSpheres(currentGeom);
            // No triangles for spheres
            node.triIndexStart = -1;
            node.triIndexEnd = -1;
		}
		else if(currentGeom.type == MESH) {
			node.aabb = calculateAABBMeshs(currentGeom);
            node.triIndexStart = currentGeom.triIndexStart;
            node.triIndexEnd = currentGeom.triIndexEnd;
		}
        node.isLeaf = true;
        node.left = -1;
        node.right = -1;

        bvhNodes[nodeIndex] = node;
        return nodeIndex;
    }
    else {
        Geom& currentGeom = geoms[start];
  //      if (currentGeom.type == CUBE) {
  //          node.aabb = calculateAABBCubes(currentGeom);
  //      }else if(currentGeom.type == SPHERE) {
		//	node.aabb = calculateAABBSpheres(currentGeom);
		//}
		//else if (currentGeom.type == MESH) {
		//	node.aabb = calculateAABBMeshs(currentGeom);
		//}
        
        // Reference from: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
        //determine split axis and position
        glm::vec3 extent = node.aabb.max - node.aabb.min;
        //start with x axis
        int axis = 0;
        // y axis is bigger
        if (extent.y > extent.x) axis = 1;
        // z axis is bigger
        if (extent.z > extent[axis]) axis = 2;
        float splitPos = node.aabb.min[axis] + extent[axis] * 0.5f;
        //sort the primitives
        std::sort(geoms.begin() + start, geoms.begin() + end, [this,axis](const Geom& a, const Geom& b) {
            return a.getCentroid(triangles)[axis] < b.getCentroid(triangles)[axis];
         });

        int mid = start + (end - start) / 2;
        node.left = buildBVH(geoms, start, mid);
        node.right = buildBVH(geoms, mid, end);
        node.isLeaf = false;

        node.aabb = mergeAABBs(bvhNodes[node.left].aabb, bvhNodes[node.right].aabb);

		bvhNodes[nodeIndex] = node;
		return nodeIndex;
    }
}
#endif

void Scene::UpdateNodeBounds(int nodeIdx) {
    BVHNode& node = bvhNodes[nodeIdx];
    node.aabb.min = glm::vec3(1e30f);
    node.aabb.max = glm::vec3(1e30f);
    //for (int first = node.triIndexStart, i = 0; i < node.triIndexEnd; i++)
    for (int i = node.triIndexStart; i < node.triIndexEnd; i++)
    {
        //Triangle& leafTri = triangles[first + i];
        Triangle& leafTri = triangles[triIdx[i]];
        node.aabb.min = min(node.aabb.min, leafTri.verts[0]);
        node.aabb.min = min(node.aabb.min, leafTri.verts[1]);
        node.aabb.min = min(node.aabb.min, leafTri.verts[2]);
        node.aabb.max = max(node.aabb.max, leafTri.verts[0]);
        node.aabb.max = max(node.aabb.max, leafTri.verts[1]);
        node.aabb.max = max(node.aabb.max, leafTri.verts[2]);
    }
}
int rootNodeIdx = 0, nodesUsed = 0;
void Scene::Subdivide(int nodeIdx) {
    // terminate recursion
    BVHNode& node = bvhNodes[nodeIdx];
    int triCount = node.triIndexEnd - node.triIndexStart;
    if (triCount <= 2) return;
    // determine split axis and position
    glm::vec3 extent = node.aabb.max - node.aabb.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = node.aabb.min[axis] + extent[axis] * 0.5f;
    // in-place partition
    int i = node.triIndexStart;
    //int j = i + node.triCount - 1;
    int j = i + triCount - 1;
    while (i <= j)
    {
        if (triangles[triIdx[i]].centroid[axis] < splitPos)
            i++;
        else
            swap(triIdx[i], triIdx[j--]);
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.triIndexStart;
    if (leftCount == 0 || leftCount == triCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    printf("Subdivide node %d into %d and %d\n", nodeIdx, leftChildIdx, rightChildIdx);
    bvhNodes[leftChildIdx].triIndexStart = node.triIndexStart;
    //bvhNodes[leftChildIdx].triCount = leftCount;
    bvhNodes[leftChildIdx].triIndexEnd = i;
    bvhNodes[rightChildIdx].triIndexStart = i;
    bvhNodes[rightChildIdx].triIndexEnd = node.triIndexEnd;
    node.left = leftChildIdx;
    //???
    node.right = rightChildIdx;
    //node.triCount = 0;
    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);
    // recurse
    Subdivide(leftChildIdx);
    Subdivide(rightChildIdx);
}

//int rootNodeIdx = 0, nodesUsed = 0;
void Scene::BuildBVH() {
    const int triSize = triangles.size();
    for (int i = 0; i < triangles.size(); ++i) {
        //triIdx[i] = i;  // Assign each triangle an index
        triIdx.push_back(i);
    }
    //BVHNode bvhNode[size * 2 - 1];
    bvhNodes.resize(triSize * 2 - 1);
    BVHNode& root = bvhNodes[rootNodeIdx];
    root.left = 0, root.right = 0;
    // Loop all??? or just triangles from one mesh
    root.triIndexStart = 0;
    root.triIndexEnd = triSize;
    UpdateNodeBounds(rootNodeIdx);
    Subdivide(rootNodeIdx);
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
//---------------------Assign materials from JSON---------------------
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
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            const auto& specCol = p["SPECRGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasRefractive = 1.0f;
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            const auto& specCol = p["SPECRGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasRefractive = 1.0f;
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Reflective")
        {
            const auto& col = p["RGB"];
            //newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.exponent = p["EXPONENT"];
            newMaterial.hasReflective = 1.0f;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);

    }
//---------------------Assign camera and objects from JSON---------------------
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            //std::cout << "CUBE MATERIALID is:" << newGeom.materialid << endl;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            //std::cout << "SPHERE MATERIALID is:" << newGeom.materialid << endl;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            //Add for normal map not from mtl file
            if (p.contains("NORMALMAP")) {
                newGeom.hasNormal = 1;
                std::cout << "Loaded normal map from " << p["NORMALMAP"] << endl;
                loadNormal(p["NORMALMAP"], newGeom, "../scenes/");
                std::cout << "normal map id is " << newGeom.normalid << endl;
            }

            if (p.contains("MATERIAL")) {;
                newGeom.materialid = MatNameToID[p["MATERIAL"]];
                std::cout << "MESH MATERIALID is:" << newGeom.materialid << endl;
            }
            //Loading vertices, normals, uvs and Read mtl file
            loadFromOBJ(p["OBJ"], newGeom, MatNameToID);
            //std::cout << "Loaded mesh from " << p["OBJ"] << endl;
            
            //Add for texture not from mtl file
            if (p.contains("TEXTURE")) {
                newGeom.hasTexture = 1;
                std::cout << "Loaded texture from " << p["TEXTURE"] << endl;
                loadTexture(p["TEXTURE"], newGeom, "../scenes/");
                std::cout << "texture id is " << newGeom.textureid << endl;
            }

        }
        //newGeom.materialid = MatNameToID[p["MATERIAL"]];
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
#if 0
    //print out all the materials
    for (int i = 0; i < materials.size(); ++i) {
        cout << "Material " << i << endl;
		cout << "Color: " << glm::to_string(materials[i].color) << endl;
		cout << "Emittance: " << materials[i].emittance << endl;
		cout << "IndexOfRefraction: " << materials[i].indexOfRefraction << endl;
		cout << "HasRefractive: " << materials[i].hasRefractive << endl;
		cout << "Specular Color: " << glm::to_string(materials[i].specular.color) << endl;
		cout << "Specular Exponent: " << materials[i].specular.exponent << endl;
		cout << "HasReflective: " << materials[i].hasReflective << endl;
		cout << " " << endl;
    }
#endif

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

void Scene::loadTexture(const std::string& filename, Geom& newGeom, std::string path) {
    int width, height, channels;
    unsigned char* data = stbi_load((path + filename).c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << path + filename << std::endl;
        exit(1);
    }

    // Create a new texture
    Texture newTexture;
    newTexture.width = width;
    newTexture.height = height;
    newTexture.channels = channels;
    newTexture.data = data;

    // Add the texture to the scene
    textures.push_back(newTexture);
    newGeom.textureid = textures.size() - 1;
}

void Scene::loadNormal(const std::string& filename, Geom& newGeom, std::string path) {
    int width, height, channels;
    unsigned char* data = stbi_load((path + filename).c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load normal map: " << path + filename << std::endl;
        exit(1);
    }

    // Create a new texture
    Texture newTexture;
    newTexture.width = width;
    newTexture.height = height;
    newTexture.channels = channels;
    newTexture.data = data;

    // Add the texture to the scene
    normals.push_back(newTexture);
    newGeom.normalid = normals.size() - 1;
}

// Reference to the tinyobj loader example:
// https://github.com/tinyobjloader/tinyobjloader/blob/release/examples/viewer/viewer.cc

void Scene::loadFromOBJ(const std::string& filename, Geom& newGeom, std::unordered_map<std::string, uint32_t>& MatNameToID) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tobj_materials;
    std::string warn, err;
    const std::string path = "../scenes/";
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &tobj_materials, &warn, &err, (path + filename).c_str(), path.c_str());
    cout<< "Loading from OBJ: " << filename << endl;

    if (!warn.empty()) {
        std::cout << "WARNING: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load " << filename << std::endl;
        exit(1);
    }
#if 0
    // Print info
    printf("# of vertices  = %d\n", (int)(attrib.vertices.size()) / 3);
    printf("# of normals   = %d\n", (int)(attrib.normals.size()) / 3);
    printf("# of texcoords = %d\n", (int)(attrib.texcoords.size()) / 2);
    printf("# of materials = %d\n", (int)tobj_materials.size());
    printf("# of shapes    = %d\n", (int)shapes.size());
#endif

    if (!tobj_materials.empty()){
        for (size_t matID = 0; matID < tobj_materials.size(); matID++) {
            const tinyobj::material_t& mat = tobj_materials[matID];
    #if 0
            // Print material name: newmtl name
            printf("material[%d].name = %s\n", int(matID), mat.name.c_str());

            // Print ambient color: Ka
            printf("material[%d].ambient = (%f, %f, %f)\n", int(matID),
                mat.ambient[0], mat.ambient[1], mat.ambient[2]);

            // Print diffuse color: Kd
            printf("material[%d].diffuse = (%f, %f, %f)\n", int(matID),
                mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);

            // Print specular color: Ks
            printf("material[%d].specular = (%f, %f, %f)\n", int(matID),
                mat.specular[0], mat.specular[1], mat.specular[2]);

            // Print index of refraction: Ni
            printf("material[%d].ior = %f\n", int(matID), mat.ior);

            // Print Transparency: Tr
            printf("material[%d].Transparency = (%f, %f, %f)\n", int(matID),
			    mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);

            // Print illumination mode: illum
            printf("material[%d].illum = %d\n", int(matID), mat.illum);

            // Print specular exponent: Ns
            printf("material[%d].shininess = %f\n", int(matID), mat.shininess);


    #endif
            Material geoMat{};

            /***********************************************
             * Mapping tinyobj::material_t to Material struct
             *
             * tinyobj::material_t        |    Material
             * ------------------------------------------------
             * mat.diffuse(Kd)            | geoMat.color
             * - Diffuse color            | - Base color
             * ------------------------------------------------
             * mat.emission               | geoMat.emittance
             * - Emissive color           | - Light emission
             * ------------------------------------------------
             * mat.transmittance          | geoMat.hasRefractive
             * - Transparency             | - Refractive flag
             * ------------------------------------------------
             * mat.ior                    | geoMat.indexOfRefraction
             * - Index of refraction      | - Refraction index
             * ------------------------------------------------
             * mat.specular               | geoMat.specular.color
             * - Specular color           | - Highlight color
             * ------------------------------------------------
             * mat.shininess              | geoMat.specular.exponent
             * - Shininess factor         | - Specular exponent
             * ------------------------------------------------
             * mat.illum                  | Illumination model
             * - Illumination mode        | - Reflect/refract flag
             ***********************************************/


            // Set the diffuse color (Kd) as the base color
            geoMat.color = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);

            // Handle emittance like light sources
            glm::vec3 emissive = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
            geoMat.emittance = glm::length(emissive) > 0.0f ? glm::length(emissive) : 0.0f;

            // Handle transparency and refraction
            glm::vec3 transparency = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
            if (glm::length(transparency) > 0.0f) {
                geoMat.hasRefractive = 1.0f;
                geoMat.indexOfRefraction = mat.ior; 
            }
            else {
                geoMat.hasRefractive = 0.0f;
                geoMat.indexOfRefraction = 1.0f;
            }

            if (mat.illum == 1) {
                // No specular reflection, only diffuse
                geoMat.hasReflective = 0.0f;
            }
            else if (mat.illum == 2) {
                // Diffuse and specular reflection
                geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
                geoMat.specular.exponent = mat.shininess;
                geoMat.hasReflective = mat.shininess > 0.0f ? 1.0f : 0.0f;
            }
            else if (mat.illum == 3 || mat.illum == 4) {
                // Transparency and reflection (diffuse + transparency)
                geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
                geoMat.specular.exponent = mat.shininess;
                geoMat.hasReflective = 1.0f;
                geoMat.hasRefractive = 1.0f;
            }
            else if (mat.illum == 5 || mat.illum == 6) {
                geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
                geoMat.specular.exponent = mat.shininess;
                geoMat.hasReflective = 1.0f;
            }
            else {
                // Default case for other illumination models
                geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
                geoMat.specular.exponent = mat.shininess;
                geoMat.hasReflective = mat.shininess > 0.0f ? 1.0f : 0.0f;
            }

            materials.emplace_back(geoMat);
            MatNameToID[mat.name] = materials.size() - 1;
            if (!mat.diffuse_texname.empty()) {
                std::cout << "Loading texture in loadFromOBJ!: " << mat.diffuse_texname << std::endl;
			    loadTexture(mat.diffuse_texname, newGeom, path);
			    newGeom.hasTexture = 1;
            }
        }
    }
    else {
        std::cerr << "No materials found in " << filename << std::endl;
    }

    // Start of triangle indices for this geometry
    newGeom.triIndexStart = triangles.size();

    for (const auto& shape : shapes) {
        //int numTrianglesInShape = shape.mesh.indices.size() / 3;
        //std::cout << "Triangles in shape: " << numTrianglesInShape << std::endl;
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            int idx0 = shape.mesh.indices[i].vertex_index;
            int idx1 = shape.mesh.indices[i + 1].vertex_index;
            int idx2 = shape.mesh.indices[i + 2].vertex_index;

            // Vertices
            glm::vec3 v0(attrib.vertices[3 * idx0], attrib.vertices[3 * idx0 + 1], attrib.vertices[3 * idx0 + 2]);
            glm::vec3 v1(attrib.vertices[3 * idx1], attrib.vertices[3 * idx1 + 1], attrib.vertices[3 * idx1 + 2]);
            glm::vec3 v2(attrib.vertices[3 * idx2], attrib.vertices[3 * idx2 + 1], attrib.vertices[3 * idx2 + 2]);
            //std::cout << "V0: " << v0.x << ", " << v0.y << ", " << v0.z << std::endl;
            //std::cout << "V1: " << v1.x << ", " << v1.y << ", " << v1.z << std::endl;
            //std::cout << "V2: " << v2.x << ", " << v2.y << ", " << v2.z << std::endl;

            // UVs
            glm::vec2 uv0(0.0f), uv1(0.0f), uv2(0.0f);
            if (!attrib.texcoords.empty()) {
                int texIdx0 = shape.mesh.indices[i].texcoord_index;
                int texIdx1 = shape.mesh.indices[i + 1].texcoord_index;
                int texIdx2 = shape.mesh.indices[i + 2].texcoord_index;

                uv0 = glm::vec2(attrib.texcoords[2 * texIdx0], attrib.texcoords[2 * texIdx0 + 1]);
                uv1 = glm::vec2(attrib.texcoords[2 * texIdx1], attrib.texcoords[2 * texIdx1 + 1]);
                uv2 = glm::vec2(attrib.texcoords[2 * texIdx2], attrib.texcoords[2 * texIdx2 + 1]);
                //std::cout << "UV0: " << uv0.x << ", " << uv0.y << std::endl;
                //std::cout << "UV1: " << uv1.x << ", " << uv1.y << std::endl;
                //std::cout << "UV2: " << uv2.x << ", " << uv2.y << std::endl;
            }

            // Normals
            glm::vec3 n0(0.0f), n1(0.0f), n2(0.0f);
            if (!attrib.normals.empty()) {
                int normIdx0 = shape.mesh.indices[i].normal_index;
                int normIdx1 = shape.mesh.indices[i + 1].normal_index;
                int normIdx2 = shape.mesh.indices[i + 2].normal_index;

                n0 = glm::vec3(attrib.normals[3 * normIdx0], attrib.normals[3 * normIdx0 + 1], attrib.normals[3 * normIdx0 + 2]);
                n1 = glm::vec3(attrib.normals[3 * normIdx1], attrib.normals[3 * normIdx1 + 1], attrib.normals[3 * normIdx1 + 2]);
                n2 = glm::vec3(attrib.normals[3 * normIdx2], attrib.normals[3 * normIdx2 + 1], attrib.normals[3 * normIdx2 + 2]);
            }

            // Tangents and Bitangents
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec2 deltaUV1 = uv1 - uv0;
            glm::vec2 deltaUV2 = uv2 - uv0;
            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            glm::vec3 tangent;
            tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            tangent = glm::normalize(tangent);
            glm::vec3 bitangent;
            bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
            bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
            bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
            bitangent = glm::normalize(bitangent);

            // Centroid
            glm::vec3 centroid = (v0 + v1 + v2) / 3.0f;

            triangles.push_back({
            {v0, v1, v2},  
            {uv0, uv1, uv2},  
            {n0, n1, n2},  
            tangent,  
            bitangent,
            centroid
             });
        }
    }

    // End of triangle indices for this geometry
    newGeom.triIndexEnd = triangles.size();

    //currently one mesh has one material
    if (!tobj_materials.empty()) {
        int materialID = shapes[0].mesh.material_ids[0];
        newGeom.materialid = MatNameToID[tobj_materials[materialID].name];
    }
}