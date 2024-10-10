#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include <tiny_gltf.h>

using json = nlohmann::json;

void Scene::loadGLTFMesh(const std::string& og_filename, Geom &newGeom) {
    std::string filePrefix = og_filename.substr(0, og_filename.find_last_of("/") + 1);
	tinygltf::TinyGLTF loader;
	tinygltf::Model model;
	std::string err;
	std::string warn;

    // Try loading the file
	bool success = loader.LoadASCIIFromFile(&model, &err, &warn, og_filename);
	if (!warn.empty()) {
		std::cerr << "Warning: " << warn << std::endl;
	}
	if (!err.empty()) {
		std::cerr << "Error: " << err << std::endl;
	}
	if (!success) {
        return;
	}

	// Load each mesh
    for (const auto& mesh : model.meshes) {
		newGeom.triangleStart = triangles.size();
        for (auto& primitive : mesh.primitives) {
            int posAccessorIndex = primitive.attributes.at("POSITION");
            tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];
            tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            tinygltf::Buffer& positionBuffer = model.buffers[posBufferView.buffer];
            float* positions = reinterpret_cast<float*>(&(positionBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]));

			// Load Albedo Texture
			int index = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index;
			if (primitive.material >= 0 && index >= 0) {
				tinygltf::Texture& texture = model.textures[index];
                Texture tex;
                tex.id = textures.size();
				tex.startIndex = textureData.size();
                newGeom.usesTexture = true;
                newGeom.textureStart = tex.id;
				float* albedoTexture = stbi_loadf((filePrefix + model.images[texture.source].uri).c_str(), &tex.width, &tex.height, &tex.numChannels, 0);
				for (int i = 0; i < tex.width * tex.height; i++) {
					textureData.push_back(glm::vec3(albedoTexture[i * tex.numChannels], albedoTexture[i * tex.numChannels + 1], albedoTexture[i * tex.numChannels + 2]));
				}
				tex.endIndex = textureData.size() - 1;
				textures.push_back(tex);
				stbi_image_free(albedoTexture);
            }

            if (primitive.indices >= 0) {
				tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
				tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
				tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
				if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
					uint16_t* indices = reinterpret_cast<uint16_t*>(&(indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]));
					populateTriangleData(indices, model, indexAccessor, primitive, positions, newGeom);
				}
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
					uint32_t* indices = reinterpret_cast<uint32_t*>(&(indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]));
                    populateTriangleData(indices, model, indexAccessor, primitive, positions, newGeom);
				}
                else {
					cout << "Index component type " << indexAccessor.componentType << " not supported" << endl;
                    return;
                }
			}
        }

		newGeom.triangleEnd = triangles.size() - 1;
        cout << "This mesh has : " << newGeom.triangleEnd - newGeom.triangleStart + 1 << " triangles" << endl;
        cout << "Total triangles : " << triangles.size() << " triangles" << endl;

    }
}

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
		else if (p["TYPE"] == "Mirror")
		{
			newMaterial.hasReflective = 1;
            newMaterial.hasRefractive = 0;
			newMaterial.hasPlastic = 0;
			newMaterial.color = glm::vec3(1.0f);
			newMaterial.roughness = p["ROUGHNESS"];
		}
		else if (p["TYPE"] == "Plastic")
		{
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0;
            newMaterial.hasRefractive = 0;
            newMaterial.hasPlastic = 1;
            newMaterial.roughness = p["ROUGHNESS"];
		}
		else if (p["TYPE"] == "Glass")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1;
			newMaterial.hasRefractive = 1;
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.dispersion = p["DISPERSION"];
		}
		else if (p["TYPE"] == "Transmissive")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasReflective = 0;
			newMaterial.hasRefractive = 1;
			newMaterial.hasPlastic = 0;
		}
		else
		{
			std::cerr << "Unknown material type: " << p["TYPE"] << std::endl;
			exit(-1);
		}
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& mapData = data["EnvironmentMap"];
    // Add case to make sure data is not empty or null

	if (!mapData.empty()) {
        const auto& file = mapData["FILE"];
        std::string str = file;
        envMap = std::make_unique<Texture>();
        envMap->id = textures.size();
        envMap->startIndex = envMapData.size();
        float* envTexture = stbi_loadf(str.c_str(), &envMap->width, &envMap->height, &envMap->numChannels, 0);
        for (int i = 0; i < envMap->width * envMap->height; i++) {
            envMapData.push_back(glm::vec4(envTexture[i * envMap->numChannels], envTexture[i * envMap->numChannels + 1], envTexture[i * envMap->numChannels + 2], envTexture[i * envMap->numChannels + 3]));
        }
        envMap->endIndex = envMapData.size() - 1;
        stbi_image_free(envTexture);
	}

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.meshid = geoms.size();
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

        if (type == "cube")
        {
            newGeom.type = CUBE;
            // Add one triangle to the triangles array that takes up the same
            // bounding box as the cube for the BVH. Make sure that the
            // Triangle has some reference to the geoms index that this cube corresponds to.
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            // Add one triangle to the triangles array that takes up the same
            // bounding box as the sphere for the BVH. Make sure that the
            // Triangle has some reference to the geoms index that this sphere corresponds to.
        }
		else if (type == "mesh")
		{
            newGeom.type = MESH;
			loadGLTFMesh(p["FILE"], newGeom);
		}

        geoms.push_back(newGeom);
    }

    // If there is a mesh, build BVH
	if (triangles.size() > 0)
    {
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		bvhNodes.resize(triangles.size() * 2 - 1);
		buildBVH();
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "BVH built in " << duration << "ms" << std::endl;
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

void Scene::buildBVH() {
    // Populate Triangle Index Array
    //for (int i = 0; i < triangles.size(); i++) { triangleIndices[i] = i; }
    BVHNode& root = bvhNodes[0];
	root.leftFirst = 0;
	root.numTriangles = triangles.size();
    updateNodeBounds(0);
    subdivideBounds(0);
}

void Scene::updateNodeBounds(int nodeIdx) {
    BVHNode& node = bvhNodes[nodeIdx];
    for (int i = 0; i < node.numTriangles; i++) {
		const Triangle& tri = triangles[node.leftFirst + i];
        node.aabb.grow(tri.v1.pos);
        node.aabb.grow(tri.v2.pos);
        node.aabb.grow(tri.v3.pos);
	}
}

float Scene::EvaluateSAH(BVHNode& node, int axis, float pos)
{
    // Figure out triangle counts and bounds for this split candidate
    aabb leftBox, rightBox;
    int leftCount = 0, rightCount = 0;
    for (int i = 0; i < node.numTriangles; i++)
    {
        Triangle& tri = triangles[node.leftFirst + i];
        if (tri.centroid[axis] < pos)
        {
            leftCount++;
            leftBox.grow(tri.v1.pos);
            leftBox.grow(tri.v2.pos);
            leftBox.grow(tri.v3.pos);
        }
        else
        {
            rightCount++;
            rightBox.grow(tri.v1.pos);
            rightBox.grow(tri.v2.pos);
            rightBox.grow(tri.v3.pos);
        }
    }
    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0 ? cost : 1e30f;
}

float Scene::FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos)
{
    float bestCost = 1e30f;
    for (int a = 0; a < 3; a++)
    {

        float boundsMin = 1e30f, boundsMax = -1e30f;
        for (int i = 0; i < node.numTriangles; i++)
        {
            Triangle& tri = triangles[node.leftFirst + i];
            boundsMin = glm::min(boundsMin, tri.centroid[a]);
            boundsMax = glm::max(boundsMax, tri.centroid[a]);
        }
        if (boundsMin == boundsMax) continue;
        // populate the bins
		const int BINS = 8;
        Bin bin[BINS];
        float scale = BINS / (boundsMax - boundsMin);
        for (int i = 0; i < node.numTriangles; i++)
        {
            Triangle& tri = triangles[node.leftFirst + i];
            int binIdx = min(BINS - 1,
                (int)((tri.centroid[a] - boundsMin) * scale));
            bin[binIdx].triCount++;
            bin[binIdx].bounds.grow(tri.v1.pos);
            bin[binIdx].bounds.grow(tri.v2.pos);
            bin[binIdx].bounds.grow(tri.v3.pos);
        }
        // gather data for the 7 planes between the 8 bins
        float leftArea[BINS - 1], rightArea[BINS - 1];
        int leftCount[BINS - 1], rightCount[BINS - 1];
        aabb leftBox, rightBox;
        int leftSum = 0, rightSum = 0;
        for (int i = 0; i < BINS - 1; i++)
        {
            leftSum += bin[i].triCount;
            leftCount[i] = leftSum;
            leftBox.grow(bin[i].bounds.bmin);
            leftBox.grow(bin[i].bounds.bmax);
            leftArea[i] = leftBox.area();
            rightSum += bin[BINS - 1 - i].triCount;
            rightCount[BINS - 2 - i] = rightSum;
            rightBox.grow(bin[BINS - 1 - i].bounds.bmin);
            rightBox.grow(bin[BINS - 1 - i].bounds.bmax);
            rightArea[BINS - 2 - i] = rightBox.area();
        }
        // calculate SAH cost for the 7 planes
        scale = (boundsMax - boundsMin) / BINS;
        for (int i = 0; i < BINS - 1; i++)
        {
            float planeCost =
                leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
            if (planeCost < bestCost)
                axis = a, splitPos = boundsMin + scale * (i + 1),
                bestCost = planeCost;
        }
    }
    return bestCost;
}

void Scene::subdivideBounds(int nodeIdx) {
    BVHNode& node = bvhNodes[nodeIdx];
    
    if (node.numTriangles <= 8) {
        return;
    }

    // determine split axis using SAH
    int axis;
    float splitPos;
	float bestCost = FindBestSplitPlane(node, axis, splitPos);

    float parentArea = node.aabb.area();
    float parentCost = node.numTriangles * parentArea;
    if (bestCost >= parentCost) return;

    int i = node.leftFirst;
    int j = i + node.numTriangles - 1;

    while (i <= j) {
        if (triangles[i].centroid[axis] < splitPos) {
            i++;
        }
        else {
            std::swap(triangles[i], triangles[j--]);
        }
    }

    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.numTriangles) { 
        return;
    }

    // Make Children
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;

    bvhNodes[leftChildIdx].leftFirst = node.leftFirst;
    bvhNodes[leftChildIdx].numTriangles = leftCount;
    bvhNodes[leftChildIdx].aabb = bbox();
    bvhNodes[rightChildIdx].leftFirst = i;
    bvhNodes[rightChildIdx].numTriangles = node.numTriangles - leftCount;
    bvhNodes[rightChildIdx].aabb = bbox();

    node.leftFirst = leftChildIdx;
    node.numTriangles = 0;
    updateNodeBounds(leftChildIdx);
    updateNodeBounds(rightChildIdx);

    subdivideBounds(leftChildIdx);
    subdivideBounds(rightChildIdx);

}