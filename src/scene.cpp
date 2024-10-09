#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "stb_image.h"
using json = nlohmann::json;
using Model = tinygltf::Model;
using TinyGLTF = tinygltf::TinyGLTF;

Model model;
TinyGLTF loader;

Scene::Scene(string sceneFile, string envMapFile)
{
    cout << "Reading scene from " << sceneFile << " ..." << endl;
    cout << " " << endl;
    auto sceneExt = sceneFile.substr(sceneFile.find_last_of('.'));
	if (sceneExt == ".gltf" || sceneExt == ".glb")
	{
		loadFromGltf(sceneFile);
	}
	else
    {
        cout << "Couldn't read from " << sceneFile << endl;
        exit(-1);
    }

    cout << "Reading scene from " << envMapFile << " ..." << endl;
    cout << " " << endl;
	auto envMapExt = envMapFile.substr(envMapFile.find_last_of('.'));
    if (envMapExt == ".hdr") {
		// Load via stb_image
		int width, height, numComponents;
		float* data = stbi_loadf(envMapFile.c_str(), &width, &height, &numComponents, 0);
        if (!data) {
            std::cerr << "Failed to load HDR image" << std::endl;
            exit(-1);
        }

		envMap.width = width;
		envMap.height = height;
		envMap.numComponents = numComponents;
		envMap.size = width * height * numComponents;
		envMap.data.resize(envMap.size);
		for (int i = 0; i < width * height; ++i) {
			envMap.data[i].r = data[i * numComponents];
			envMap.data[i].g = data[i * numComponents + 1];
			envMap.data[i].b = data[i * numComponents + 2];
			envMap.data[i].a = (numComponents == 4) ? data[i * numComponents + 3] : 1.0f;
		}

		stbi_image_free(data);
    }

    buildBvh();
}

void Scene::loadFromGltf(const std::string& gltfName)
{
	std::string err;
	std::string warn;
	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
	if (!warn.empty())
	{
		std::cerr << "Warn: " << warn << std::endl;
	}

	if (!err.empty())
	{
		std::cerr << "Err: " << err << std::endl;
	}

	if (!ret)
	{
		std::cerr << "Failed to parse glTF file" << std::endl;
		return;
	}

    Camera& sceneCamera = state.camera;
	const auto& scene = model.scenes[model.defaultScene];

	for (const auto& material : model.materials)
	{
		Material newMaterial;
		const auto& pbr = material.pbrMetallicRoughness;
		newMaterial.color = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
		newMaterial.metallic = pbr.metallicFactor;
		newMaterial.roughness = pbr.roughnessFactor;

		const auto findRefractiveAttribute = material.extensions.find("KHR_materials_transmission");
		newMaterial.hasRefractive = findRefractiveAttribute != material.extensions.end();
        const auto findIorAttribute = material.extensions.find("KHR_materials_ior");
		newMaterial.indexOfRefraction = findIorAttribute != material.extensions.end() ? findIorAttribute->second.Get("ior").GetNumberAsDouble() : 1.0f;
		
		const auto findEmissiveAttribute = material.extensions.find("KHR_materials_emissive_strength");
		newMaterial.emittance = findEmissiveAttribute != material.extensions.end() ? findEmissiveAttribute->second.Get("emissiveStrength").GetNumberAsDouble() : 0.0f;
		newMaterial.emissiveFactor = findEmissiveAttribute != material.extensions.end() ? glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]) : glm::vec3(0.0f);

        // Access the base color texture
        if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            newMaterial.baseColorTextureId = textures.size();

            const auto& baseColorTexture = model.textures[material.pbrMetallicRoughness.baseColorTexture.index];
            const auto& baseColorImage = model.images[baseColorTexture.source];
            Texture colorTexture;
            colorTexture.width = baseColorImage.width;
            colorTexture.height = baseColorImage.height;
            colorTexture.numComponents = baseColorImage.component;
			colorTexture.size = baseColorImage.image.size();
            colorTexture.data.resize(colorTexture.width * colorTexture.height);

            for (int i = 0; i < colorTexture.width * colorTexture.height; ++i) {
                int index = i * colorTexture.numComponents;
                float r = baseColorImage.image[index] / 255.0f;
                float g = baseColorImage.image[index + 1] / 255.0f;
                float b = baseColorImage.image[index + 2] / 255.0f;
                float a = (colorTexture.numComponents == 4) ? baseColorImage.image[index + 3] / 255.0f : 1.0f;
                colorTexture.data[i] = glm::vec4(r, g, b, a);
            }


            textures.push_back(colorTexture);
        }

        // Access the normal texture
        if (material.normalTexture.index >= 0)
        {
			newMaterial.normalTextureId = textures.size();

            const auto& normTex = model.textures[material.normalTexture.index];
            const auto& normalImage = model.images[normTex.source];

            Texture normalTexture;
			normalTexture.width = normalImage.width;
			normalTexture.height = normalImage.height;
			normalTexture.numComponents = normalImage.component;
			normalTexture.size = normalImage.image.size();

            normalTexture.data.resize(normalTexture.width * normalTexture.height);

            for (int i = 0; i < normalTexture.width * normalTexture.height; ++i) {
                int index = i * normalTexture.numComponents;
                float r = normalImage.image[index] / 255.0f;
                float g = normalImage.image[index + 1] / 255.0f;
                float b = normalImage.image[index + 2] / 255.0f;
                float a = (normalTexture.numComponents == 4) ? normalImage.image[index + 3] / 255.0f : 1.0f;
                normalTexture.data[i] = glm::vec4(r, g, b, a);
            }

			textures.push_back(normalTexture);
        }

        if (material.emissiveTexture.index >= 0) {
			newMaterial.emissiveTextureId = textures.size();

            const auto& emissiveTex = model.textures[material.emissiveTexture.index];
			const auto& emissiveImage = model.images[emissiveTex.source];

			Texture emissiveTexture;
			emissiveTexture.width = emissiveImage.width;
			emissiveTexture.height = emissiveImage.height;
			emissiveTexture.numComponents = emissiveImage.component;
			emissiveTexture.size = emissiveImage.image.size();

			emissiveTexture.data.resize(emissiveTexture.width * emissiveTexture.height);

			for (int i = 0; i < emissiveTexture.width * emissiveTexture.height; ++i) {
				int index = i * emissiveTexture.numComponents;
				float r = emissiveImage.image[index] / 255.0f;
				float g = emissiveImage.image[index + 1] / 255.0f;
				float b = emissiveImage.image[index + 2] / 255.0f;
				float a = (emissiveTexture.numComponents == 4) ? emissiveImage.image[index + 3] / 255.0f : 1.0f;
				emissiveTexture.data[i] = glm::vec4(r, g, b, a);
			}

			textures.push_back(emissiveTexture);
		}

		materials.push_back(newMaterial);
	}

	for (const auto& node : scene.nodes)
	{
		const auto& n = model.nodes[node];
        Geom newGeom;
		newGeom.translation = (n.translation.size() == 0) ? glm::vec3(0.0f) : glm::vec3(n.translation[0], n.translation[1], n.translation[2]);
        if (n.rotation.size() == 4) {
			// GLM expects the quaternion in the order w, x, y, z, whereas GLTF provides it in the order x, y, z, w
            glm::quat quaternion(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]);
            newGeom.rotation = glm::degrees(glm::eulerAngles(quaternion));
        }
        else {
            newGeom.rotation = glm::vec3(0.0f); // Default rotation
        }		
        newGeom.scale = (n.scale.size() == 0) ? glm::vec3(1.0f) : glm::vec3(n.scale[0], n.scale[1], n.scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

		if (n.mesh >= 0)
		{
            newGeom.type = MESH;
			newGeom.meshId = meshes.size();
            Material newMaterial;
            Mesh newMesh;
            newMesh.vertStartIndex = vertices.size();
			newMesh.trianglesStartIndex = triangles.size();
            newMesh.baseColorUvIndex = baseColorUvs.size();
            newMesh.normalUvIndex = normalUvs.size();
            newMesh.emissiveUvIndex = emissiveUvs.size();

            const auto& mesh = model.meshes[n.mesh];
			for (const auto& primitive : mesh.primitives)
			{
				parsePrimitive(model, primitive, newMesh);
				// For now, we assume that each primitive has the same material
				newGeom.materialid = primitive.material;
			}
            
			geoms.push_back(newGeom);
			meshes.push_back(newMesh);
		}

		if (n.camera >= 0) {
			const auto& camera = model.cameras[n.camera];
			const auto& perspective = camera.perspective;
			sceneCamera.fov = glm::vec2(glm::degrees(perspective.yfov) * perspective.aspectRatio, glm::degrees(perspective.yfov));
			sceneCamera.position = glm::vec3(n.translation[0], n.translation[1], n.translation[2]);
			glm::quat quaternion = (n.rotation.size() == 4) ? glm::quat(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
			glm::vec3 forward = quaternion * glm::vec3(0.0f, 0.0f, -1.0f); // note that GLM overloads the * operator for quaternions, so this is effectively q_inv * v * q
            sceneCamera.lookAt = sceneCamera.position + forward;
            sceneCamera.view = glm::normalize(forward);
            sceneCamera.right = glm::normalize(quaternion * glm::vec3(1.0f, 0.0f, 0.0f));
            sceneCamera.up = glm::normalize(quaternion * glm::vec3(0.0f, 1.0f, 0.0f));
        }
	}

    // For now, just hard code other aspects of the scene
    sceneCamera.resolution.x = 800;
    sceneCamera.resolution.y = 800;
    state.iterations = 300;
    state.traceDepth = 8;
    state.imageName = "InteriorRoom";

    float yscaled = tan(sceneCamera.fov.y * (PI / 180));
    float xscaled = (yscaled * sceneCamera.resolution.x) / sceneCamera.resolution.y;
    sceneCamera.pixelLength = glm::vec2(2 * xscaled / (float)sceneCamera.resolution.x,
        2 * yscaled / (float)sceneCamera.resolution.y);

    sceneCamera.apertureRadius = 0.00;
    sceneCamera.focalLength = 4.08;

    //set up render camera stuff
    int arraylen = sceneCamera.resolution.x * sceneCamera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

}

template <typename T>
const T* getBufferData(const Model& model, const tinygltf::Accessor& accessor) {
	const auto& bufferView = model.bufferViews[accessor.bufferView];
	const auto& buffer = model.buffers[bufferView.buffer];
	return reinterpret_cast<const T*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
}

void Scene::parsePrimitive(const Model& model, const tinygltf::Primitive& primitive, Mesh& mesh)
{
    // Access the position attribute
    const auto& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
	const float* posData = getBufferData<float>(model, posAccessor);

    // Access the normal attribute
    const auto& normAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
	const float* normData = getBufferData<float>(model, normAccessor);

    // Populate vertices and normals
	int vertStartIdx = vertices.size();
    for (int i = 0; i < posAccessor.count; ++i)
    {
        vertices.push_back(glm::vec3(posData[i * 3], posData[i * 3 + 1], posData[i * 3 + 2]));
        normals.push_back(glm::vec3(normData[i * 3], normData[i * 3 + 1], normData[i * 3 + 2]));

        mesh.boundingBoxMax = glm::max(mesh.boundingBoxMax, vertices.back());
        mesh.boundingBoxMin = glm::min(mesh.boundingBoxMin, vertices.back());
    }

    // Access the indices
    const auto& indexAccessor = model.accessors[primitive.indices];
	const uint32_t* indexData32 = nullptr;
	const uint16_t* indexData16 = nullptr;
	const uint8_t* indexData8 = nullptr;
	if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
	{
		indexData16 = getBufferData<uint16_t>(model, indexAccessor);
	}
	else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
    {
		indexData32 = getBufferData<uint32_t>(model, indexAccessor);
	}
	else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
	{
		indexData8 = getBufferData<uint8_t>(model, indexAccessor);
	}   

    // Populate triangles
    for (int i = 0; i < indexAccessor.count; i += 3)
    {
        Triangle triangle;
        for (int j = 0; j < 3; ++j)
        {
			triangle.attributeIndex[j] = (indexData32) ? indexData32[i + j] 
				                       : (indexData16) ? indexData16[i + j] 
                                       : indexData8[i + j];  
        }
        
		// Used for BVH
        triangle.center = (vertices[vertStartIdx + triangle.attributeIndex[0]] + vertices[vertStartIdx + triangle.attributeIndex[1]] + vertices[vertStartIdx + triangle.attributeIndex[2]]) / 3.0f;

        triangles.push_back(triangle);
    }

    mesh.numTriangles += indexAccessor.count / 3;

    // Access the UVs
    // First the UVs for the baseColorTexture, then the UVs for the normalTexture
    if (primitive.material == -1) return;
    const auto& material = model.materials[primitive.material];

    if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
        int baseColorTexCoords = material.pbrMetallicRoughness.baseColorTexture.texCoord;
        auto baseColorAttrIt = primitive.attributes.find("TEXCOORD_" + std::to_string(baseColorTexCoords));
        if (baseColorAttrIt == primitive.attributes.end()) return;

        const auto& baseColorAccessor = model.accessors[baseColorAttrIt->second];
        const auto& baseColorBufferView = model.bufferViews[baseColorAccessor.bufferView];
        const auto& baseColorBuffer = model.buffers[baseColorBufferView.buffer];
        const float* baseColorData = reinterpret_cast<const float*>(&baseColorBuffer.data[baseColorBufferView.byteOffset + baseColorAccessor.byteOffset]);

        for (int i = 0; i < baseColorAccessor.count; ++i)
        {
            baseColorUvs.push_back(glm::vec2(baseColorData[i * 2], baseColorData[i * 2 + 1]));
        }
    }

    if (material.normalTexture.index >= 0) {
        int normalTexCoords = material.normalTexture.texCoord;
        auto normalAttrIt = primitive.attributes.find("TEXCOORD_" + std::to_string(normalTexCoords));
        if (normalAttrIt == primitive.attributes.end()) return;

        const auto& normalAccessor = model.accessors[normalAttrIt->second];
        const auto& normalBufferView = model.bufferViews[normalAccessor.bufferView];
        const auto& normalBuffer = model.buffers[normalBufferView.buffer];
        const float* normalData = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset]);

        for (int i = 0; i < normalAccessor.count; ++i)
        {
            normalUvs.push_back(glm::vec2(normalData[i * 2], normalData[i * 2 + 1]));
        }
    }

    if (material.emissiveTexture.index >= 0) {
        int emissiveTexCoords = material.emissiveTexture.texCoord;
        auto emissiveAttrIt = primitive.attributes.find("TEXCOORD_" + std::to_string(emissiveTexCoords));
        if (emissiveAttrIt == primitive.attributes.end()) return;

        const auto& emissiveAccessor = model.accessors[emissiveAttrIt->second];
        const auto& emissiveBufferView = model.bufferViews[emissiveAccessor.bufferView];
        const auto& emissiveBuffer = model.buffers[emissiveBufferView.buffer];
        const float* emissiveData = reinterpret_cast<const float*>(&emissiveBuffer.data[emissiveBufferView.byteOffset + emissiveAccessor.byteOffset]);

        for (int i = 0; i < emissiveAccessor.count; ++i)
        {
            emissiveUvs.push_back(glm::vec2(emissiveData[i * 2], emissiveData[i * 2 + 1]));
        }
    }
}

void Scene::buildBvh() {
	for (int i = 0; i < meshes.size(); ++i) {
		Mesh& mesh = meshes[i];
		BvhNode rootNode;
		rootNode.trianglesStartIdx = mesh.trianglesStartIndex;
		rootNode.numTriangles = mesh.numTriangles;
		rootNode.min = mesh.boundingBoxMin;
		rootNode.max = mesh.boundingBoxMax;
		rootNode.leftChild = -1;
		rootNode.rightChild = -1;
		mesh.bvhRootIndex = bvhNodes.size();
		bvhNodes.push_back(rootNode);

		splitNode(mesh, bvhNodes.size() - 1, 0);
	}
}

void Scene::splitNode(const Mesh& mesh, int parentIdx, int depth) {
	if (parentIdx == -1) return;

	BvhNode& parent = bvhNodes.at(parentIdx);
	if (parent.numTriangles <= MAX_TRIANGLES_PER_LEAF || depth >= MAX_BVH_DEPTH) return;

	glm::vec3 extent = parent.max - parent.min;
    int splitAxis = 0;
    if (extent.y > extent.x) splitAxis = 1;
    if (extent.z > extent.y && extent.z > extent.x) splitAxis = 2;
    float splitPos = (parent.min[splitAxis] + parent.max[splitAxis]) / 2.0f;

    // Create two child nodes and use their trianglesStartIdx to track the starting points as we
	// sort the triangles by the split axis
	BvhNode leftChild;
	BvhNode rightChild;
	leftChild.trianglesStartIdx = parent.trianglesStartIdx;
	rightChild.trianglesStartIdx = parent.trianglesStartIdx;
    
    for (int i = parent.trianglesStartIdx; i < parent.trianglesStartIdx + parent.numTriangles; ++i) {
		const Triangle& triangle = triangles[i];
		bool left = (triangle.center[splitAxis] < splitPos);
		BvhNode& child = left ? leftChild : rightChild;
		child.numTriangles++;

		glm::vec3 triangleMin = glm::min(glm::min(vertices[mesh.vertStartIndex + triangle.attributeIndex[0]], vertices[mesh.vertStartIndex + triangle.attributeIndex[1]]), vertices[mesh.vertStartIndex + triangle.attributeIndex[2]]);
		glm::vec3 triangleMax = glm::max(glm::max(vertices[mesh.vertStartIndex + triangle.attributeIndex[0]], vertices[mesh.vertStartIndex + triangle.attributeIndex[1]]), vertices[mesh.vertStartIndex + triangle.attributeIndex[2]]);

		child.min = glm::min(child.min, triangleMin);
		child.max = glm::max(child.max, triangleMax);

        if (!left) continue;

		// Swap the triangle with the first triangle in the right child,
		// and increment the right child's start index
		std::swap(triangles[i], triangles[rightChild.trianglesStartIdx]);
		rightChild.trianglesStartIdx++;
    }

    // Pushing to bvhNodes invalidates the reference to 'parent', so we need to access it via its index directly.
    if (leftChild.numTriangles > 0) {
	    bvhNodes.at(parentIdx).leftChild = bvhNodes.size();
	    bvhNodes.push_back(leftChild);
    }
    else {
		bvhNodes.at(parentIdx).leftChild = -1;
    }

    if (rightChild.numTriangles > 0) {
        bvhNodes.at(parentIdx).rightChild = bvhNodes.size();
	    bvhNodes.push_back(rightChild);
    }
    else {
		bvhNodes.at(parentIdx).rightChild = -1;
    }

    // Stop early if the bounding box is the same as the parent's bounding box
	// E.g. for an axis-aligned cube.
	if (leftChild.min == bvhNodes.at(parentIdx).min && leftChild.max == bvhNodes.at(parentIdx).max) return;

	splitNode(mesh, bvhNodes.at(parentIdx).leftChild, depth + 1);
	splitNode(mesh, bvhNodes.at(parentIdx).rightChild, depth + 1);
}