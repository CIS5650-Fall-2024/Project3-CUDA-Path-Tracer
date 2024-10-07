#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_obj_loader.h"
#include <stb_image.h>
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

GeomType Scene::getGeometryType(const std::string& type) {
    if (type == "sphere") return SPHERE;
    else if (type == "cube") return CUBE;
    else if (type == "obj") return OBJ;
    return ERROR;
}
int Scene::loadTexture(const std::string& name)
{
    // file path
    std::string file_path = "../scenes/";
    file_path += std::string("textures/") + name + std::string(".png");
    // Initialize the texture object and load the texture data.
    Texture_Data m_texture;
    float* data = stbi_loadf(file_path.c_str(), &m_texture.width, &m_texture.height, nullptr, 4);
    m_texture.index = pixels.size();
    for (int i = 0; i < m_texture.width * m_texture.height; ++i) {
        pixels.emplace_back(
            data[i * 4 + 0], // R
            data[i * 4 + 1], // G
            data[i * 4 + 2], // B
            data[i * 4 + 3]  // A
        );
    }
    // debug
    // Output a success message.
    std::cout << "Loaded " << m_texture.width << " x " << m_texture.height << " pixels from " << file_path << std::endl;
    stbi_image_free(data);
    m_textures.push_back(m_texture);
    return m_textures.size() - 1;
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
        // handle materials loading differently
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
            const auto& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f - roughness;
        }
        else if (p["TYPE"] == "Refractive") 
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        if (p.contains("ALBEDO")) {
            // debug
            std::cout << "Material " << name << " contains ALBEDO: " << p["ALBEDO"] << std::endl;
            newMaterial.albedo = loadTexture(p["ALBEDO"]);
        }
        if (p.contains("NORMAL")) {
            // debug
            std::cout << "Material " << name << " contains NORMAL: " << p["NORMAL"] << std::endl;
            newMaterial.normal = loadTexture(p["NORMAL"]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // handle the mesh object
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom m_geometry;
        GeomType geomType = getGeometryType(type);
        switch (geomType)
        {
        case SPHERE:
            m_geometry.type = SPHERE;
            break;
        case CUBE:
            m_geometry.type = CUBE;
            break;
        case OBJ:
            m_geometry.type = OBJ;
            break;
        case ERROR:
            std::cerr << "Geometry Type Error!" << std::endl;
            break;
        default:
            break;
        }

        m_geometry.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        m_geometry.translation = glm::vec3(trans[0], trans[1], trans[2]);
        m_geometry.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        m_geometry.scale = glm::vec3(scale[0], scale[1], scale[2]);
        m_geometry.transform = utilityCore::buildTransformationMatrix(
            m_geometry.translation, m_geometry.rotation, m_geometry.scale);
        m_geometry.inverseTransform = glm::inverse(m_geometry.transform);
        m_geometry.invTranspose = glm::inverseTranspose(m_geometry.transform);

        // load obj mesh file
        if (m_geometry.type == OBJ) {
            // file path
            std::string file_path = "../scenes/";
            file_path += std::string("mesh/") + std::string(p["NAME"]) + std::string(".obj");
            tinyobj::ObjReader reader{};

            // parse obj file
            if (!reader.ParseFromFile(file_path)) {
                std::cerr << "Parse obj file " << reader.Error() << std::endl;
                std::abort();
            }

            // init data containers
            size_t index_offset = 0;
            const tinyobj::attrib_t& attribs = reader.GetAttrib();
            const tinyobj::shape_t& shape = *reader.GetShapes().begin();

            // init bounding volumn 
            if (bvh_datas.size() == 0) {
                bvh_datas.push_back(BVH_Data());
            }

            // iter over all the faces
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
                const auto c = shape.mesh.num_face_vertices[f];
                const size_t num_vertex = c;
                for (size_t i = 0; i < num_vertex; ++i) {
                    const tinyobj::index_t& v_data = shape.mesh.indices[i + index_offset];
                    Mesh_Data vertex;
                    // update vertex
                    const size_t n_v = 3 * v_data.vertex_index;
                    const size_t n_n = 3 * v_data.normal_index;
                    const size_t n_c = 2 * v_data.texcoord_index;
                    vertex.point = glm::vec3
                    (
                        attribs.vertices[n_v],
                        attribs.vertices[n_v + 1],
                        attribs.vertices[n_v + 2]
                    );
                    if (v_data.normal_index >= 0) {
                        vertex.normal = glm::vec3
                        (
                            attribs.normals[n_n],
                            attribs.normals[n_n + 1],
                            attribs.normals[n_n + 2]
                        );
                    }
                    if (v_data.texcoord_index >= 0) {
                        vertex.coordinate = glm::vec2
                        (
                            attribs.texcoords[n_c],
                            attribs.texcoords[n_c + 1]
                        );
                    }
                    vertex.material = m_geometry.materialid;
                    m_data.push_back(vertex);
                }
                index_offset += num_vertex;
            }

            // compute vertex tangent
            for (size_t i = 0; i < m_data.size(); i += 3) {
                const glm::vec3 v0 = m_data[i + 1].point - m_data[i].point;
                const glm::vec3 v1 = m_data[i + 2].point - m_data[i].point;
                const glm::vec2 v2 = m_data[i + 1].coordinate - m_data[i].coordinate;
                const glm::vec2 v3 = m_data[i + 2].coordinate - m_data[i].coordinate;
                const float d = v2.x * v3.y - v2.y * v3.x;
                if (d != 0.0) {
                    const glm::vec3 tangent = (v0 * v3.y - v2.y * v1) / d;
                    for (size_t v_i = i; v_i < i + 3; ++v_i) {
                        m_data[v_i].tangent = tangent;
                        const float a = glm::dot(m_data[v_i].tangent, m_data[v_i].normal);
                        //m_data[v_i].tangent -= m_data[v_i].normal * glm::dot(m_data[v_i].tangent, m_data[v_i].normal);
                        m_data[v_i].tangent = m_data[v_i].tangent - m_data[v_i].normal * a;
                        m_data[v_i].tangent = glm::normalize(m_data[v_i].tangent);
                    }
                }
            }
            // debug
            std::cout << "Successfully read " << m_data.size() << " vertices from " << file_path << std::endl;
            bvh_datas.begin()->bvh_mesh_data.insert(bvh_datas.begin()->bvh_mesh_data.end(),
                m_data.begin(), m_data.end());
        }
        geoms.push_back(m_geometry);
    }
    // BVH
    if (bvh_datas.size() > 0) {
        std::vector<int> indices;
        indices.push_back(0);
        while (indices.size() != 0) {
            int current_index = indices.back();
            indices.pop_back();
            BVH_Data& bvh_data = bvh_datas[current_index];
            std::vector<Mesh_Data>& vertices = bvh_data.bvh_mesh_data;
            glm::vec3 min_corner(std::numeric_limits<float>::max());
            glm::vec3 max_corner(std::numeric_limits<float>::lowest());
            float avg_radius = 0.0;
            for (size_t i = 0; i < vertices.size(); i += 3) {
                glm::vec3 centroid = compute_centroid(vertices, i);
                float radius = compute_bounding_radius(vertices, i, centroid);
                update_bounding_box(min_corner, max_corner, centroid, radius);
                avg_radius += radius;
            }
            avg_radius /= (vertices.size() / 3.0);
            bvh_data.center = 0.5f * (min_corner + max_corner);
            bvh_data.radius = 0.5f * glm::distance(min_corner, max_corner);
            glm::vec3 dimensions = max_corner - min_corner;
            float max_length = glm::max(dimensions.x, glm::max(dimensions.y, dimensions.z)) - glm::epsilon<float>();
            int separation_axis = determine_separation_axis(dimensions, max_length);
            std::vector<Mesh_Data> separated_vertices[2] = { std::vector<Mesh_Data>(), std::vector<Mesh_Data>() };
            for (size_t i = 0; i < vertices.size(); i += 3) {
                glm::vec3 centroid = compute_centroid(vertices, i);
                float radius = compute_bounding_radius(vertices, i, centroid);
                int target_index = determine_target_index(centroid, bvh_data.center, radius, avg_radius, separation_axis);
                for (int j = 0; j < 3; ++j) {
                    separated_vertices[target_index].push_back(vertices[i + j]);
                }
            }
            if (separated_vertices[0].empty() || separated_vertices[1].empty()) continue;
            for (size_t i = 0; i < 2; ++i) {
                bvh_data.child_indices[i] = bvh_datas.size() + i;
            }
            bvh_data.bvh_mesh_data.clear();
            for (size_t i = 0; i < 2; ++i) {
                bvh_datas.push_back(BVH_Data());
                bvh_datas.back().bvh_mesh_data = separated_vertices[i];
                indices.push_back(bvh_datas.size() - 1);
            }
        }
        for (auto& data : bvh_datas) {
            bvh_main_datas.emplace_back();
            BVH_Main_Data& main_data = bvh_main_datas.back();
            main_data.center = data.center;
            main_data.radius = data.radius;
            main_data.child_indices[0] = data.child_indices[0];
            main_data.child_indices[1] = data.child_indices[1];
            if (data.bvh_mesh_data.size() != 0) {
                main_data.index = m_data.size();
                main_data.count = data.bvh_mesh_data.size();
                m_data.insert(m_data.end(), data.bvh_mesh_data.begin(), data.bvh_mesh_data.end());
            }
        }
        bvh_datas.clear();
        //debug
        std::cout << "Generated " << this->bvh_main_datas.size() << " bounding spheres" << std::endl;
        std::cout << "Generated " << this->m_data.size() << " Mesh Data" << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
}

glm::vec3 Scene::compute_centroid(const std::vector<Mesh_Data>& data, size_t index) {
    return (data[index].point + data[index + 1].point + data[index + 2].point) / 3.0f;
}

float Scene::compute_bounding_radius(const std::vector<Mesh_Data>& data, std::size_t index, const glm::vec3& centroid) {
    float dist0 = glm::distance(data[index].point, centroid);
    float dist1 = glm::distance(data[index + 1].point, centroid);
    float dist2 = glm::distance(data[index + 2].point, centroid);
    return glm::max(dist0, glm::max(dist1, dist2));
}

void Scene::update_bounding_box(glm::vec3& min_corner, glm::vec3& max_corner, const glm::vec3& centroid, float radius) {
    min_corner.x = glm::min(min_corner.x, centroid.x - radius);
    min_corner.y = glm::min(min_corner.y, centroid.y - radius);
    min_corner.z = glm::min(min_corner.z, centroid.z - radius);

    max_corner.x = glm::max(max_corner.x, centroid.x + radius);
    max_corner.y = glm::max(max_corner.y, centroid.y + radius);
    max_corner.z = glm::max(max_corner.z, centroid.z + radius);
}

int Scene::determine_separation_axis(const glm::vec3& dimensions, float max_length) {
    if (dimensions.x >= max_length) return 0;
    if (dimensions.y >= max_length) return 1;
    return 2;
}

int Scene::determine_target_index(const glm::vec3& centroid, const glm::vec3& center, float radius, float avg_radius, int axis) {
    switch (axis) {
        case 0: return centroid.x + radius - avg_radius < center.x ? 0 : 1;
        case 1: return centroid.y + radius - avg_radius < center.y ? 0 : 1;
        default: return centroid.z + radius - avg_radius < center.z ? 0 : 1;
    }
}