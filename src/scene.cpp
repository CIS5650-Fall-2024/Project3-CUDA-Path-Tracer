#include <iostream>
#include <cstring>
#include <chrono>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
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
    Timer timer;
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Parse materials
    timer.start();

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        MaterialType type;
        glm::vec3 albeto;
        glm::vec3 specular;
        glm::vec3 transmittance;
        glm::vec3 emission;
        float kd;
        float ks;
        float shininess;
        float ior;
        bool delta = true;
        if (p["TYPE"] == "Diffuse")
        {
            type = DIFFUSE;
            const auto& col = p["ALBETO"];
            albeto = glm::vec3(col[0], col[1], col[2]);
            delta = false;
        }
        else if (p["TYPE"] == "Emitting")
        {
            type = EMISSION;
            const auto& col = p["EMISSION"];
            emission = glm::vec3(col[0], col[1], col[2]) * (float)p["EMITTANCE"];
            delta = false;
        }
        else if (p["TYPE"] == "Specular")
        {
            type = MIRROR;
            const auto& col = p["SPECULAR"];
            specular = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Glossy") 
        {
            type = GLOSSY;
            const auto& al = p["ALBETO"];
            albeto = glm::vec3(al[0], al[1], al[2]);
            const auto& sp = p["SPECULAR"];
            specular = glm::vec3(sp[0], sp[1], sp[2]);
            kd = (float)p["KD"];
            ks = (float)p["KS"];
            shininess = (float)p["SHININESS"];
        }
        else if (p["TYPE"] == "Glass")
        {
            type = GLASS;
            const auto& sp = p["SPECULAR"];
            specular = glm::vec3(sp[0], sp[1], sp[2]);
            const auto& tr = p["TRANSMITTANCE"];
            transmittance = glm::vec3(tr[0], tr[1], tr[2]);
            ior = (float)p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(type, albeto, specular, transmittance, emission, kd, ks, shininess, ior, delta);
    }

    timer.stop();
    printf("Finish processing %lu materials in %.4f seconds.\n", (int)materials.size(), timer.duration());

    // Parse lights
    timer.start();

    const auto& lightsData = data["Lights"];
    for (const auto& l : lightsData)
    {
        const auto& col = l["RGB"];
        LightSourceType type;
        glm::vec3 color = glm::vec3(col[0], col[1], col[2]) * (float)l["EMITTANCE"];
        glm::vec3 pos;
        glm::vec3 dir;
        glm::vec3 dimX;
        glm::vec3 dimY;
        float angle = 0.0f;
        bool delta = true;
        if (l["TYPE"] == "Area") {
            type = AREALIGHT;
            const auto& po = l["POS"];
            const auto& di = l["DIR"];
            const auto& dx = l["DIMX"];
            const auto& dy = l["DIMY"];
            pos = glm::vec3(po[0], po[1], po[2]);
            dir = glm::vec3(di[0], di[1], di[2]);
            dimX = glm::vec3(dx[0], dx[1], dx[2]);
            dimY = glm::vec3(dy[0], dy[1], dy[2]);
            delta = false;
        }
        else if (l["TYPE"] == "Directional") {
            type = DIRECTIONALLIGHT;
            const auto& di = l["DIR"];
            dir = glm::vec3(di[0], di[1], di[2]);
        }
        else if (l["TYPE"] == "Point") {
            type = POINTLIGHT;
            const auto& po = l["POS"];
            pos = glm::vec3(po[0], po[1], po[2]);
        }
        else if (l["TYPE"] == "Spot") {
            type = SPOTLIGHT;
            const auto& po = l["POS"];
            const auto& di = l["DIR"];
            pos = glm::vec3(po[0], po[1], po[2]);
            dir = glm::vec3(di[0], di[1], di[2]);
            angle = l["ANGLE"];
        }
        lights.emplace_back(type, color, pos, dir, dimX, dimY, angle, delta);
    }

    timer.stop();
    printf("Finish processing %lu lights in %.4f seconds.\n", (int)lights.size(), timer.duration());

    // Parse objects
    timer.start();

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.type = type == "mesh" ? MESH : type == "cube" ? CUBE : SPHERE;
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
        int curGeomId = geoms.size();

        if (newGeom.type == CUBE)
        {
            // Not really used since we have cube.obj, only for the box
            Primitive newPrim;
            newPrim.type = CUBEP;
            newPrim.geomId = curGeomId;

            // Compute vertices in world coordinate system
            glm::vec3 q1 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f));
            glm::vec3 q2 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, -0.5f, 0.5f, 1.f));
            glm::vec3 q3 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, 0.5f, -0.5f, 1.f));
            glm::vec3 q4 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, 0.5f, 0.5f, 1.f));
            glm::vec3 q5 = glm::vec3(newGeom.transform * glm::vec4(0.5f, -0.5f, -0.5f, 1.f));
            glm::vec3 q6 = glm::vec3(newGeom.transform * glm::vec4(0.5f, -0.5f, 0.5f, 1.f));
            glm::vec3 q7 = glm::vec3(newGeom.transform * glm::vec4(0.5f, 0.5f, -0.5f, 1.f));
            glm::vec3 q8 = glm::vec3(newGeom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f));

            // Compute bounding box
            float minX = glm::min(glm::min(glm::min(q1.x, q2.x), glm::min(q3.x, q4.x)), glm::min(glm::min(q5.x, q6.x), glm::min(q7.x, q8.x)));
            float minY = glm::min(glm::min(glm::min(q1.y, q2.y), glm::min(q3.y, q4.y)), glm::min(glm::min(q5.y, q6.y), glm::min(q7.y, q8.y)));
            float minZ = glm::min(glm::min(glm::min(q1.z, q2.z), glm::min(q3.z, q4.z)), glm::min(glm::min(q5.z, q6.z), glm::min(q7.z, q8.z)));
            float maxX = glm::max(glm::max(glm::max(q1.x, q2.x), glm::max(q3.x, q4.x)), glm::max(glm::max(q5.x, q6.x), glm::max(q7.x, q8.x)));
            float maxY = glm::max(glm::max(glm::max(q1.y, q2.y), glm::max(q3.y, q4.y)), glm::max(glm::max(q5.y, q6.y), glm::max(q7.y, q8.y)));
            float maxZ = glm::max(glm::max(glm::max(q1.z, q2.z), glm::max(q3.z, q4.z)), glm::max(glm::max(q5.z, q6.z), glm::max(q7.z, q8.z)));

            // Store min at p2 and max at p3 since they are not used
            newPrim.p2 = glm::vec3(minX, minY, minZ);
            newPrim.p3 = glm::vec3(maxX, maxY, maxZ);

            prims.push_back(newPrim);
        }
        else if (newGeom.type == SPHERE)
        {
            Primitive newPrim;
            newPrim.type = SPHEREP;
            newPrim.geomId = curGeomId;

            // Compute edge vertices in world coordinate system
            glm::vec3 q1 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f));
            glm::vec3 q2 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, -0.5f, 0.5f, 1.f));
            glm::vec3 q3 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, 0.5f, -0.5f, 1.f));
            glm::vec3 q4 = glm::vec3(newGeom.transform * glm::vec4(-0.5f, 0.5f, 0.5f, 1.f));
            glm::vec3 q5 = glm::vec3(newGeom.transform * glm::vec4(0.5f, -0.5f, -0.5f, 1.f));
            glm::vec3 q6 = glm::vec3(newGeom.transform * glm::vec4(0.5f, -0.5f, 0.5f, 1.f));
            glm::vec3 q7 = glm::vec3(newGeom.transform * glm::vec4(0.5f, 0.5f, -0.5f, 1.f));
            glm::vec3 q8 = glm::vec3(newGeom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f));

            // Compute bounding box
            float minX = glm::min(glm::min(glm::min(q1.x, q2.x), glm::min(q3.x, q4.x)), glm::min(glm::min(q5.x, q6.x), glm::min(q7.x, q8.x)));
            float minY = glm::min(glm::min(glm::min(q1.y, q2.y), glm::min(q3.y, q4.y)), glm::min(glm::min(q5.y, q6.y), glm::min(q7.y, q8.y)));
            float minZ = glm::min(glm::min(glm::min(q1.z, q2.z), glm::min(q3.z, q4.z)), glm::min(glm::min(q5.z, q6.z), glm::min(q7.z, q8.z)));
            float maxX = glm::max(glm::max(glm::max(q1.x, q2.x), glm::max(q3.x, q4.x)), glm::max(glm::max(q5.x, q6.x), glm::max(q7.x, q8.x)));
            float maxY = glm::max(glm::max(glm::max(q1.y, q2.y), glm::max(q3.y, q4.y)), glm::max(glm::max(q5.y, q6.y), glm::max(q7.y, q8.y)));
            float maxZ = glm::max(glm::max(glm::max(q1.z, q2.z), glm::max(q3.z, q4.z)), glm::max(glm::max(q5.z, q6.z), glm::max(q7.z, q8.z)));

            // Store min at p2 and max at p3 since they are not used
            newPrim.p2 = glm::vec3(minX, minY, minZ);
            newPrim.p3 = glm::vec3(maxX, maxY, maxZ);

            prims.push_back(newPrim);
        }
        else if (newGeom.type == MESH) 
        {
            std::string inputfile = p["PATH"];
            tinyobj::ObjReaderConfig reader_config;
            tinyobj::ObjReader reader;

            std::cout << "Loading " << inputfile << std::endl;

            if (!reader.ParseFromFile(inputfile, reader_config)) {
                if (!reader.Error().empty()) {
                    std::cerr << "TinyObjReader: " << reader.Error();
                }
                exit(1);
            }

            if (!reader.Warning().empty()) {
                std::cout << "TinyObjReader: " << reader.Warning();
            }

            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();

            int faceCount = 0;
            int vertexCount = 0;
            int normalCount = 0;

            // Loop over shapes
            for (size_t s = 0; s < shapes.size(); s++)
            {
                size_t index_offset = 0;
                
                faceCount += shapes[s].mesh.num_face_vertices.size();

                // Loop over face
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
                {
                    size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                    vertexCount += fv;
                    
                    if (fv != 3) 
                    {
                        std::cout << "Not an triangle!" << std::endl;
                    }

                    Primitive newPrim;
                    newPrim.type = TRIANGLE;
                    newPrim.geomId = curGeomId;
                    
                    std::vector<glm::vec3> tmpPos {};
                    std::vector<glm::vec3> tmpNor {};

                    // Loop over vertices in the face
                    for (size_t v = 0; v < fv; v++) 
                    {
                        // Access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                        tmpPos.push_back(glm::vec3(vx, vy, vz));

                        // Check if `normal_index` is zero or positive. negative = no normal data
                        if (idx.normal_index >= 0) {
                            normalCount++;

                            tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                            tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                            tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                            tmpNor.push_back(glm::vec3(nx, ny, nz));
                        }
                    }

                    newPrim.p1 = glm::vec3(newGeom.transform * glm::vec4(tmpPos[0], 1.f));
                    newPrim.p2 = glm::vec3(newGeom.transform * glm::vec4(tmpPos[1], 1.f));
                    newPrim.p3 = glm::vec3(newGeom.transform * glm::vec4(tmpPos[2], 1.f));

                    newPrim.n1 = glm::vec3(newGeom.invTranspose * glm::vec4(tmpNor[0], 0.f));
                    newPrim.n2 = glm::vec3(newGeom.invTranspose * glm::vec4(tmpNor[1], 0.f));
                    newPrim.n3 = glm::vec3(newGeom.invTranspose * glm::vec4(tmpNor[2], 0.f));
                    prims.push_back(newPrim);

                    index_offset += fv;
                }
            }

            printf("Total %lu faces, %lu vertices, %lu normals captioned.\n", faceCount, vertexCount, normalCount);
        }

        geoms.push_back(newGeom);
    }

    timer.stop();
    printf("Finish processing %lu objects, %lu primitives in %.4f seconds.\n", (int)geoms.size(), (int)prims.size(), timer.duration());

    // Parse camera
    timer.start();

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
    camera.aperture = cameraData["APERTURE"];
    camera.focal = cameraData["FOCAL"];
    camera.sample = cameraData["SAMPLE"];

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

    timer.stop();
    printf("Finishing processing camera in %.4f seconds.\n", timer.duration());

#if BVH
    // Constructing BVH
    timer.start();

    std::vector<int> initialIndices{};
    for (int i = 0; i < prims.size(); i++)
    {
        initialIndices.push_back(i);
    }
    constructBVH(prims, initialIndices, bvh);
    buidlStackless(bvh);

    timer.stop();
    printf("Finish constructing BVH for %lu primitives in %.4f seconds.\n", (int)prims.size(), timer.duration());

    // test constructed BVH
    //printf(" \n");
    //printf("============\n");
    //printf("scene bvh display\n");

    //for (int i = 0; i < bvh.size(); i++) {
    //    BVHNode& b = bvh[i];
    //    printf("Current bvh index %d, left child index %d, right child index %d, escape index %d\n", i, b.leftNodeIndex, b.rightNodeIndex, b.escapseIndex);
    //    printf("BBox minC %.4f %.4f %.4f\n", b.bb.minC[0], b.bb.minC[1], b.bb.minC[2]);
    //    printf("BBox maxC %.4f %.4f %.4f\n", b.bb.maxC[0], b.bb.maxC[1], b.bb.maxC[2]);
    //    printf("BVHNode prims indices %d %d %d %d, isleaf %d\n", b.p1I, b.p2I, b.p3I, b.p4I, b.p1I >= 0);
    //}
#endif
}
