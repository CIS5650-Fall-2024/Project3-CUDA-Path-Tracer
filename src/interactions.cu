#include "interactions.h"

// Index of refraction for air
#define EXT_IOR 1.000277f

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 woW,
    glm::vec3 normal, // Here normal is in world space
    glm::vec3 &wiW,
    float &pdf,
    glm::vec3 &c,
    float &eta,
    const Material &m,
    const TextureValues& texVals,
    thrust::default_random_engine &rng)
{
    glm::mat3 worldToLocal = WorldToLocal(normal);
    glm::mat3 localToWorld = LocalToWorld(normal);
    glm::vec3 woL = worldToLocal * woW; 
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 sample2D(u01(rng), u01(rng));

    glm::vec3 actualAlbedo = m.color;
    glm::vec3 actualNormal = normal;

    if (texVals.albedo != glm::vec4(INFINITY)) {
        actualAlbedo = glm::vec3(texVals.albedo);
    }
    
    if (texVals.normal != glm::vec4(INFINITY)) {
        // texNormal already have *2 - 1 applied
        glm::vec3 texNormal = glm::normalize(glm::vec3(texVals.normal));
        actualNormal = glm::normalize(localToWorld * texNormal);
    }

    if (texVals.bump != glm::vec4(INFINITY)) {
        float du = texVals.bump.x;
        float dv = texVals.bump.y;
        glm::vec3 tangent = localToWorld[0];
        glm::vec3 bitangent = localToWorld[1];

        // Perturb the normal using the bump map derivatives du and dv
        actualNormal = glm::normalize(actualNormal + du * tangent + dv * bitangent);
    }

    if (m.type == MatType::DIFFUSE) {
        c = sampleDiffuse(actualAlbedo, actualNormal, sample2D, wiW, eta);
        glm::vec3 wiL = worldToLocal * wiW;
        pdf = pdfDiffuse(woL, wiL);
    }
    else if (m.type == MatType::MIRROR) {
        c = sampleMirror(actualNormal, worldToLocal, woW, wiW, m.specularColor, eta);
        pdf = pdfMirror();
    }
    else if (m.type == MatType::DIELECTRIC) {
        c = sampleDielectric(actualNormal, worldToLocal, localToWorld, woW, sample2D.x, EXT_IOR, m.indexOfRefraction, m.specularColor, wiW, eta);
        pdf = pdfDielectric();
    }
    else if (m.type == MatType::MICROFACET) {
        float tmp = (EXT_IOR - m.indexOfRefraction) / (EXT_IOR + m.indexOfRefraction);
        // Specular component based on Fresnel term
        float m_ks = tmp * tmp; // This is F0
        // Diffuse component, ensuring energy conservation
        glm::vec3 m_kd = (1.0f - m_ks) * actualAlbedo;

        // Given that sampleMicrofacet also calculates the pdf, we can just pass it in as a parameter
        c = sampleMicrofacet(actualNormal, worldToLocal, localToWorld, woW, m_kd, m_ks, m.roughness, EXT_IOR, m.indexOfRefraction, sample2D, wiW, pdf, eta);
    }
}