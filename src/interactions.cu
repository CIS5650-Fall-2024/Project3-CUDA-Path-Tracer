#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    glm::vec3 &out_ray)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    out_ray = calculateRandomDirectionInHemisphere(normal, rng);
}

#define sat(x)	glm::clamp(x, 0.f, 1.f)
#define S(a, b, c)	glm::smoothstep(a, b, c)
#define S01(a)	S(0.f, 1.f, a)

__device__ float sum2(glm::vec2 v) { return dot(v, glm::vec2(1.f)); }

__device__ float h31(glm::vec3 p3) {
	p3 = fract(p3 * .1031f);
	glm::vec3 r1{ p3.y, p3.z, p3.x };
	p3 += dot(p3, r1 + 333.3456f);
	glm::vec2 r2{ p3.x, p3.y };
	return glm::fract(sum2(r2) * p3.z);
}

__device__ float h21(glm::vec2 p) { return h31(glm::vec3(p.x, p.y, p.x)); }

__device__ float n31(glm::vec3 p) {
	const glm::vec3 s = glm::vec3(7, 157, 113);

	// Shane - https://www.shadertoy.com/view/lstGRB
	glm::vec3 ip = floor(p);
	p = fract(p);
	p = p * p * (3.f - 2.f * p);
	glm::vec4 h = glm::vec4(0.f, glm::vec2(s.y, s.z), sum2(glm::vec2(s.y, s.z))) + dot(ip, s);
	h = glm::mix(glm::fract(sin(h) * 43758.545f), glm::fract(sin(h + s.x) * 43758.545f), p.x);
	h.x = glm::mix(h.x, h.y, p.y);
	h.y = glm::mix(h.z, h.w, p.y);
	return glm::mix(h.x, h.y, p.z);
}

// roughness: (0.0, 1.0], default: 0.5
// Returns unsigned noise [0.0, 1.0]
__device__ float fbm(glm::vec3 p, int octaves, float roughness) {
	float sum = 0.,
		amp = 1.,
		tot = 0.;
	roughness = sat(roughness);
	for (int i = 0; i < octaves; i++) {
		sum += amp * n31(p);
		tot += amp;
		amp *= roughness;
		p *= 2.;
	}
	return sum / tot;
}

__device__ glm::vec3 randomPos(float seed) {
	glm::vec4 s = glm::vec4(seed, 0, 1, 2);
	return glm::vec3(h21(glm::vec2(s.x, s.y)), h21(glm::vec2(s.x, s.z)), h21(glm::vec2(s.x, s.w))) * 100.f + 100.f;
}

// Returns unsigned noise [0.0, 1.0]
__device__ float fbmDistorted(glm::vec3 p) {
	p += (glm::vec3(n31(p + randomPos(0.f)), n31(p + randomPos(1.f)), n31(p + randomPos(2.f))) * 2.f - 1.f) * 1.12f;
	return fbm(p, 8, .5);
}

// vec3: detail(/octaves), dimension(/inverse contrast), lacunarity
// Returns signed noise.
__device__ float musgraveFbm(glm::vec3 p, float octaves, float dimension, float lacunarity) {
	float sum = 0.,
		amp = 1.,
		m = pow(lacunarity, -dimension);
	for (float i = 0.; i < octaves; i++) {
		float n = n31(p) * 2. - 1.;
		sum += n * amp;
		amp *= m;
		p *= lacunarity;
	}
	return sum;
}

// Wave noise along X axis.
__device__ glm::vec3 waveFbmX(glm::vec3 p) {
	float n = p.x * 20.;
	n += .4 * fbm(p * 3.f, 3, 3.);
	return glm::vec3(sin(n) * .5f + .5f, glm::vec2(p.y, p.z));
}

__device__ float remap01(float f, float in1, float in2) { return sat((f - in1) / (in2 - in1)); }

// Noise function for wood
glm::vec3 matWood(glm::vec3 p) {
	float n1 = fbmDistorted(p * glm::vec3(7.8, 1.17, 1.17));
	n1 = glm::mix(n1, 1.f, .2f);
	float n2 = glm::mix(musgraveFbm(glm::vec3(n1 * 4.6f), 8.f, 0.f, 2.5f), n1, .85f),
		dirt = 1. - musgraveFbm(waveFbmX(p * glm::vec3(.01, .15, .15)), 15., .26, 2.4) * .4;
	float grain = 1. - S(.2f, 1.f, musgraveFbm(p * glm::vec3(500, 6, 1), 2., 2., 2.5)) * .2;
	n2 *= dirt * grain;

	// The three vec3 values are the RGB wood colors - Tweak to suit.
	return glm::mix(glm::mix(glm::vec3(.03, .012, .003), glm::vec3(.25, .11, .04), remap01(n2, .19, .56)), glm::vec3(.52, .32, .19), remap01(n2, .56, 1.));
}
