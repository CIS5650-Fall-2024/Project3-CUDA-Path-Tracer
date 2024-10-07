#pragma once
#include "utilities.h"

__inline__ __device__ float pow5(float x) {
	return x * x * x * x * x;
}

__inline__ __device__ float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
	return 1 / (NdotV + sqrt(Square(VdotX * ax) + Square(VdotY * ay) + Square(NdotV)));
}

__inline__ __device__ glm::vec3 SchlickFresnel(const glm::vec3& F0, float cosTheta)
{
	return F0 + (glm::vec3(1.0f) - F0) * pow5(1.0f - cosTheta);
}

__inline__ __device__ // Cosine-weighted hemisphere sampling implementation
glm::vec3 cosineSampleHemisphere(thrust::default_random_engine& rng) {
	glm::vec3 normal = glm::vec3(0, 0, 1);
	thrust::uniform_real_distribution<float> u01(0, 1);

	// The random generated direction is cosine weighted by sqrt the random number
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

	// the final direction is a combination of a linear combination of the two perpendicular directions and the normal
	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

// diffuse
__inline__ __device__ float pdf_baseDiffuse(const glm::vec3& wi)
{
	return AbsCosTheta(wi) / PI;
}

__inline__ __device__ glm::vec3 Sample_baseDiffuse(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi)
{
	glm::vec3 F0 = glm::vec3(0.5) + 2 * m.roughness * Square(glm::dot(wh, wi));
	return m.color / PI * SchlickFresnel(F0, AbsCosTheta(wi)) * SchlickFresnel(F0, AbsCosTheta(wo)) * AbsCosTheta(wi);
}


// Lommel-Seeliger subsurface scattering approximation  
__inline__ __device__ float Base_subsurface(float roughness, const glm::vec3& wi, const glm::vec3& wh)
{
	return roughness * Square(AbsDot(wi, wh));
}

__inline__ __device__ glm::vec3 Sample_subsurface(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi)
{
	glm::vec3 F0(Base_subsurface(m.roughness, wi, wh));
	return 1.65f * m.color / PI 
		* ((SchlickFresnel(F0, AbsCosTheta(wi))) * (SchlickFresnel(F0, AbsCosTheta(wo))) * (1.f / (AbsCosTheta(wi) + AbsCosTheta(wo)) - 0.5f) + 0.5f) 
		* AbsCosTheta(wi);
}

// Metal


// calculate microfacet normal
__inline__ __device__ glm::vec3 sampleGGXVNDF(glm::vec3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
	// Section 3.2: transforming the view direction to the hemisphere configuration
	glm::vec3 Vh = normalize(glm::vec3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	glm::vec3 T1 = lensq > 0 ? glm::vec3(-Vh.y, Vh.x, 0) * 1.0f/ sqrt(lensq) : glm::vec3(1, 0, 0);
	glm::vec3 T2 = cross(Vh, T1);
	// Section 4.2: parameterization of the projected area
	float r = sqrt(U1);
	float phi = 2.0 * PI * U2;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
	// Section 4.3: reprojection onto hemisphere
	glm::vec3 Nh = t1 * T1 + t2 * T2 + sqrt(glm::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
	// Section 3.4: transforming the normal back to the ellipsoid configuration
	glm::vec3 Ne = normalize(glm::vec3(alpha_x * Nh.x, alpha_y * Nh.y, glm::max(0.0f, Nh.z)));
	return Ne;
}

__inline__ __device__ float GGX_Distribution(const glm::vec3& wh, float ax, float ay)
{
	
	return 1.0f / (PI * ax * ay * Square(Square(wh.x / ax) + Square(wh.y / ay) + Square(wh.z)));
}

__inline__ __device__ float GGX_Smith(float ax, float ay, const glm::vec3& wl)
{
	float a2 = (sqrt((1.0f + (Square(wl.x * ax) + Square(wl.y * ay)) / Square(wl.z)) + 1.0f) - 1.0f) / 2.0f;
	return 1.0f / (1.0f + a2);
}

__inline__ __device__ float GGX_Geometry(const glm::vec3& wo, const glm::vec3& wi, float ax, float ay)
{
	return smithG_GGX_aniso(wi.z, wi.x, wi.y, ax, ay) * smithG_GGX_aniso(wo.z, wo.x, wo.y, ax, ay);
}

// microfacet reflection pdf
__inline__ __device__ float pdf_microfacet(float D, float G, const glm::vec3 & wi, const glm::vec3& h)
{
	return D * G / (4 * AbsDot(wi, h));
}

__inline__ __device__ void axay(float roughness, float anisotropic, float& ax, float& ay)
{
	float roughnessSquared = roughness * roughness;

	// Anisotropy adjustment
	ax = roughnessSquared * (1.0f + anisotropic);
	ay = roughnessSquared * (1.0f - anisotropic);
}




// integrate the BRDF
__inline__ __device__ glm::vec3 Sample_disneyBSDF(const Material& m, const glm::vec3& woW, const glm::vec2& xi, glm::vec3& wi, const glm::mat3& ltw, const glm::mat3& wtl, float& pdf,
	thrust::default_random_engine& rng)
{
	glm::vec3 n = glm::vec3(0, 0, 1);
	glm::vec3 wo = normalize(wtl * woW);

	float ax, ay; ax = ay = 0.0f;
	axay(1, m.anisotropic, ax, ay);
	// diffuse
	glm::vec3 wh_d = normalize(sampleGGXVNDF(wo, ax, ay, xi.x, xi.y));
	//glm::vec3 wi_d = normalize(cosineSampleHemisphere(n, rng));
	//glm::vec3 wi_d = wtl * wi;
	glm::vec3 wi_d = reflect(-wo, wh_d);
	glm::vec3 fd = Sample_baseDiffuse(m, wo, normalize(wi_d + wo), wi_d);
	float fd_pdf = pdf_baseDiffuse(wi_d);
	wi = normalize(ltw * wi_d);

	// subsurface scattering
	glm::vec3 fss = Sample_subsurface(m, wo, normalize(wi + wo), wi);
	fd = Lerp(fd, fss, m.subsurface);

	pdf = fd_pdf;
	return wi;
}










// btdf

__inline__ __device__ bool Refract(const glm::vec3 wi, float eta, glm::vec3& wt) {
	// Compute cos theta using Snell's law
	float cosThetaI = glm::dot(wi, glm::vec3(0, 0, 1));
	float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
	float sin2ThetaT = eta * eta * sin2ThetaI;

	// Handle total internal reflection for transmission
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = sqrt(1 - sin2ThetaT);
	glm::vec3 normal = cosThetaI > 0 ? glm::vec3(0, 0, 1) : glm::vec3(0, 0, -1);
	wt = eta * -wi + (eta * cosThetaI - cosThetaT)* normal;
	return true;
}


__inline__ __device__ glm::vec3 Sample_f_specular_trans(const Material& m, const glm::vec3& wi, bool refract)
{
	return refract ? m.color / AbsCosTheta(wi) : glm::vec3(0.0f);
}


__inline__ __device__ glm::vec3 Sample_f_specular_refl(const Material& m, const glm::vec3 wi)
{
	return m.color / AbsCosTheta(wi);
}


__inline__ __device__ glm::vec3 FresnelDielectricEval(float ior, float cosThetaI) {
	// We will hard-code the indices of refraction to be
	// those of glass
	float etaI = 1.;
	float etaT = ior;
	cosThetaI = Clamp(cosThetaI, -1.f, 1.f);

	// potentially swap indices of refractions
	bool entering = cosThetaI > 0.f;
	if (!entering) {
		float t = etaI;
		etaI = etaT;
		etaT = t;
		cosThetaI = abs(cosThetaI);
	}

	// compute cosThetaT
	float sinThetaI = sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1)
		return glm::vec3(1.);

	float cosThetaT = sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));

	// compute reflectance
	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
		((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
		((etaI * cosThetaI) + (etaT * cosThetaT));
	return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2);
}


__inline__ __device__ glm::vec3 Sample_btdf(const Material& m, const glm::vec3& wo, const glm::vec3& wi, float& pdf, const bool refract, const bool reflect)
{

	if (reflect)
	{
		pdf = 1.0f;
		//return glm::vec3(0, 0., 1);
		glm::vec3 R = Sample_f_specular_refl(m, wi);
		return 2.0f * FresnelDielectricEval(m.ior, glm::dot(wi, glm::vec3(0, 0, 1))) * R;
	}
	else if (refract)
	{
		pdf = 1.0f;
		glm::vec3 T = Sample_f_specular_trans(m, wi, refract);
		return 2.0f * (1.0f - FresnelDielectricEval(m.ior, glm::dot(wi, glm::vec3(0, 0, 1)))) * T;
	}
	return glm::vec3(0.0f);
}







// Disney BRDF
__inline__ __device__ float SchlickFresnel(float u)
{
	float m = Clamp(1.f - u, 0.f, 1.f);
	float m2 = m * m;
	return m2 * m2 * m; // pow(m,5)
}

__inline__ __device__ float GTR1(float NdotH, float a)
{
	if (a >= 1) return 1 / PI;
	float a2 = a * a;
	float t = 1 + (a2 - 1) * NdotH * NdotH;
	return (a2 - 1) / (PI * log(a2) * t);
}

__inline__ __device__ float GTR2(float NdotH, float a)
{
	float a2 = a * a;
	float t = 1 + (a2 - 1) * NdotH * NdotH;
	return a2 / (PI * t * t);
}

__inline__ __device__ float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
	return 1 / (PI * ax * ay * Square(Square(HdotX / ax) + Square(HdotY / ay) + NdotH * NdotH));
}

__inline__ __device__ float smithG_GGX(float NdotV, float alphaG)
{
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1 / (NdotV + sqrt(a + b - a * b));
}



__inline__ __device__ glm::vec3 mon2lin(const glm::vec3& x)
{
	return glm::vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}

__inline__ __device__ glm::vec3 BRDF(const Material& m, const glm::vec3& L, const glm::vec3& V, const glm::vec3& N, const glm::vec3& X, const glm::vec3& Y, float& pdf)
{
	float NdotL = dot(N, L);
	float NdotV = dot(N, V);
	if (NdotL < 0 || NdotV < 0) return glm::vec3(0);

	glm::vec3 H = normalize(L + V);
	float NdotH = dot(N, H);
	float LdotH = dot(L, H);

	glm::vec3 Cdlin = mon2lin(m.color);
	float Cdlum = .3 * Cdlin[0] + .6 * Cdlin[1] + .1 * Cdlin[2]; // luminance approx.

	glm::vec3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : glm::vec3(1); // normalize lum. to isolate hue+sat
	glm::vec3 Cspec0 = Lerp(m.specular * .08f * Lerp(glm::vec3(1), Ctint, m.specularTint), Cdlin, m.metallic);
	glm::vec3 Csheen = Lerp(glm::vec3(1), Ctint, m.sheenTint);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
	float Fd90 = 0.5 + 2 * LdotH * LdotH * m.roughness;
	float Fd = Lerp(1.0f, Fd90, FL) * Lerp(1.0f, Fd90, FV);

	// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LdotH * LdotH * m.roughness;
	float Fss = Lerp(1.0f, Fss90, FL) * Lerp(1.0f, Fss90, FV);
	float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

	// specular
	float aspect = sqrt(1 - m.anisotropic * .9);
	float ax = glm::max(.001f, Square(m.roughness) / aspect);
	float ay = glm::max(.001f, Square(m.roughness) * aspect);
	float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
	float FH = SchlickFresnel(LdotH);
	glm::vec3 Fs = Lerp(Cspec0, glm::vec3(1), FH);
	float Gs;
	Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
	Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

	// sheen
	glm::vec3 Fsheen = FH * m.sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = GTR1(NdotH, Lerp(.1, .001, m.clearcoatGloss));
	float Fr = Lerp(.04, 1.0, FH);
	float Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25);

	float pdf_diffuse = AbsCosTheta(L) / PI;
	float pdf_specular = Ds * NdotH / (4 * glm::dot(L, H));
	pdf = Lerp(pdf_diffuse, pdf_specular, Fd);
	//return glm::vec3(pdf);
	return ((1 / PI) * Lerp(Fd, ss, m.subsurface) * Cdlin + Fsheen)
		* (1 - m.metallic)
		+ Gs * Fs * Ds + .25f * m.clearcoat * Gr * Fr * Dr;
}

__inline__ __device__ glm::vec3 Sample_microfacet(const Material& m, const glm::vec3& wo, const glm::vec3& wh, const glm::vec3& wi, float& pdf)
{
	float ax, ay; ax = ay = 0.0f;
	glm::vec3 F0(m.metallic);

	axay(m.roughness, m.anisotropic, ax, ay);
	glm::vec3 fd = m.color / PI * SchlickFresnel(F0, AbsCosTheta(wi)) * SchlickFresnel(F0, AbsCosTheta(wo));
	float D = 0;

	glm::vec3 F = SchlickFresnel(glm::vec3(F0), AbsDot(wi, wh));
	float G = GGX_Geometry(wo, wi, ax, ay);

	float cosThetaWH = AbsCosTheta(wh);
	float pdf_diffuse = AbsCosTheta(wi) / PI;
	float pdf_specular = D * cosThetaWH / (4 * glm::dot(wo, wh));
	pdf = Lerp(pdf_diffuse, pdf_specular, F.r);
	return fd;
	return fd + D * F * G / (4 * AbsCosTheta(wi) * AbsCosTheta(wo));
}

__inline__ __device__ void BSDF_setUp(const Material& m, glm::vec3& wi, const glm::vec3& wo, thrust::default_random_engine& rng, bool& isRefract, bool& isReflect)
{
	isRefract = false;
	isReflect = false;
	if (m.type == MaterialType::DIFFUSE)
	{
		wi = glm::normalize(cosineSampleHemisphere(rng));
	}
	else if (m.type == MaterialType::TRANSMIT)
	{
		float u01 = thrust::uniform_real_distribution<float>(0, 1)(rng);
		if (u01 < 0.5)
		{
			wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
			isReflect = true;
		}
		else
		{
			float eta = glm::dot(wo, glm::vec3(0, 0, 1)) < 0 ? m.ior / 1.0f : 1.0f / m.ior;
			isRefract = Refract(wo, eta, wi);
		}
	}
	else
	{
		glm::vec2 xi = glm::vec2(thrust::uniform_real_distribution<float>(0, 1)(rng), thrust::uniform_real_distribution<float>(0, 1)(rng));
		float ax, ay; ax = ay = 0.0f;
		axay(m.roughness, m.anisotropic, ax, ay);
		glm::vec3 wh_d = normalize(sampleGGXVNDF(wo, ax, ay, xi.x, xi.y));
		wi = reflect(-wo, wh_d);
	}
	
	
}

__inline__ __device__ glm::vec3 Evaluate_disneyBSDF(const Material& m, const glm::vec3& wi, const glm::vec3& wo, float& pdf, bool refract, bool reflect)
{
	if (m.type == MaterialType::DIFFUSE)
	{
		pdf = AbsCosTheta(wi) / PI;
		return m.color / PI;

	}
	else if (m.type == MaterialType::TRANSMIT)
	{

		return Sample_btdf(m, wo, wi, pdf, refract, reflect);
	}
	else
	{
		return BRDF(m, wi, wo, glm::vec3(0, 0, 1), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), pdf);
	}
	pdf = 0;
	return glm::vec3(0.);
}
