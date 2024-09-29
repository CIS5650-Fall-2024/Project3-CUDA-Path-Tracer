#include "distribution1D.h"

Distribution1D& Distribution1D::operator=(Distribution1D&& other) noexcept
{
	func = std::move(other.func);
	cdf = std::move(other.cdf);
	funcInt = other.funcInt;
	return *this;
}

Distribution1D::Distribution1D(std::vector<float> vals) : func(vals), cdf(vals.size() + 1)
{
	int n = vals.size();
	cdf[0] = 0;
	for (int i = 1; i < n + 1; ++i)
	{
		cdf[i] = cdf[i - 1] + func[i - 1] / n;
	}
	funcInt = cdf[n];
	if (funcInt == 0) {
		for (int i = 1; i < n + 1; ++i)
			cdf[i] = static_cast<float>(i) / static_cast<float>(n);
	}
	else {
		for (int i = 1; i < n + 1; ++i)
			cdf[i] /= funcInt;
	}
}

Distribution1D::Distribution1D(const float* vals, int n) : func(vals, vals + n), cdf(n + 1)
{
	cdf[0] = 0;
	for (int i = 1; i < n + 1; ++i)
	{
		cdf[i] = cdf[i - 1] + func[i - 1] / n;
	}
	funcInt = cdf[n];
	if (funcInt == 0) {
		for (int i = 1; i < n + 1; ++i)
			cdf[i] = static_cast<float>(i) / static_cast<float>(n);
	}
	else {
		for (int i = 1; i < n + 1; ++i)
			cdf[i] /= funcInt;
	}
}

Distribution1D::Distribution1D(const Distribution1D& other)
{
	func = other.func;
	cdf = other.cdf;
	funcInt = other.funcInt;
}

int Distribution1D::Count() const { return func.size(); }

float Distribution1D::sampleContinuous(float u, float& pdf)const
{
	u = glm::clamp(u, 0.f, 1.f);
	int left = 0, right = cdf.size() - 1;
	while (right > left)
	{
		int mid = (right + left) / 2;
		if (cdf[mid] <= u)
		{
			left = mid + 1;
		}
		else
		{
			right = mid;
		}
	}
	int offset = glm::clamp(right - 1, static_cast <int>(0), static_cast<int>(cdf.size() - 2));

	pdf = func[offset] / funcInt;
	float du = u - cdf[offset];
	if ((cdf[offset + 1] - cdf[offset]) > 0)
	{
		du /= (cdf[offset + 1] - cdf[offset]);
	}
	else
	{
		du = 0;
	}
	return (offset + du) / Count();
}

int Distribution1D::sampleDiscrete(float u, float& pdf)const
{
	u = glm::clamp(u, 0.f, 1.f);
	int left = 0, right = cdf.size() - 1;
	while (right > left)
	{
		int mid = (right + left) / 2;
		if (cdf[mid] <= u)
		{
			left = mid + 1;
		}
		else
		{
			right = mid;
		}
	}
	int offset = glm::clamp(right - 1, static_cast <int>(0), static_cast<int>(cdf.size() - 2));

	pdf = func[offset] / funcInt;

	return offset;
}

void DevDistribution1D::create(Distribution1D& srcDistribution)
{
	size = srcDistribution.func.size();
	cudaMalloc(&func, size * sizeof(float));
	cudaMemcpy(func, srcDistribution.func.data(), size * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("DevDistribution1D create()::func");

	cudaMalloc(&cdf, (size + 1) * sizeof(float));
	cudaMemcpy(cdf, srcDistribution.cdf.data(), (size + 1) * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("DevDistribution1D create()::cdf");

	this->funcInt = srcDistribution.funcInt;
	this->size = size;
}

void DevDistribution1D::destroy()
{
	cudaSafeFree(func);
	cudaSafeFree(cdf);
	size = 0;
}