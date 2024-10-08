CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* ADITHYA RAJEEV
  * [LinkedIn](https://www.linkedin.com/in/adithyar262/)
* Tested on: Windows 11, i7 13th Gen @ 2.40GHz 16GB, GeForce RTX 4050 8GB (Personal)

### Overview

This project implements a CUDA-based path tracer with various advanced rendering features. The path tracer is capable of simulating complex light interactions in 3D scenes, producing physically-based, photorealistic images.

![](img/testRuns/Presentation1.png)

![](img/testRuns/Presentation2.png)

![](img/testRuns/Presentation3.png)


## Features

### Core Functionality

* **BSDF Evaluation**:
  * Ideal diffuse surfaces (Lambertian reflection)

![](img/testRuns/Cornell_Box.png)

  * Perfectly specular-reflective surfaces (mirrors)

![](img/testRuns/Specular_Scene.png)

* **Path continuation/termination** using Stream Compaction
* **Material-based ray sorting** for optimized performance

I implemented material sorting for my path segments and intersections before calling the material shading kernel, expecting it to improve performance. 
However, I was surprised to find that this actually made my path tracer slower. Upon reflection, I realized that this might be due to the lack of diversity in my material types. It seems that in my current scene setup, the overhead from sorting is greater than any performance gains from reducing warp divergence. This suggests that the effectiveness of material sorting might depend heavily on the complexity and variety of materials in the scene.

![](img/testRuns/MaterialSortTable.png)

* **Stochastic sampled antialiasing**

#### With Anti-Aliasing

![](img/testRuns/WithoutAntiAliasing.png)

#### Without Anti-Aliasing

![](img/testRuns/WithAntiAliasing.png)


### Advanced Features

* **Refraction** with Fresnel effects (e.g., glass, water)

![](img/testRuns/Diffuse_Reflection_Refraction.png)

* **Physically-based depth-of-field**

#### Depth of Field Enabled

![](img/testRuns/depth_of_field1.png)

#### Depth of Field Disabled

![](img/testRuns/depth_of_field_disabled1.png)

* **Direct lighting simulation**

#### Direct Lighting Enabled

![](img/testRuns/DirectLightingEnabled.png)

#### Direct Lighting Disabled

![](img/testRuns/DirectLightingDisabled.png)

### Russian Roulette Path Termination

Russian Roulette technique was implemented to terminate unimportant paths early. Our analysis shows:

* fps with Russian Roulette Enabled - 43.2 fps
* fps with Russian Roulette Enabled - 34.9 fps
* 23.78% Increase in FPS
* Negligible impact on image quality

#### Russian Roulette Enabled

![](img/testRuns/RussianRouletteEnabled.png)

#### Russian Roulette Disabled

![](img/testRuns/RussianRouletteDisabled.png)

* **Physically-based rendering with Metallic and Platic materials**

![](img/testRuns/materials.png)

## Bloopers

![](img/testRuns/Blooper1.png)

![](img/testRuns/Blooper2.png)

![](img/testRuns/Blooper3.png)

![](img/testRuns/Blooper4.png)

![](img/testRuns/Blooper5.png)

![](img/testRuns/Blooper6.png)

## References

* [PBRT-v3 Book](https://www.pbrt.org/)
* [Physically Based Rendering: From Theory To Implementation](http://www.pbr-book.org/)
* [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
