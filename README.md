**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

- Wang Ruipeng
    - [LinkedIn](https://www.linkedin.com/in/ruipeng-wang-681b89287/)
    - [Personal Website](https://wang-ruipeng.github.io/)
- Tested on: Windows 10, i7-10750H CPU @ 2.60GHz 16GB, RTX 2070 Max-Q 8GB

# CART: CUDA Accelerated Ray Tracer

This is my CUDA-accelerated path tracer project. To compile, use CMake to generate Visual Studio 2022 files. Since this project uses precompiled outside dlls, this project currently only supports Windows platform with CUDA 12.6 or later installed.

![cornell.2024-10-03_17-11-30z.5000samp.png](img/cornell.2024-10-03_17-11-30z.5000samp.png)

Source: [Journey](https://sketchfab.com/3d-models/journey-character-clothing-concept-147b14b62268478da7e59fa36c949bae)

## Visual Features

### Refractions

Refraction in path tracing refers to the bending of light as it passes from one medium to another, such as from air to water or glass. In path tracing, refraction is simulated to accurately model the way light behaves as it enters and exits transparent or semi-transparent materials, like glass or water. 

Reference: [PBRv4 9.3](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)

No Color | Blue | Multiple Objects 
--- | --- | ---
![No Color](img/cornell.2024-09-21_22-09-05z.2881samp.png) | ![Blue](img/cornell.2024-10-03_21-37-09z.2399samp.png) | ![Multiple Objects](img/cornell.2024-10-03_21-53-58z.5000samp.png)

### Physically-based Depth-of-field

Physically-based depth-of-field simulates camera focus by jittering rays within an aperture, creating sharp focus on the focal plane and blur for objects outside it. A larger aperture increases the blur, mimicking real-world optics.

Reference: [PBRv4 5.2](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField)

Without Depth-of-field | With Depth-of-field
--- | --- 
![With](img/cornell.2024-10-03_15-30-22z.5000samp.png) | ![Without](img/cornell.2024-10-03_17-11-30z.5000samp%201.png)

### Procedural Textures

Here are some of procedural textures I implemented, mainly utilizing perlin noise and some calculation on uvs.

Checker board semisphere | Rainbow Walls
--- | --- 
![Checker board semisphere](img/cornell.2024-10-03_21-00-21z.5000samp.png) | ![Rainbow Walls](img/cornell.2024-09-25_23-24-42z.5000samp.png)

Broken Avocado | Sun with Spots
--- | --- 
![Broken Avocado](img/cornell.2024-09-28_04-03-12z.5000samp.png) | ![Sun with Spots](img/cornell.2024-10-03_21-18-24z.356samp.png)

### Texture and Normal Mapping + Metallic Mapping

**Texture mapping** is a technique where a 2D image texture is applied to a 3D model’s surface to add details like color or patterns, making the model appear more detailed without increasing its geometric complexity.

**Normal mapping** enhances surface detail by using a texture to simulate small-scale bumps and dents on the surface. It alters the surface normals used for lighting calculations, making the model appear more complex and textured without actually changing the geometry.

Even though I implemented both **OBJ** and **gltf** model import, texture mapping is **only supported for gltf files**.

Reference: [PBRv4 10.4](https://pbr-book.org/4ed/Textures_and_Materials/Image_Texture)

Only Texture Mapping | All Mapping
--- | --- 
![Only Texture Mapping](img/cornell.2024-09-28_15-06-30z.5000samp.png) | ![All Mapping](img/cornell.2024-09-28_03-50-52z.5000samp.png)

The difference is subtle though. I am only putting it here for reference.

### Subsurface scattering

**Subsurface scattering** (SSS) is the process by which light penetrates a translucent material, scatters beneath the surface, and exits at a different point. This creates a soft, glowing effect, as light diffuses through the material instead of reflecting directly off its surface. Since we are doing rat tracing here, the simulation process is natural.

With SSS | Without SSS
--- | --- 
![With SSS](img/cornell.2024-09-28_22-57-16z.5000samp.png) | ![Without SSS](img/cornell.2024-09-28_23-01-44z.5000samp.png)

### Stratified Sampling vs. Cosine Weighed Sampling

**Stratified sampling on a semi-sphere** divides the surface into equal regions and samples within each region, ensuring more even coverage of the hemisphere. This reduces variance in the light sampling process, improving the accuracy of the final render.

**Cosine-weighted sampling on a semi-sphere** generates samples that are more concentrated around the normal of the surface, based on the cosine of the angle to the surface. This technique prioritizes sampling directions where light is more likely to contribute.

Reference: [PBRv3 13.6](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations)

Stratified sampling | Cosine-weighted sampling
--- | --- 
![RR.png](img/RR.png)| ![RR_stratified.png](img/RR_stratified.png)

You can really see that the stratified sampling makes the shadow larger.

### Motion Blur

In my ray tracing, **motion blur** simulates the effect of fast-moving objects appearing blurred due to their motion during the exposure time. My approach is to move different positions for each sample to mimics how light captures an object’s movement over time. And it looks pretty good.

Sphere Blur | Godray by Motion Blur
--- | --- 
![Sphere Blur](img/cornell.2024-09-29_00-35-48z.5000samp.png) | ![Godray by Motion Blur](img/cornell.2024-10-04_01-52-56z.5000samp.png)

### Open Image AI Denoiser

I used this [open source denoiser](https://github.com/RenderKit/oidn) and the results looks stunning. It is an image denoiser which works by applying a filter on Monte-Carlo-based pathtracer output. 

Without Denoiser | With Denoiser
--- | --- 
![Without Denoiser](img/cornell.2024-09-24_22-51-44z.5000samp.png) | ![With Denoiser](img/cornell.2024-09-24_22-45-56z.5000samp.png)

## Performance Optimizations

### Russian Roulette Path Termination

Russian Roulette(RR) path termination is a probabilistic technique used in path tracing to decide when to stop tracing light paths. Paths that contribute less are more likely to be terminated. This technique balances performance and image quality by reducing unnecessary calculations without introducing significant bias.

Reference: [PBRv3 13.7](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting)

The Russian Roulette Path Termination does have subtle impact on the rendering result, but the benefit it brings is also obvious:

Without RR | With RR
--- | --- 
![RR-no.png](img/RR-no.png) | ![RR.png](img/RR%201.png)

For closed scene, I observed a notable increase in time, for the scene rendered above:

|  | With RR | Without RR |
| --- | --- | --- |
| Avg. FPS | 36 | 21 |

As for open scene, the increase is less obvious:

|  | With RR | Without RR |
| --- | --- | --- |
| Avg. FPS | 65 | 57 |

### BVH

**Bounding Volume Hierarchy (BVH)** is a spatial acceleration structure used in ray tracing to speed up the process of finding ray-object intersections. In BVH, objects are grouped into hierarchical bounding volumes that enclose one or more objects. The hierarchy is built as a tree, where each node contains a bounding volume and its child nodes represent smaller volumes or objects.

The performance increase is stunning, for a scene with 70k triangles, the FPS is attached below:

|  | With BVH | Without BVH |
| --- | --- | --- |
| Avg. FPS | 65 | 57 |

### Some broken (but artistically interesting) rendering result

Small Bunny | Crystal Bunny | Why so many spots?!
--- | --- | ---
![Small Bunny](img/cornell.2024-09-24_02-26-18z.5000samp.png) | ![Crystal Bunny](img/cornell.2024-10-03_21-41-54z.1640samp.png) | ![Why so many spots?!](img/cornell.2024-10-03_20-30-34z.5000samp.png)
