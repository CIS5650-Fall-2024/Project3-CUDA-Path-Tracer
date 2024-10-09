CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Christine Kneer
  * https://www.linkedin.com/in/christine-kneer/
  * https://www.christinekneer.com/
* Tested on: Windows 11, i7-13700HX @ 2.1GHz 32GB, RTX 4060 8GB (Personal Laptop)

## Part 1: Introduction

|![cornell 2024-10-06_03-36-00z 5000samp](https://github.com/user-attachments/assets/c98a8a83-12b8-498e-af44-8ca9f28be19e)|
|:--:|
|*D.Va & Meka*|

|![cornell 2024-10-06_22-05-35z 5000samp](https://github.com/user-attachments/assets/5d19131a-a646-4a3d-b858-afb0eeb62b38)|
|:--:|
|*Meka*|

|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/799935af-e51a-4f2e-96ba-cfa50d2cbdd1)|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/84b47b80-5891-421f-b0fe-7959071d3969)|
|:--:|:--:|

|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/78120cec-a94a-4369-9600-53a9bfac1d88)|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/9c97f0a1-03e8-43af-b3b4-e91aa7ac7d5d)|
|:--:|:--:|


In this project, I implemented a CUDA-based path tracer capable of rendering globally-illuminated images. The path tracer incorporates the following features:
* Core
  * BSDF evaluation: Perfectly Specular, Perfectly Diffuse
  * Path continuation/termination using Stream Compaction
  * Material sorting on intersections
  * Stochastic sampled antialiasing
* Extra
  *  BSDF evaluation: Refraction, Microfacet
  *  Hierarchical spatial data structures: BVH
  *  Russian roulette path termination
  *  Arbitrary mesh loading, with texture mapping and mtl file loading (OBJ)
  *  Environment map
  *  Physically-based depth-of-field
  *  Open Image AI Denoiser
  
## Part 2: Features

In this part, we will go over the most important features in this path tracer.

### Part 2.1: BSDF evaluation

**BSDF** stands for **Bidirectional Scattering Distribution Function**. It generalizes how light is scattered by a material and can include both reflection and transmission of light. This path tracer supports the following material BSDFs: diffuse, specular, refractive (like glass), and microfacets.

|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/e7de5683-56dc-4575-8944-a5ff0f748c33)|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/4b399a98-8618-4ead-a29f-489be68caa3e)|![cornell 2024-10-06_23-04-01z 5000samp](https://github.com/user-attachments/assets/88f4c76f-154d-4df8-8fd7-36e443fa9f5c)|
|:--:|:--:|:--:|
|*Diffuse*|*Specular*|*Refractive*|

**Microfacets** are a concept used to model how light interacts with rough surfaces at a microscopic level. Rather than treating surfaces as perfectly smooth, microfacet models assume that a surface is made up of many tiny facets, each reflects light in different directions. In mircofacet model, **roughness** is used to descirbe how rough the surface is.

|![cornell 2024-10-06_22-05-35z 5000samp](https://github.com/user-attachments/assets/8fd7292e-8e43-4780-985d-7db4847b6985)|
|:--:|
|*Microfacets with roughness from 0.1 - 0.9*|

### Part 2.2: Stochastic sampled antialiasing

**Stochastic sampled antialiasing** is a technique used to reduce aliasing (jagged edges) in rendered images by oversampling pixels using randomly jittered sample points. Instead of sampling the color at the center of each pixel, stochastic SSAA takes several samples per pixel, each with a slight random offset (jitter), producing smoother edges and reducing artifacts.

|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/a5ccd028-4cc5-4422-9de8-0e0f341c140a)|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/de44fdd8-366f-42fb-b838-7a2d326cba42)|
|:--:|:--:|
|*Without AA, more jagged edges*|*With AA, smoother edges*|

### Part 2.3: Hierarchical spatial data structures

**Bounding Volume Hierarchy** (BVH) is a spatial data structure used in ray tracing to efficiently speed up the process of finding ray-object intersections. It groups objects into a hierarchy of nested bounding volumes, allowing for quick rejection of large sets of objects that don't intersect with a ray, thus reducing the number of intersection tests from O(n) (naively iterate through all objects in the scene) to O(log(n)) (efficiently iterate through the tree of bounding volumes).

|![image](https://github.com/user-attachments/assets/fe4fec66-db4b-4b0e-83bb-5332bdb6a2e4)|
|:--:|
|*Example of bounding volume hierarchy, Schreiberx - Own work, from: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy#/media/File:Example_of_bounding_volume_hierarchy.svg*|

Later in Part 3, we will compare the performance of our path tracer with and without BVH.

### Part 2.4: Arbitrary mesh loading

This path tracer supports OBJ mesh and optionally MTL file loading. OBJ is widely supported and used in 3D modeling software and game engines to describe the shape of 3D objects, whereas the MTL file format is used to store material properties for objects in an associated OBJ file.

OBJ meshes are loaded into the path tracer as triangles, which are then used to construct the BVH tree. At each intersection, uvs and normals are interpolated using Barycentric Interpolation, which allows us to sample the texture and shade the mesh properly.

<p align="center">
<img width="600" alt="cornell 2024-10-06_23-09-46z 3004samp" src="https://github.com/user-attachments/assets/0d828b30-0f88-494b-a452-95ac04264e34">
</p>

MTL file specifices the diffuse texture or albedo of the mesh, with other properties like **Ks** (Specular color) & **Ns** (Shininess), which gets translated into BSDF propeties supported by our path tracer. For example, shininess is mapped to roughness in our microfacet model, where higher shininess means that the material is less rough.

<p align="center">
<img width="600" alt="Screenshot 2024-10-06 164438" src="https://github.com/user-attachments/assets/249f500b-8618-43b7-860c-1640c95e799b">
</p>

### Part 2.5: Environment map

When a ray escapes the scene without hitting an object, it is considered to have hit the environment map. The direction of the ray is used to sample the environment texture to determine the color of the background or reflections. This path tracer uses spherical mapping, where the environment is treated as if it's projected onto a sphere around the scene.

<p align="center">
<img width="600" alt="cornell 2024-10-06_23-09-46z 3004samp" src="https://github.com/user-attachments/assets/1e7908e7-d8ad-4f38-9ef4-680ba291fda3">
</p>

### Part 2.6: Physically-based depth-of-field

Physically-based Depth of Field simulates the way real cameras blur objects that are not at the point of focus. It is implemented by jittering rays within an aperture, which creates the blurry effect.

**Focal length** controls the distance between objects in focus and the camera. As focal length increases, objects further away are more in focus.

|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/d536b798-c34f-4832-a69c-fed93eb3d281)|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/233f331b-7b75-4c26-960c-0ead3dabdcde)|![cornell 2024-10-06_23-04-01z 5000samp](https://github.com/user-attachments/assets/0b8e0f40-1768-4012-a79e-38acc93bb0ed)|
|:--:|:--:|:--:|
|*Focal Length = 6.5*|*Focal Length = 8.5*|*Focal Length = 10.5*|

On the other hand, **aperture radius** has an effect on the amount of blur for out-of-focus objects. As apertrue radius incrases, objects out of focus appear to be more blurry.

|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/5dd3a09e-b048-43ec-8818-83c0ffd7814d)|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/2e2c7af6-278e-4ce5-9b95-dbef8a07b545)|![cornell 2024-10-06_23-04-01z 5000samp](https://github.com/user-attachments/assets/282b3d99-ba20-4886-a210-282a2d80d313)|
|:--:|:--:|:--:|
|*Aperture Radius = 0.2*|*Aperture Radius = 0.5*|*Aperture Radius = 0.8*|

### Part 2.7: Open Image AI Denoiser

This path tracer uses [OIDN](https://github.com/RenderKit/oidn), which is an open-source image denoiser that works by applying a filter on Monte-Carlo-based pathtracer output. Using a denoiser allows the render to appear at a higher quality with fewer iterations. 

Below shows the comparison of renders of the **same scene** at **500** iterations, with and without denoiser.

|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/13f40525-652d-46e4-97cc-6bd787e8c409)|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/3ba5c06c-693e-43e5-a0f3-f17630cce1ef)|
|:--:|:--:|
|*Without Denoiser*|*With Denoiser*|

## Part 3: Performance Analysis

In this part, we discuss the performance of our path tracer under different performance improvement techiniques.

### Part 3.1: Stream compaction

**Stream compaction** is an optimization technique that works by reducing unnecessary computations related to rays that are no longer active. At each iteration, we "remove" inactive rays from the list of active rays, effectively spending our computational resource only on active rays in the scene. We compare the number of active rays at each trace depth under different scenes.

![Semi-Open Scene (1)](https://github.com/user-attachments/assets/36babc3f-2535-4e03-95f9-87962e15d4e8)

![Open Scene (With Stream Compaction)](https://github.com/user-attachments/assets/45d864c6-dbd0-44e5-9d9f-d6ebc7d96906)

In both **semi-open** & **completely open** scenes, the number of active rays after each trace depth is significantly reduced after stream compaction. In fact, in a completely open scene, we are only left with single digit active rays at the second pass. It shows that stream compaction is particularly benificial when the scene is somewhat open, where rays are more likely to be terminated because they have not hit any object.

![Closed Scene](https://github.com/user-attachments/assets/d99c010e-290c-4beb-8ad2-cc4ad27f4f7e)

In **closed** scenes, rays bounce between surfaces without escaping, so even at depth = 10, many remain active. Stream compaction removes terminated rays, but most rays keep bouncing. In such cases, the overhead of performing stream compaction may be more dominant than the removal of inactive rays. In fact, our path tracer experiences a slight framerate decrease when using stream compaction in closed scenes.

### Part 3.2: Material sorting

**Material sorting** is a technique used to reduce warp divergence particularly in BSDF evaluation kernels where divergence is inevitable due to different material types. Theoretically, sorting rays by material type before passed into the shading kernal should reduce warp divergence, thus improve performance. However, due to the simplicity of our scenes, the overhead of material sort outperforms its benifit. Under the regular cornell box scene with **5 - 10** material types, material sorting actually introduces almost **50% FPS decrease**; on the other hand, when more complex objs are loaded with **15 - 20** different materials, we are still seeing a slight FPS decrease when material sorting is performed.

Due to limited resource, we were unable to test this technique in more complex scenes, which might offer deeper insights into the trade-off between sorting overhead and warp divergence reduction. Nonetheless, for simple scenes like ours, the overhead of sorting appeared much more dominant.

### Part 3.3: Bounding Volume Hierarchy (BVH)

As explained previously, **Bounding Volume Hierarchy** (BVH) is a spatial data structure used to efficiently speed up the process of finding ray-object intersections.

![chart](https://github.com/user-attachments/assets/9f7bab1e-b2a5-4ec7-b4a8-29edbc457311)

In simple scenes with only 10 geometries, using a BVH may not provide much benefit, as the overhead of stack traversal can outweigh its advantages. However, as scene complexity increases, BVH becomes indispensable. For example, in scenes with 6,000 triangles, we observed a **40x speedup** using BVH, and in even more complex scenarios—such as rendering the D.Va OBJ model with 400,000 triangles—the path tracer becomes completely unrunnable without BVH, but achieves an impressive **5 FPS** with it. Without BVH, rendering such scenes would be impossible.

While there is some overhead associated with constructing the BVH, particularly for large scenes, this overhead is typically only incurred once at the start for static scenes, making it irrelevant for real-time performance afterwards. For any reasonably complex scene, BVH (or other spatial acceleration structures) is absolutely necessary to ensure that the path tracer runs efficiently.

### Part 3.4: Russian roulette path termination

**Russian Roulette** is a probabilistic technique used to terminate rays early in order to improve performance, without introducing bias. Instead of always tracing rays for a fixed number of bounces, Russian Roulette allows the path tracer to probabilistically decide whether to terminate or continue a ray.

![Semi-Open Scene (3)](https://github.com/user-attachments/assets/cc9cfb2c-5ff0-448c-949c-210f540d67f8)

![Closed Scene (1)](https://github.com/user-attachments/assets/26604814-07ad-4566-a934-709bdaf467b9)

In **semi-open** scenes, where rays can escape the scene, Russian Roulette termination is less impactful because many rays naturally terminate, leading to only a slight improvement in performance.

However, in **closed** scenes, rays would continue bouncing until they reach the maximum number of allowed bounces without Russian Roulette. This creates a significant computational load, as rays are continually traced even when their contribution becomes minimal. In these cases, Russian Roulette can terminate rays early, significantly reducing the number of unnecessary bounces, which leads to a major performance boost. In fact, we can witness an almost **50%** FPS increase with Russian Roulette in closed scenes.

## Part 4: References & Credits
* BSDF implementation:
  * PBRT: https://pbrt.org/
  * Codebase & my own implementation from CIS 5610: Advanced Rendering
* BVH: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/ 
* OIDN : https://github.com/RenderKit/oidn
* tinyobj : https://github.com/tinyobjloader/tinyobjloader
* D.Va OBJ model:
  * character from *Overwatch*, Blizzard Entertainment
  * model from https://www.cgtrader.com/3d-print-models/miniatures/figurines/dva-and-meka
