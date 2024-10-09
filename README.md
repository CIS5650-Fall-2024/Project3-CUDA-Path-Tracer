CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Kevin Dong
* Tested on: Windows 11, i7-10750H CPU @ 2.60GHz 2.59 GHz, RTX 2060

![Top Image](img/cornell_obj_tree.2024-10-08_10-10-25z.5000samp.png)

## Summary
This repository implements a ray tracer using CUDA. A ray tracer simulates the path of light rays as they interact with 
objects in a scene. During the simulation, we first cast rays from the camera into the scene. These rays then interact 
with objects, and we then record whether the ray hits an object. We then use this information to shade the pixel in the 
image. The final rendered output is created by using imGUI.

The path tracer implemented in this repository is capable of simulating the following effects:

### Visual Effects
- Diffusion
- Perfect Specular Reflection
- Stochastic Anti-Aliasing
- Imperfect Specular Reflection
- Refraction with Fresnel Effects
- Depth of Field

### Mesh Loading
- glTF 2.0 loading, using [tinygltf](https://github.com/syoyo/tinygltf)
- obj loading, using [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)

### Performance Improvements
- Stream Compaction
- Material Sorting
- Russian Roulette Path Termination
- Hierarchical Bounding Volume Hierarchy (BVH)
- Open Image Denoiser AI (OIDN) (Host-side denoising)

Ray tracing is a very complex task with numerous features and optimizations. The functionalities implemented in this 
repository are only a subset of the features available in a full-fledged ray tracer. Besides, due to hardware 
limitations, the performance of the ray tracer is not very impressive. However, we can still clearly see 
performance improvements when we enable certain features.

## Ray Tracing Pipeline
In this section, we will discuss the ray tracing pipeline implemented in this repository.

### Scene Initialization
The scene in this ray tracer is generated in the json format. The json file is then loaded to `scene.h/cpp` where 
materials, cameras, and objects are initialized. The scene is then uploaded to the GPU.

Several types of objects are supported: sphere, cube, and mesh. The mesh object can either be loaded from a glTF file 
or an obj file. If BVH is enabled, the mesh object will be converted to a BVH tree, which will greatly increase the 
traversal process that will be discussed later.

### Ray Generation
In this function, we generate rays from the camera. This is also where we implement stochastic antialiasing and depth 
of field. The rays are then uploaded to the GPU.

### Compute Intersections
In this function, we compute the intersections between the rays and the objects in the scene. We use the BVH tree to 
accelerate the mesh intersection process.

### Shading
In this function, we shade the pixels based on the intersection information. This is where we apply diffuse/specular, 
etc. shading to the pixels.

### Post-Processing
In this part, we perform stream compaction to remove the terminated rays, and we also use Open Image Denoiser AI (OIDN) 
to denoise the image once the ray tracing is done or after a few iterations.

## Visual Outcomes

### Diffusion
Diffusion can also be thought as roughness == 1.0.
![diffusion](img/cornell_test.2024-10-07_02-23-04z.5000samp.png)

### Perfect Specular Reflection
![perfect specular reflection](img/cornell_test.2024-10-07_02-35-22z.5000samp.png)

### Imperfect Specular Reflection (Adjustable Roughness)
Setting roughness to be 0.1, 0.5, and 0.9 respectively:
![imperfect specular reflection](img/cornell_test.2024-10-07_02-48-37z.5000samp.png)

### Refraction with Fresnel Effects
A perfect specular sphere on the left, a glass sphere on the right, and refractive ground:
![refraction with fresnel effects](img/cornell_refraction.2024-10-07_03-01-31z.5000samp.png)

### Depth of Field
![depth of field](img/cornell_dof.2024-10-07_03-12-24z.5000samp.png)

### Mesh Loading

| glTF                                                                          | obj                                                                                       |
|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| ![glTF mesh loading](img/cornell_gltf_duck.2024-10-08_07-34-22z.5000samp.png) | ![obj mesh loading](img/cornell_obj_tree_bigTree_chair.2024-10-08_06-44-46z.5000samp.png) |

### OIDN Denoising

| Before Denoising                                                                  | After Denoising                                                                  |
|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| ![before denoising](img/cornell_obj_tree_chair.2024-10-06_08-11-17z.5000samp.png) | ![after denoising](img/cornell_obj_tree_chair.2024-10-08_09-36-29z.5000samp.png) |

## Performance Analysis
We will now analyze the performance of the ray tracer with different features/optimizations enabled. The scene used for 
performance analysis is the standard Cornell Box with a perfect specular sphere in it. To get a sense of how it looks 
like, please refer to the Perfect Specular Reflection image above. The default kernel test size is 128.

### Stream Compaction
For stream compaction, we will compare the performance with and without stream compaction, the performance 
change with different kernel sizes, and the performance change within a single iteration. For testing purposes, we limit 
the number of iterations to 500.

Before testing, our hypothesis is that enabling stream compaction will indeed increase the performance of the ray tracer 
if not significantly. This is because stream compaction will remove the terminated rays, which will reduce the number of 
threads that need to be processed in the next iteration.

#### Performance Change with on/off

|             | Stream Compaction Enabled | Stream Compaction Disabled |
|-------------|---------------------------|----------------------------|
| Average FPS | 9.31167                   | 4.73829                    |

As we can see from the table above, enabling stream compaction increases the performance of the ray tracer by almost 
100%. We can see that stream compaction is a very important optimization for the ray tracer.

#### Performance with different kernel sizes
![kernel_size graph](img/Figure_3.png)

From the graph, we can see that when kernel size is 16, the performance is the best. This is because the number of 
divergence within kernel execution may increase as the kernel size increases - as we can see from the code, we are 
using a lot of if-else statements, so the divergence may be significant. The performance difference increases 
a bit when the kernel size is 128, which may indicate that kernel size 128 may be able to utilize the GPU better than 
other kernel sizes as 64 or 256.

#### Performance within a Single Iteration
![singe iteration graph](img/Figure_2.png)

As we can see, as the bounce number increases, the number of paths decreases as well as time passed per bounce. This 
shows that the effect of stream compaction is more significant as the number of bounces increases.

### Material Sorting
For material sorting, we will compare the performance with and without material sorting. Our hypothesis is that 
enabling material sorting will increase the performance of the ray tracer, but if there are only a few materials, the 
overhead of sorting may not be worth it.

When we test the performance on the default scene, we got:
|             | Material Sorting Enabled | Material Sorting Disabled |
|-------------|--------------------------|---------------------------|
| Average FPS | 9.33258                  | 17.0346                   |

When we test the performance on the title image scene, we got:
|             | Material Sorting Enabled | Material Sorting Disabled |
|-------------|--------------------------|---------------------------|
| Average FPS | 4.84631                  | 4.78012                   |

From the results, we can see that enabling material sorting does not significantly increase the performance of the ray 
tracer. This may be caused by the fact that the overhead of sorting is really large compared to the performance 
increase.

### Russian Roulette Path Termination
For Russian Roulette Path Termination, we will compare the performance with and without Russian Roulette Path Termination.

|             | Russian Roulette Enabled | Russian Roulette Disabled |
|-------------|--------------------------|---------------------------|
| Average FPS | 9.04456                  | 8.25438                   |

We can see that enabling Russian Roulette Path Termination increases the performance of the ray tracer, but also not by 
a significant amount.

### Hierarchical Bounding Volume Hierarchy (BVH)
For BVH, we will compare the performance with and without BVH. The test scene is the scene with only an obj tree. We 
use this simple mesh loading scene because of the hardware limitation - having more complex scenes will make the scene 
completely unrenderable (really really slow).

|             | BVH Enabled | BVH Disabled |
|-------------|-------------|--------------|
| Average FPS | 10.8595     | 7.79822      |

We can see that enabling BVH increases the performance of the ray tracer by a significant amount. This is because BVH is 
really useful for accelerating the intersection process for mesh objects.

## References
Many of the features were implemented with the help of Physically Based Rendering: From Theory to Implementation (PBRT), 
which includes detailed explanation of many features with pseudocode explanations. Other references include official 
website documentation/guide for libraries or videos/stackOverflow. Here is a detailed list of references:
- [Diffuse/Perfect Specular (PBRTv4 9.2)](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection)
- [Stochastic Anti-Aliasing](https://paulbourke.net/miscellaneous/raytracing/)
- [Imperfect Specular Reflection (PBRTv4 9.3)](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)
- [Refraction with Fresnel Effects (PBRTv4 9.4)](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)
- [Depth of Field (PBRTv4 5.2.3)](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField)
- [glTF structure](https://www.slideshare.net/slideshow/gltf-20-reference-guide/78149291#1)
- [sample glTF models](https://github.com/KhronosGroup/glTF-Sample-Models)
- [Russian Roulette (PBRTv4 14.5.4)](https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Path_Tracing)
- [BVH (PBRTv4 4.3)](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#BVHAccel::recursiveBuild)
- [Open Image Denoiser](https://www.openimagedenoise.org/)
- [sample obj models](https://free3d.com/3d-models/obj)