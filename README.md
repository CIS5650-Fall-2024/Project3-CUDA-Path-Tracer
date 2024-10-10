# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Dineth Meegoda
  - [LinkedIn](https://www.linkedin.com/in/dinethmeegoda/), [Personal Website](https://www.dinethmeegoda.com).
- Tested on: Windows 10 Pro, Ryzen 9 5900X 12 Core @ 3.7GHz 32GB, RTX 3070 8GB

## Summary

This project is a foray into learning CUDA, specifically for graphics programming applications. The goal of this Path Tracer was to take the existing graphics concepts I'm already familiar with, and try to use parallel programming on the GPU with separate kernel invocations to 'simulate' the steps in the graphics pipeline that graphics frameworks typically abstract away. In the end, I wanted to implement as many visually striking features as I could to increase the number of tools I have to create pretty pictures, while making performance boosts to keep the path tracer usable for complex scenes.

### Path Tracing

Path tracing is a rendering technique where we 'shoot' out rays of light from each pixel and follow it's path to figure out what the contribution of light is given to that pixel. The specific color and direction of a ray depends on the materials it intersects and its physical properties. The specific kind of Path Tracing done in this program uses the Monte-Carlo Method in which a random sample is selected and divided by the value of a probability density function to make up for the contribution. Since it's random, one sample isn't enough to obtain an accurate image, so we average samples and eventually get an image we're happy with.

For more on path tracing, check out [PBRT](https://pbr-book.org/4ed/contents), which really helped me out during this project.

### Features Implemented

- Visual Features:
  - [BSDFs/BRDFs](#bsdfsbrdfsbtdf)
  - [Stochastic Sampled Anti-Aliasing](#stochastic-sampled-anti-aliasing)
  - [glTF Mesh Loading & Texture Mapping](#gltf-mesh-loading--texture-mapping)
  - [Reinhard Operator & Gamma Correction](#reinhard-operator--gamma-correction)
  - [Environment Mapping](#environment-mapping)
  - [Ray Dispersion](#ray-dispersion)
  - [Denoising](#denoising)
- Performance Features:
  - [Bounding Volume Heirarchy](#bounding-volume-hierarchy)
  - [Stream Compaction](#stream-compaction)
  - [Material Sorting](#material-sorting)

## Visual Features

These are the features I implemented to help things look physically accurate (and pretty).

### BSDFs/BRDFs/BTDF

A BSDF/BRDF/BTDF is a function that defines a material by how it scatters rays of light when they intersect with it. A BSDF Scatters lights, BRDFs Reflect light, and BTDFs Transmit Light. With these functions, a number of materials were implemented.

The implemented BSDF represents diffuse surfaces, while the BRDF represents perfectly reflective surfaces, such as a mirror.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/bsdf.png" alt="bsdf" width="600"/>
      <p align="center">BSDF Saul Goodman</p>
    </td>
    <td>
      <img src="img/brdf.png" alt="brdf" width="600"/>
      <p align="center">BRDF Saul Goodman</p>
    </td>
  </tr>
</table>
</div>

These Distribution Functions can be used in tandem to represent specular diffuse surfaces like plastic with varying amounts of 'roughness'. The more rough a material is, the more diffuse the surface acts.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/plastic0.png" alt="plastic0" width="600"/>
      <p align="center">0% Roughness</p>
    </td>
    <td>
      <img src="img/plastic30.png" alt="plastic30" width="600"/>
      <p align="center">30% Roughness</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="img/plastic50.png"" alt="plastic50" width="600"/>
      <p align="center">50% Roughness</p>
    </td>
    <td>
      <img src="img/plastic80.png"" alt="plastic80" width="600"/>
      <p align="center">80% Roughness</p>
    </td>
  </tr>
  </tr>
</table>
      <img src="img/plastic100.png"" alt="plastic100" width="500"/>
      <p align="center">100% Roughness</p>
</div>

Finally, we have the BTDF, which transmits light through an object. This can be combined with the BRDF to create a glass like material. BTDF materials have an Index of Refraction (IOR) component which describes how much light should bend (refract) while passing through an object.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/btdf.png" alt="btdf" width="600"/>
      <p align="center">BTDF with IOR 1.33</p>
    </td>
    <td>
      <img src="img/glass.png" alt="glass" width="600"/>
      <p align="center">BTDF/BRDF with IOR 1.6</p>
    </td>
  </tr>
</table>
</div>

### Stochastic Sampled Anti-Aliasing

Aliasing happens during rasterization where jagged edges form on the edges of polygons when drawing to pixels. However, since we are path tracing, we can use a technique called Stochastic Sampled Anti-Aliasing to natually anti-aliase the image. This is done by randomly jittering the ray within each pixel so that the ray will not hit the edge of the object everytime. As we take the average of several samples, this effectively 'blurs' out the edges and gets rid of the aliasing. The more we jitter, the more we get rid of these edges, but the image gets more blurry as well.

<div align="center">
<img src="img/ssaa.png" alt="Image 2" width="1000"/>
<p align="center">Without SSAA (Left) and With SSAA (Right)</p>
</div>

### glTF Mesh Loading & Texture Mapping

In order to have more than just boxes and spheres in our scene, I implemented arbritary mesh loading with the glTF file format. glTF is being used in the industry much more often, and contains inherent references to different texture maps within the file. In order to parse this format, I used the [tinyglTF Library](https://github.com/syoyo/tinygltf) and the Khronos Group's [Guide to parsing glTF files](https://www.slideshare.net/slideshow/gltf-20-reference-guide/78149291#1). Some reinterpret casts later, I had the data loaded into my pathtracer.

After parsing the vertices, we can create flat normals by taking the cross product of the triangle edges. However, it's usually better to perform barycentric interpolation with the vertex normals to obtain 'smooth shading' on the model. Barycentric interpolation was also used to map the model's texture files onto itself.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/gltf1.png" alt="Image 1" width="600"/>
      <p align="center">Flat Normal Shading with Cross Product (Low Poly Look)</p>
    </td>
    <td>
      <img src="img/gltf2.png" alt="Image 2" width="600"/>
      <p align="center">Smooth Shading with Barycentric Interpolation on Normals</p>
    </td>
  </tr>
  <tr>
      <td>
      <img src="img/gltf3.png" alt="Image 3" width="600"/>
      <p align="center">Saul's Albedo Texture Map</p>
    </td>
    <td>
      <img src="img/gltf4.png" alt="Image 4" width="600"/>
      <p align="center">Texture Map applied to Saul with Barycentric Interpolation</p>
    </td>
  </tr>
</table>
</div>

### Reinhard Operator & Gamma Correction

I also implemented some post processing onto the image in order to more accurately represent the scene as if it was captured by a real digital camera. First, I implemented the Reinhard operator, which is used to map the High Dynamic Range Image to a lower range output device, such as digital monitors.

Then the image was gamma corrected from the RGB value of a pixel to the actual luminance of the color so that our eyes can see the colors as if it was captured by a real digital camera.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/noRorG.png" alt="Image 1" width="600"/>
      <p align="center">Without Reinhard Operator & Gamma Correction</p>
    </td>
    <td>
      <img src="img/RnoG.png" alt="Image 2" width="600"/>
      <p align="center">With only Reinhard Operator</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="img/noRG.png" alt="Image 1" width="600"/>
      <p align="center">With only Gamma Correction</p>
    </td>
    <td>
      <img src="img/RG.png" alt="Image 2" width="600"/>
      <p align="center">With both Reinhard Operator & Gamma Correction</p>
    </td>
  </tr>
</table>
</div>

### Environment Mapping

In order to have scenes that take place anywhere other than indoors, it would be ideal to have the outside anything other than the black void that is returned when rays intersect with anything but geometry. To instead have better visuals and have an additional source of light, I implemented spherical environment mapping.

With an high definition image that's mapped onto a sphere, we can sample the color at the pixel that corresponds to the direction our ray bounces to simulate being in an environment.
<br></br>

<div align = "center">
<img src="img/envMap.png" alt="Image 2" width="600"/>
<p align="center">Nice Romantic European Evening with Saul.</p>
</div>

### Ray Dispersion

In some refractive materials, ray dispersion occurs. This happens when some materials refract different rays of light differently based on their wavelength of visible light. This allows for chromatic aberration/prismatic effects.

This was implemented by mapping each wavelength of light from 360 nm to 850 nm to a color and passing it to the GPU as constant memory. Each ray would be randomly assigned a wavelength and be set to that specific color. Based on the material's dispersion coefficient, the ray would refract differently based on its wavelength.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/sphereDispersion.png" alt="Image 1" width="600"/>
      <p align="center">Sphere Dispersion with coefficent 0.2</p>
    </td>
    <td>
      <img src="img/dispersionCube.png" alt="Image 2" width="600"/>
      <p align="center">Cube Dispersion with coefficent 0.4</p>
    </td>
  </tr>
</table>
      <img src="img/dispersionSaul.png" alt="Image 3" width="500"/>
      <p align="center">Saul Dispersion with coefficent 0.8</p>
</div>

Although this creates colorful effects, the image visibly converges slower since each pixel starts off as a different color before it eventually converges to its actual color, even with no glass or dispersive objects in the scene.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/regStart.png" alt="Image 1" width="600"/>
      <p align="center">Pathtracing at Iteration 1 without Dispersion</p>
    </td>
    <td>
      <img src="img/dispersionStart.png" alt="Image 2" width="600"/>
      <p align="center">Pathtracing at Iteration 1 with Dispersion</p>
    </td>
  </tr>
</table>
</div>

### Denoising

Finally, even if we waited for 5000 samples per pixel, most of these images would not have looked very smooth. This is where denoising comes in. I implemented Intel's [Open Image Denoise Library](https://www.openimagedenoise.org/) within my pathtracer to help speed things along and look better. The Library is a pretrained machine learning algorithm built to denoise rendered/path traced images.

To aid with this, I passed in prefiltered (denoised themselves) albedo and normal buffers to contribute to a more accurately denoised image.

<div align="center">
<table>
  <tr>
    <td>
      <img src="img/albedoBuffer.png" alt="Image 1" width="600"/>
      <p align="center">Albedo Buffer</p>
    </td>
    <td>
      <img src="img/normalBuffer.png" alt="Image 2" width="600"/>
      <p align="center">Normal Buffer</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="img/noised.png" alt="Image 1" width="600"/>
      <p align="center">Image @ 1000 spp without Denoising</p>
    </td>
    <td>
      <img src="img/denoised.png" alt="Image 2" width="600"/>
      <p align="center">Denoised Image @ 1000 spp</p>
    </td>
  </tr>
</table>
</div>

The image denoises through a preset frame interval. If we set the denoiser to run often (ex. every other frame), we get a less accurate image, but sometimes it can look pretty painterly/stylized.

<div align="center">
<img src="img/painterly.png" alt="Image 2" width="600"/>
</div>

## Performance Features

Next, I'll go through the features I implemented to make things faster, as well as a performance comparison. For the majority of these toggles, I used preprocessor Macro Definitons in `utilities.h`.

### Bounding Volume Hierarchy

With more complex meshes, it becomes difficult to naively check for intersection with every single triangle, every frame, for each ray, for each bounce! Instead, acceleration structures are used to more efficiently test for which triangles the ray might intersect on a given bounce.

The structure that I implemented was a Bounding Volume Hierarchy, or BVH. This is essentially a tree of triangles in three dimensional world space in which we can traverse by testing each "node" of the tree's intersection with our ray to change our overall complexity from O(n) to O(logn), where n is the number of triangles in a scene.

#### BVH Performance Based on Scene Complexity

<img src="img/dragon.png" alt="Image 1" width="300"/>

To test the performance of the BVH, I recorded the average FPS and time taken to build the tree. This was done in an open scene with an environment map and one type of model. The model had a regular white diffuse material with no texturing. Denoising and Dispersion were turned off. The Saul model contained 2109 triangles so it was duplicated around the scene for subsequent trials between 2k to 33k Triangles. I used the Stanford Dragon Model for the 91k to 364k triangles trials since it had a larger amount of base triangles.

<img src="img/bvhGraph.png" alt="Image 1" width="600"/>

From looking at the data, we can easily tell that there's a big speed up to be gained from using the BVH in every case above 2000 triangles. The performance does fall quick as more triangles are in the scene, but it is most definitely preferable to not using an acceleration structure. The BVH is set to stop building nodes when they have 8 or more triangles, meaning after traversing at most log(n) triangles, there are at most 8 triangles that we have to test intersections on. This is almost always preferable to naively checking each triangle. The potential slow down for a BVH is for lower triangle scenes in which the time taken to traverse the BVH is not worth it. As for the building time, it seems to increase exponentially, but as it only happens once before the scene and is still a fraction of a second, this is not a problem.

### Stream Compaction

When we path trace, if a ray doesn't intersect anything on any bounce, we stop following it. But we still launch the thread that ends up doing nothing. Instead, we can use Stream Compaction to sort the array of paths in parallel such that all rays that still have to bounce are in the front. This helps with warp divergence and decreases warp sizes so that threads that perform similarly will be in the same warp. The stream compaction was implemented with the Thrust Library.

#### Scene Performance with Stream Compaction

To test Stream Compaction Performance, I ran both a closed and open scene with the denoiser disabled, and with one model with texturing disabled. The model was a glTF mesh and utilized the BVH data structure and traversal.

<img src="img/streamCompactGraph.png" alt="Image 1" width="600"/>

We can see a big speed up for using compaction in the open scene as depth increases. This makes since since rays will usually bounce once before running out of meshes to intersect in an open scene. Although the overhead cost of compacting the stream each depth iteration causes it to be slower when the trace depth is low, we will realistically not use that amount of trace depth in order to get an accurate render. In a closed scene, the benefits of stream compaction are not as great and only start being obvious at higher trace depths, but as these are more reliable trace depths for renders, using the stream compaction optimization is a good choice.

### Material Sorting

Another optimization to also help with warp divergence is to sort rays by the type of material they intersected with before we go to shade them. Although we have the overhead cost of sorting the rays each depth iteration, this can help reduce warp divergence as threads on the same warp have a greater chance to perform the same instruction and not be locked since different materials perform different operations on the shader.

#### Scene Performance with Material Sorting

<img src="img/matSortGraph.png" alt="Image 1" width="600"/>

Unfortunately, the material sort always results in a reduction of performance. The overhead cost is not worth the amount of materials in which we test this optimization. There might be a chance that this might be worth for scenes that contain hundreds's of materials, but this is not realistic for our purposes in this kind of path tracer.

## Bloopers

Since this wasn't the easiest project to implement, here are some blooper images for fun!

<img src="img/bloopers/bvhBox.png" alt="Image 1" width="600"/>

Accidentally Visualized my BVH

<img src="img/bloopers/denoiseMuch.png" alt="Image 1" width="600"/>

Denoised into the Dark Dimension...

<img src="img/bloopers/dividezero.png" alt="Image 1" width="600"/>

I think I divided by zero somewhere

<img src="img/bloopers/funnyTransmission.png" alt="Image 1" width="600"/>

My path tracer is deterministic...is this real time rendering?

<img src="img/bloopers/striped_avocado.png" alt="Image 1" width="600"/>

I accidentally peeled my avocado with this BVH thing.

## References

### Models

- [glTF Sample Models (Avocado, Dragon, etc.)](https://github.com/KhronosGroup/glTF-Sample-Models)

- [Saul Goodman Model - by Cesar da Mata](https://sketchfab.com/3d-models/saul-goodman-531a84899eb44401a1ff5d8f735aa6ad)

### Resources

- [PBRT](https://pbr-book.org/4ed/contents)

- [Open Image Denoise Library](https://www.openimagedenoise.org/)

- [Paul Bourke's SSAA Notes](https://paulbourke.net/miscellaneous/raytracing/)

- [glTF Guide](https://www.slideshare.net/slideshow/gltf-20-reference-guide/78149291#1)

- [glTF Parsing Troubleshooting](https://www.reddit.com/r/vulkan/comments/oeg87z/loading_some_indexed_gltf_meshes_cause_weird/)

- [tinyglTF Library](https://github.com/syoyo/tinygltf)

- [glTF Spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)

- [BVH Guide](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)

- Adam Mally's CIS 5610 Class for teaching me Rendering, Environment Maps, and my previous Path Tracer.
