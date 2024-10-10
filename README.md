# CUDA Path Tracer


* Maya Diaz Huizar
* Tested on: Windows 10, AMD Ryzen 7 5800X @ 3.8GHz 32GB, NVIDIA RTX 3080 10GB

---

## Overview

This project is a CUDA-based path tracer capable of rendering globally illuminated scenes efficiently. The path tracer supports various features such as diffuse and specular materials, Russian roulette path termination, anti-aliasing, and GLTF mesh loading with texture mapping. The implementation focuses on optimizing performance while producing high-quality images.

## Features

### Core Features

- **Ideal Diffuse Surfaces:** Implemented using cosine-weighted hemisphere sampling.
- **Perfectly Specular Reflective Surfaces:** Utilized `glm::reflect` for mirror-like reflections.
- **Stream Compaction:** Applied to path segments to improve performance by removing terminated rays.
- **Material Sorting:** Path segments are sorted by material type to enhance memory coherence and performance.
- **Stochastic Anti-Aliasing:** Implemented by jittering ray directions within each pixel.

### Additional Features

#### Physically-Based Depth of Field

![Depth of Field](img/depth_field.PNG)

- **Description:** Simulated camera aperture effects by jittering rays within a lens aperture. This creates images where objects at certain distances appear sharp while others are blurred.
- **Implementation:** Modified the camera ray generation to include lens radius and focal distance parameters, sampling rays across the lens area.

#### Russian Roulette Path Termination

- **Description:** Probabilistically terminates paths that contribute little to the final image to improve performance without introducing bias.
- **Implementation:** Added a termination probability based on the path's remaining color intensity, adjusting the path contribution accordingly.

#### GLTF Mesh Loading with Texture Mapping

![GLTF Model](img/gltf.PNG)

- **Description:** Implemented loading of GLTF models using `tinygltf`, allowing the renderer to handle complex meshes with texture mapping.
- **Implementation:** Extended the scene loading capabilities to parse GLTF files, extract mesh data, materials, and textures.

## Results

### Stream Compaction Benefits

Stream compaction significantly reduces the number of path segments processed in each iteration, especially after several bounces where many rays terminate. Below are sample outputs showing the number of paths before and after compaction at each bounce:

```
Number of paths before compaction: 480,000
Number of paths after compaction: 240,606
Number of paths before compaction: 240,606
Number of paths after compaction: 94,324
Number of paths before compaction: 94,324
Number of paths after compaction: 39,271
Number of paths before compaction: 39,271
Number of paths after compaction: 8,428
Number of paths before compaction: 8,428
Number of paths after compaction: 0
```

- **Observation:** Stream compaction reduces computational workload by eliminating terminated paths, leading to performance improvements.

### Material Sorting Impact

- **Observation:** Material sorting did not yield significant performance improvements in scenes with predominantly diffuse materials or few specular materials. In some cases, it slightly decreased performance due to the overhead of sorting.
- **Conclusion:** Material sorting is more beneficial in scenes with a diverse set of materials where divergence in the shading kernel is more pronounced.

### Russian Roulette Path Termination

- **Performance Gain:** Implementing Russian roulette path termination improved overall performance by approximately **15%** in scenes with complex lighting and specular elements.
- **Visual Integrity:** The images maintained visual consistency without noticeable bias introduced by early termination of low-contribution paths.

### Anti-Aliasing

- **Implementation:** Added stochastic sampling by jittering the ray direction within each pixel, resulting in smoother edges and reduced aliasing artifacts.
- **Performance Impact:** Minimal performance overhead due to the simplicity of the implementation.

### Rendered Images

#### Cornell Box with Reflective Box

![Cornell Box](img/depth_field.PNG)

- **Description:** Rendered the Cornell Box scene featuring a reflective box, demonstrating specular reflections and global illumination. Also, depth of field.

#### GLTF Model Rendering

![GLTF Model Render](img/texture.png)

![GLTF Model Render](img/gltf.PNG)


- **Description:** Textured Mapping Example. The more complex model is rendered with my debug shader, as the path tracer struggled with complex models without an acceleration structure.

## Challenges

### Parsing GLTF Files

- **Issue:** GLTF files have a complex structure, making parsing and data extraction non-trivial.
- **Solution:** Utilized the `tinygltf` library to load GLTF models. Had to carefully map GLTF material properties to the renderer's material system.
- **Difficulty:** Handling different texture coordinate conventions and material parameters required extensive debugging and validation.

### Texture Mapping

- **Issue:** Encountered issues with incorrect UV ranges and schemas when applying textures from GLTF models.
- **Solution:** Implemented debug shaders and visualization tools to inspect UV coordinates and texture mappings.
- **Outcome:** Successfully mapped textures to models, but the process was time-consuming due to the need for precise alignment of texture data.

### Debugging Techniques

- **Approach:** Developed debug shaders and added extensive logging in the scene processing pipeline.
- **Benefit:** Allowed for visualization of intermediate data such as normals, UV coordinates, and texture lookups, greatly aiding in identifying and fixing issues.

## Performance Analysis

### Stream Compaction vs. No Stream Compaction

- **Closed Scenes (e.g., Cornell Box):** Stream compaction provided significant performance gains due to a higher number of terminated paths per iteration.
- **Open Scenes:** Lesser impact since rays are less likely to terminate early.
- **Conclusion:** Stream compaction is most effective in closed scenes where light paths are constrained and terminate more frequently.

### Russian Roulette Effectiveness

- **Performance Improvement:** Approximately 15% faster render times in scenes with complex lighting and multiple bounces.
- **Visual Quality:** No perceptible loss in image quality, maintaining unbiased rendering results.

## Future Work

- **Code Cleanup:** Plan to refactor the codebase for better readability and maintainability.
- **UI Improvements:** Enhance the user interface for better control over rendering parameters and scene navigation.
- **Extended GLTF Support:** Add support for additional GLTF features such as animations, more material types, and advanced texture mappings.
- **Acceleration Structures:** Implement bounding volume hierarchies (BVH) or octrees to optimize ray-scene intersection tests, improving performance in scenes with large numbers of objects.

## Acknowledgments

- **GLTF Models:** Used models from the [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository, licensed under [CC-BY 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- **Libraries:**
  - **tinygltf:** For loading GLTF models.
  - **GLM:** OpenGL Mathematics library for vector and matrix operations.

## Project Reflections

### Lessons Learned

- **Importance of Debugging Tools:** Developing custom debug shaders and logging mechanisms was crucial in diagnosing and fixing issues, especially when dealing with complex models and textures.
- **Understanding GLTF Specifications:** Gained a deeper understanding of the GLTF file format and the challenges associated with parsing and using its data.
- **Performance Optimization:** Implementing features like Russian roulette path termination and stream compaction significantly improved performance, highlighting the importance of optimization in GPU programming.

### Potential Improvements

- **Acceleration Structures:** Implementing BVH or octree structures to accelerate ray tracing would greatly enhance performance in scenes with many objects. In hindsight, such an acceralation structure is essentially necessary to support custom model loading, for anything but the simplest of scenes.
- **Additional Material Models:** Adding support for more complex materials, such as glossy reflections and subsurface scattering.
- **Better Material Sorting:** Investigate adaptive sorting strategies to maximize the benefits of material sorting in various scene compositions.

## Contact

For any questions or suggestions, please contact Maya Diaz Huizar at [your.email@example.com](mailto:huizar@seas.upenn.edu).

---

**Note:** All models used in the reference images are from the [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository and are licensed under [CC-BY 4.0 International](https://creativecommons.org/licenses/by/4.0/).

---

## References

- **PBRT Book:** [Physically Based Rendering: From Theory to Implementation](https://www.pbr-book.org/)
- **CUDA Programming Guide:** NVIDIA CUDA Toolkit Documentation
- **GLTF Format Specification:** [glTF Overview](https://www.khronos.org/gltf/)

---

**GitHub Repository:** [https://github.com/yourusername/CUDA-Path-Tracer]([https://github.com/yourusername/CUDA-Path-Tracer](https://github.com/Aorus1/Project3-CUDA-Path-Tracer))
