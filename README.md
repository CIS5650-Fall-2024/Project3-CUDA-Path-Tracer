CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zhaojin Sun
  * www.linkedin.com/in/zjsun
* Tested on: Windows 11, i9-13900HX @ 2.2GHz 64GB, RTX 4090 Laptop 16GB

## 1. Project Overview
This is a project that uses CUDA for ray tracing. Unlike the traditional rendering pipeline, ray tracing employs a different approach to handling the 3D rendering problem. It works by casting multiple rays from the viewpoint and simulating phenomena such as reflection, refraction, and absorption within the 3D scene, though sometimes these simulations are simplified for computational efficiency. Ray tracing naturally addresses the challenges of handling complex lighting models, such as shadows, in the rendering process! 

Previously, I worked on a NeRF (Neural Radiance Fields) project, which focused on reconstructing the hidden 3D geometry from 2D images. The two projects are somewhat similar but opposite in direction: the former focuses on projecting from 3D to 2D, while the latter reconstructs 3D from 2D. Both methods take full advantage of the rich physics and geometry of the real world and are ingenious approaches!

## 2. Details on Features
### BSDF shading kernel
#### (i) Diffuse Reflection
For perfect diffusion (scattering), it is sufficient to use the Lambertian cosine model. The following image shows the result of diffusion on a sphere.
![sphere_diffuse.png](img%2Fsphere_diffuse.png)
#### (ii) Specular Reflection
For specular reflection, it is sufficient to calculate the reflected ray direction entirely using the reflection model. The following image shows the result of a specular reflection on a sphere. Since there is "the vast emptiness of space" behind the camera, the reflection on this side of the sphere appears black due to the absence of any light source or object in the reflected direction.
![sphere_specular.png](img%2Fsphere_specular.png)
#### (iii) Imperfect Specular Reflection
My idea for imperfect specular reflection is quite simple: it involves weighting the specular reflection and scattering based on the object's roughness, without using random numbers. My reasoning is that for a partially smooth object, the reflected rays do not diverge like they do in refraction. Instead, a 'variance' from scattering is added on top of the specular reflection, and it is reasonable to directly combine the two.
Below is imperfect specular reflection with a roughness of 50%
![sphere_imperfect_specular.png](img%2Fsphere_imperfect_specular.png)
#### (iv) Refraction [An Extra Feature]
For the refraction model, since reflection and refraction are two completely different light paths, introducing randomness is necessary. The ratio between reflection and refraction follows Schlick’s approximation, while the direction of refraction is determined by Snell's Law.
Below is refraction of a transparent sphere with an index of 1.5 similar to glass.
![sphere_transparent.png](img%2Fsphere_transparent.png)
#### Bloopers Time!
To prevent the self-intersection problem, the intersection points should be moved slightly along the normal direction to avoid holding the rays in place. However, when dealing with refraction, a specific challenge arises: if the ray is refracted into the object, the movement must be along the normal into the object's interior to ensure correct refraction behavior. Otherwise, moving in the wrong direction could cause the refraction to incorrectly behave like a reflection, which is undesirable.
Here is an image showing the phenomenon caused by incorrectly moving the intersection point.
![blooper_refraction.png](img%2Fblooper_refraction.png)


### Optimization Techniques
#### (i) Path Termination via Stream Compaction
Using stream compaction to remove rays that terminate early at a given bounce step can reduce the scope of ray calculations, thereby improving speed. There are two cases where rays terminate early: the first is when the ray hits a light source, and the second is when the ray hits nothing and shoots off into distant space. In both cases, the remaining bounce count must be manually zeroed out. Since we want the rays displayed on the screen to be the ones that have been zeroed out (even though some rays will have a color of 0 because they didn't hit a light source), after early termination, these rays must be buffered for final output. Otherwise, they will be discarded!
Based on the chart below, we can see that stream compaction brings a significant performance boost by terminating those rays with no intersections, almost doubling the FPS from 60 to 120 in practical tests.
![stream_compaction.png](img%2Fstream_compaction.png)
However, we can also see that in a closed box, rays cannot escape, and the only condition for early termination is when a ray hits a light source. In this case, the effectiveness of stream compaction is significantly reduced. Since fewer rays are terminated early, the performance improvement is limited compared to an open environment where rays can exit the scene without hitting any objects. As a result, the FPS increase is much smaller, and the computational savings from stream compaction are less noticeable. This highlights the dependency of stream compaction's efficiency on the scene setup, particularly the presence of open spaces where rays can terminate without interaction.

#### (ii) Sort Intersections and Rays by Material
Because only a limited number of materials are actually used in ray tracing, sorting based on materials alone significantly slows down the BSDF. Currently, the branching factor is not solely dependent on the type of material; there are many other branch conditions involved. Therefore, merely sorting the materials does not improve warp performance.

#### Blooper Time! Wait... Is this really bad?
The main issue I encountered with this step was that, initially, my screen appeared brighter than the reference image, as shown below, which indicated that some rays weren’t being properly zeroed out. However, I am confident that stream compaction was functioning correctly. Later, I realized that I needed to terminate rays that didn’t intersect with any objects earlier in the cycle, which makes sense because once a ray flies out of the bounding box, it can't interact with anything further.
![blooper_compaction.png](img%2Fblooper_compaction.png)
For rays without intersections, their color had already been set to zero, and once the bounces are exhausted, the pixel color should be 0 in the image. So something was preventing the bounces from being properly reduced. Upon investigation, I found that the only function in the code responsible for reducing bounces is ScatterRay, which suggested that rays without intersections weren’t calling ScatterRay. Upon further inspection, that was indeed the case! Therefore, manually clearing bounces for rays without intersections is essential.

However, this lighting effect is not entirely without merit: the final effect actually preserves all rays that don’t intersect with any objects. It can be considered as if these rays ultimately hit a light source with a brightness of 1, rather than not intersecting with anything at all. This unintended outcome can, in fact, be treated as a form of global light.

### Smoother visual effects
#### (i) Stochastic Anti-Aliasing
In stochastic anti-aliasing, to achieve sub-pixel level blurring, the sampling range needs to be small enough; otherwise, the perturbation of the rays becomes too strong, and "anti-aliasing" turns into full-blown Gaussian blurring. After some experimentation, a normal sampling distribution width of 0.3 produced the best results. 
The first image below shows the result without anti-aliasing, and the second image shows the result with anti-aliasing applied. The images above have also been processed with anti-aliasing.

![spherecube_no_SAA.png](img%2Fspherecube_no_SAA.png)
![spherecube_SAA03.png](img%2Fspherecube_SAA03.png)

#### (ii) Physically-Based Depth-Of-Field [An Extra Feature]
I have also implemented a simulated "focusing" here. The method involves randomly sampling some points (actually only one here so not very robust) within the camera’s lens circle, and these points converge at the image’s focal point on a projection plane located at a focal distance from the lens circle. To achieve this, the current rays within the lens circle are first projected onto the camera’s z-axis, and then their distances are adjusted to ensure they pass through the focal point. Under this converging effect, the camera focuses on a specific object,
as shown in the image below, where it is focused on the smooth sphere in the background.
![spherecube_DOF.png](img%2Fspherecube_DOF.png)

### Mesh and Texture Support
#### (i) Load OBJ Mesh [An Extra Feature]
Having only default spheres and cubes gets really boring! Therefore, by using tinyobjloader, we can load models from .obj files and convert them into a series of faces and vertices. However, not all faces in an .obj file are already triangulated, so manual triangulation is required. The method used for triangulation is the ear clipping algorithm that we learned in CIS560, which eventually breaks the faces into sets of three vertices for each triangle.

Here, I encountered two main problems. The first issue was how to load multiple different meshes at once. Since the GPU cannot use std::vector, storing a set of vertices for each geom and then transferring them to the GPU would be quite cumbersome. Therefore, I loaded all the meshes and their corresponding textures into a single vector, and each geom only needs to record an offset, allowing for fast indexing later. The second issue was how to calculate the normal for each face of the mesh. For complex models like Mario, I used barycentric interpolation. However, for simpler meshes like cubes or dodecahedrons, I directly calculated the normal using the cross product.

Below is a loaded Mario and dodecahedron mesh scene. To make them clearer, I moved the light slightly forward.
![mesh_load.png](img%2Fmesh_load.png)
I have also implemented bounding volume intersection culling, based on the existing cube intersection test and using an AABB (Axis-Aligned Bounding Box) to exclude certain rays before traversing the mesh. The speed with AABB is about 20% faster, which suggests that the performance gain was limited, likely because Mario's mesh is quite large, which reduces the effectiveness of the culling.

#### (ii) Texture Mapping [An Extra Feature]
Next, let's add texture mapping! It's actually quite simple. You just need to pass the offset of the intersected geometry into the shading kernel and record the interpolated uv coordinates at the point of intersection.
Below are the textured Mario and the dodecahedron.
![mesh_texture.png](img%2Fmesh_texture.png)
I also generated a set of sampled noise to replace the loaded textures. However, the overall speed improvement was negligible because the bottleneck is clearly in calculating the intersection with the mesh. While sampling involves reading textures from global memory, it doesn't require inner loops, so it has less of an impact.

#### (iii) Normal Mapping [An Extra Feature]
Once we have the normal at the intersection point in world coordinates, applying normal mapping becomes very straightforward. We just need to construct an orthogonal basis using the tangent and bitangent, which allows us to transform the normal map from the object's local coordinate system back to the world coordinate system.
Below is the image of the dodecahedron with normal mapping added on top of the texture mapping. 
![mesh_normal.png](img%2Fmesh_normal.png)

Have fun and play around as much as you like! Maybe I will add BVH and better physical models like blurring, dispersion when I am available again in the future!