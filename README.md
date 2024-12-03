# Procedural Ray Marching Renderer

This project is a procedural ray marching renderer implemented in GLSL, capable of rendering complex 3D objects and scenes using distance fields. It supports various geometric primitives, constructive solid geometry (CSG) operations, noise-based perturbations, lighting, shadows, and texture generation techniques.

## Features

- **Ray Marching and Sphere Tracing**: Utilizes ray marching to render 3D objects defined by signed distance fields (SDFs). The renderer computes intersections using the sphere tracing algorithm.

- **Geometric Primitives**: Supports various geometric shapes, including:
  - Spheres
  - Planes
  - Capsules
  - Boxes
  - Ellipsoids
  - Tori
  - Cylinders

- **Constructive Solid Geometry (CSG)**: Enables the combination of primitives using operations like union and intersection to create more complex shapes.

- **Noise and Perturbations**:
  - Implements Perlin noise and fractional Brownian motion (fBM) to apply procedural textures and perturbations to objects.
  - Adds detail and realism by deforming surfaces with noise functions.

- **Lighting and Shading**:
  - Supports Phong shading with ambient, diffuse, and specular components.
  - Calculates soft shadows using multiple light samples, as well as hard shadows.
  - Provides ambient occlusion to simulate indirect lighting.

- **Texture Generation**:
  - Creates textures using functions such as concentric rings and fBM-based patterns for added visual effects.
  - Modifies object appearance by applying procedural textures to the distance fields.

## Project Structure

- **Primitive Functions**: Functions for calculating the signed distance for various 3D shapes, such as `Sphere`, `Plane`, `Box`, etc.
- **CSG Operations**: Functions for combining primitives using union and intersection, like `Union` and `Intersection`.
- **Noise and Perturbation Functions**: Utilities for generating noise (`noise`) and applying perturbations (`Perturb`) to objects.
- **Ray Marching and Tracing**: Functions for ray tracing and computing the intersection with objects using the `SphereTrace` function.
- **Lighting and Shadows**: Functions for calculating lighting, including soft and hard shadows, ambient occlusion, and Phong shading.
- **Texture Generation**: Procedural texture generation functions like `ConcentricRings` for creating patterns on objects.

## Key Algorithms

### Ray Marching

Ray marching is used to render scenes by marching along the direction of a ray and checking for intersections with objects in the scene. The algorithm uses a signed distance field to determine how far to march at each step, reducing the distance to the nearest object.

### Signed Distance Fields (SDF)

Each object in the scene is represented as a signed distance field (SDF), which provides the distance to the nearest surface. Positive values indicate points outside the object, while negative values indicate points inside.

### Fractional Brownian Motion (fBM)

Fractional Brownian motion is used to perturb the signed distance fields, adding procedural detail to surfaces. This technique enables the generation of realistic textures and deformations on objects.

### Ambient Occlusion

The ambient occlusion algorithm calculates the amount of occlusion around a point by tracing rays in random directions. The result simulates soft shadows and adds depth to the scene.

### Soft Shadows

Soft shadows are computed by sampling multiple light points along a segment and checking whether they are occluded. The average occlusion across all samples is used to determine the shadow intensity.

## Getting Started

To use this project, you'll need a GLSL-capable environment such as a shader editor or a game engine that supports GLSL. The code can be integrated into a rendering pipeline by following these steps:

1. **Setup the GLSL environment**: Make sure your rendering framework supports GLSL and ray marching.
2. **Integrate the code**: Use the provided functions for distance fields, lighting, and shading in your shader code.
3. **Customize objects and scenes**: Add primitives and combine them using CSG operations to create custom scenes.
4. **Experiment with lighting and textures**: Adjust the parameters for lighting, shadows, and noise to achieve the desired visual effect.

## Usage

The code is organized into different sections for primitives, noise functions, CSG operations, ray marching, lighting, and textures. You can start by creating basic objects using the primitive functions and then apply transformations, CSG operations, and textures.

