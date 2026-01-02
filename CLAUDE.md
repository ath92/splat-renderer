# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a WebGPU-based 3D SDF (Signed Distance Field) modeler that uses GPU-powered gradient descent to project points onto surfaces defined by signed distance functions. The project demonstrates real-time GPU compute shader pipelines for particle simulation and 3D visualization with camera controls.

## Build and Development Commands

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview

# Type check
tsc --noEmit
```

## Architecture

The entry point (`index.html`) loads the main application from `src/main.ts`. Core implementation files are in the `src/` directory, with WGSL shaders in `src/shaders/`.

### Core GPU Pipeline

The application follows a multi-stage GPU pipeline pattern:

1. **Gradient Evaluation** (Compute Shader)
   - `GradientSampler` evaluates SDF gradients at each particle position
   - Uses analytic gradient functions (`sdgSphere`, `sdgBox`, etc.) defined in WGSL
   - Outputs: distance and gradient vector for each point

2. **Position Update** (Compute Shader)
   - `PositionUpdater` projects points toward surface using gradients
   - Implements instant projection: `newPos = pos - normalize(gradient) * distance`
   - Uses ping-pong buffers to avoid read/write conflicts
   - Note: Instant projection can cause uneven distribution; consider using small step sizes for better coverage

3. **Rendering** (Render Pipeline)
   - `Renderer` visualizes particles in 3D space
   - Renders particles as instanced quads colored by distance from surface
   - Uses camera matrices for 3D projection

### Key Components

**PointManager**
- Manages ping-pong GPU buffers for particle positions (3D coordinates)
- `getCurrentPositionBuffer()` / `getNextPositionBuffer()` / `swap()` pattern
- Dynamically calculates point count based on scene (30k base points per primitive, scaled by surface area)
- Distributes initial positions on AABB surfaces (6 faces) proportionally by primitive surface area
- Provides excellent convergence by starting points on bounding box surfaces

**GradientSampler**
- Creates compute pipeline for gradient evaluation
- Bind group: uniforms, position buffer (read), gradient buffer (write)
- Optional async readback for debugging (non-blocking)

**PositionUpdater**
- Creates compute pipeline for position updates
- Bind group: uniforms, current positions (read), gradients (read), next positions (write)

**Renderer**
- Render pipeline for particle visualization
- Uses indirect rendering (`drawIndirect`) for efficient particle rendering
- Bind groups shared across pipelines where possible

**Camera / OrbitCameraController**
- Camera manages view-projection matrices using gl-matrix
- OrbitCameraController handles mouse/touch input for camera rotation and zoom

### WGSL Shader Organization

- SDF functions use the naming convention `sd*` for distance-only, `sdg*` for distance+gradient
- Main scene defined in `sceneSDF()` function in `src/shaders/compute-gradient.wgsl`
- Gradient functions must return `vec4f(distance, gradient.xyz)` for 3D
- Compute shaders use `@workgroup_size(64)` for optimal GPU utilization

### Buffer Patterns

**Ping-Pong Buffers**
- Position buffers swap each frame to avoid read/write hazards
- Pattern: compute reads from current, writes to next, then swap

**Storage Buffers**
- Positions: `array<vec3f>`
- Gradients: `array<vec4f>` (distance + 3D gradient components)

**Uniform Buffers**
- `mat4x4f viewProj` - view-projection matrix (64 bytes)
- `vec3f cameraPos` - camera position (12 bytes)
- `f32 time` - elapsed time (4 bytes)
- Total: 80 bytes

### Render Loop Structure

```typescript
// 1. Update uniforms
device.queue.writeBuffer(uniformBuffer, 0, uniformData);

// 2. Create command encoder
const commandEncoder = device.createCommandEncoder();

// 3. Compute passes (gradient evaluation, position update)
gradientSampler.evaluateGradients(...);
positionUpdater.updatePositions(...);

// 4. Submit compute work
device.queue.submit([commandEncoder.finish()]);

// 5. Swap position buffers
pointManager.swap();

// 6. Render (separate command encoder)
renderer.render(...);
```

### Important Constraints

- WebGPU requires Chrome 113+, Edge 113+, or recent Chrome Canary
- Buffer sizes must be aligned to 4 bytes
- Uniform buffer structs require padding for alignment
- Compute shader workgroup size is typically 64 or 256
- Always separate compute and render command encoders to avoid pipeline conflicts

## Scene Definition System

The SDF scene is defined CPU-side and compiled to WGSL shaders. This provides performance with flexibility.

### Creating a Scene

```typescript
import { SDFScene } from "./sdf/Scene";
import { Sphere, Box } from "./sdf/Primitive";
import { vec3 } from "gl-matrix";

const scene = new SDFScene();

// Add primitives
const sphere = new Sphere({
  id: "sphere1",
  position: vec3.fromValues(0, 0, 0),
  radius: 0.5,
});

const box = new Box({
  id: "box1",
  position: vec3.fromValues(1, 0, 0),
  size: vec3.fromValues(0.3, 0.3, 0.3),
});

// Combine with operations
scene.add(sphere).add(box).smoothUnion(0.1);
```

### Supported Primitives

- **Sphere**: position (vec3), radius (f32)
- **Box**: position (vec3), size (vec3)
- **Torus**: position (vec3), majorRadius (f32), minorRadius (f32)
- **Capsule**: position (vec3), height (f32), radius (f32)

### Supported Operations

- **union()**: Standard union (min)
- **intersection()**: Intersection (max)
- **subtraction()**: Subtract second from first
- **smoothUnion(k)**: Smooth blend with parameter k (uses polynomial smooth min with correct gradient computation)

### Animating the Scene

Primitive parameters can be modified every frame without shader recompilation:

```typescript
// In render loop
sphere.position[0] = Math.sin(time);
sphere.radius = 0.5 + 0.1 * Math.cos(time);

// Update GPU buffer
gradientSampler.updateSceneParameters();
```

### Modifying Scene Structure

Adding/removing primitives or changing operations requires shader recompilation (~10-100ms):

```typescript
scene.add(newPrimitive).union();
gradientSampler.rebuildIfNeeded(); // Detects structure change and rebuilds
```

### How It Works

1. **Scene Graph**: Primitives and operations form a tree structure
2. **Code Generation**: Tree is compiled to WGSL shader code
3. **Parameter Buffer**: All positions, sizes, etc. are in a uniform buffer
4. **Shader Compilation**: Only when structure changes, parameters update every frame
5. **Performance**: Near-zero overhead vs hardcoded shaders

### Point Distribution

The `PointManager` automatically:
- Calculates point count based on primitives (30k base × √surface_area per primitive)
- Distributes points on AABB surfaces (6 faces per box) proportionally by primitive surface area
- Initializes points on bounding box surfaces for fast convergence to actual SDF surfaces
- Clamps total point count between 10k-200k for performance

This ensures accurate surface representation with efficient point usage. Points start on the AABB surface and converge inward/outward to the actual SDF surface via gradient descent.

See `src/sdf/` directory for implementation details.

## Interactive Controls

- Mouse drag to orbit camera
- Scroll to zoom in/out
