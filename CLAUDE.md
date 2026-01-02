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
   - Implements gradient descent: `newPos = pos - normalize(gradient) * distance`
   - Uses ping-pong buffers to avoid read/write conflicts

3. **Rendering** (Render Pipeline)
   - `Renderer` visualizes particles in 3D space
   - Renders particles as instanced quads colored by distance from surface
   - Uses camera matrices for 3D projection

### Key Components

**PointManager**
- Manages ping-pong GPU buffers for particle positions (3D coordinates)
- `getCurrentPositionBuffer()` / `getNextPositionBuffer()` / `swap()` pattern
- Initialized with random positions in 3D space

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

## Modifying SDFs

To add or modify shapes:

1. Add/modify SDF functions in the WGSL shader files
2. For gradient descent to work, implement the `sdg*` variant that returns distance and gradient
3. Update `sceneSDF()` to combine shapes using min/max/smin operations
4. Note: Gradient combination for smooth min (`smin`) requires special handling

## Interactive Controls

- Mouse drag to orbit camera
- Scroll to zoom in/out
