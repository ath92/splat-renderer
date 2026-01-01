# 3D SDF Splat System - Implementation Plan

## Overview
Create a 3D SDF system where points ("splats") are projected instantly onto 3D implicit surfaces and rendered with an orbit camera. This will be a **separate implementation** alongside the existing 2D version.

## Design Decisions
- ✅ **Rendering**: Point sprites (billboarded quads)
- ✅ **Camera**: Orbit camera with mouse controls
- ✅ **Primitives**: Sphere + Box only (initially)
- ✅ **Motion**: Instant projection to surface (no gradual movement)
- ✅ **Lighting**: Distance-based coloring only (no lighting)
- ✅ **Code**: Keep 2D version separate, create new 3D files
- ✅ **Dependencies**: Use gl-matrix library for matrix math

---

## 1. 3D SDF Primitives & Gradients

### Initial Primitives (Phase 1)
Based on Inigo Quilez's 3D distance functions from https://iquilezles.org/articles/distgradfunctions/

**Sphere:**
```wgsl
fn sdgSphere(p: vec3f, radius: f32) -> vec4f {
  let d = length(p);
  return vec4f(d - radius, p / d);
}
// Returns: vec4(distance, gradient.x, gradient.y, gradient.z)
```

**Box:**
```wgsl
fn sdgBox(p: vec3f, b: vec3f) -> vec4f {
  let w = abs(p) - b;
  let g = max(w, vec3f(0.0));
  let s = sign(p);
  // ... (fetch full implementation from IQ's article)
  return vec4f(distance, gradient);
}
```

### Operations
**Union (min):**
```wgsl
fn opUnion(d1: vec4f, d2: vec4f) -> vec4f {
  return select(d2, d1, d1.x < d2.x); // Pick closest
}
```

**Smooth Union:**
```wgsl
fn opSmoothUnion(d1: vec4f, d2: vec4f, k: f32) -> vec4f {
  let h = clamp(0.5 + 0.5 * (d2.x - d1.x) / k, 0.0, 1.0);
  let d = mix(d2.x, d1.x, h) - k * h * (1.0 - h);
  let grad = mix(d2.yzw, d1.yzw, h);
  return vec4f(d, normalize(grad));
}
```

---

## 2. 3D Point System

### Point Data Structure
```wgsl
struct PointData {
  positions: array<vec3f>, // 12 bytes per point
}
```

No velocity/momentum needed since we're doing instant projection.

### Initialization
Spawn points randomly in 3D bounding volume:
```typescript
for (let i = 0; i < numPoints; i++) {
  positions[i * 3 + 0] = Math.random() * 2 - 1; // x in [-1, 1]
  positions[i * 3 + 1] = Math.random() * 2 - 1; // y in [-1, 1]
  positions[i * 3 + 2] = Math.random() * 2 - 1; // z in [-1, 1]
}
```

### Surface Projection (Instant)
No gradual descent - project directly to surface:
```wgsl
fn updatePosition(pos: vec3f, gradient: vec3f, distance: f32) -> vec3f {
  let gradLen = length(gradient);
  if (gradLen > 0.0001) {
    let normalizedGrad = gradient / gradLen;
    return pos - normalizedGrad * distance; // Instant projection
  }
  return pos;
}
```

---

## 3. Camera System (Orbit)

### Camera Class
Using **gl-matrix** library for matrix operations.

**Parameters:**
```typescript
class Camera {
  target: vec3;        // Look-at point (usually origin)
  distance: number;    // Distance from target
  azimuth: number;     // Horizontal rotation (radians)
  elevation: number;   // Vertical rotation (radians)
  fov: number;         // Field of view (degrees)
  aspect: number;      // Width / height
  near: number;        // Near clipping plane
  far: number;         // Far clipping plane
}
```

**Matrices:**
- `getViewMatrix()` - Computes camera position from spherical coords, creates lookAt matrix
- `getProjectionMatrix()` - Perspective projection
- `getViewProjectionMatrix()` - Combined VP matrix for shaders

### Camera Controller

**Mouse Controls:**
- **Left drag**: Rotate (update azimuth/elevation)
- **Right drag** or **Middle drag**: Pan (move target)
- **Scroll wheel**: Zoom (adjust distance)

**Implementation:**
```typescript
class OrbitCameraController {
  private camera: Camera;
  private isDragging: boolean = false;
  private lastMouseX: number = 0;
  private lastMouseY: number = 0;

  onMouseDown(event: MouseEvent) { /* ... */ }
  onMouseMove(event: MouseEvent) {
    if (isDragging) {
      const dx = event.clientX - lastMouseX;
      const dy = event.clientY - lastMouseY;

      camera.azimuth += dx * 0.01;
      camera.elevation = clamp(camera.elevation + dy * 0.01, -Math.PI/2, Math.PI/2);
    }
  }
  onWheel(event: WheelEvent) {
    camera.distance *= (1 + event.deltaY * 0.001);
  }
}
```

---

## 4. Point Sprite Rendering

### Approach
Each point rendered as a billboarded quad that always faces the camera.

**Vertex Shader:**
```wgsl
@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  // Get 3D position from storage buffer
  let worldPos = positions[instanceIndex];

  // Project to clip space
  let clipPos = uniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);

  // Create billboard quad in screen space
  let quadOffset = array<vec2f, 6>(
    vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
    vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1)
  );
  let pointSize = 0.01; // World-space size
  let screenOffset = quadOffset[vertexIndex] * pointSize;

  output.position = vec4f(
    clipPos.xy + screenOffset * clipPos.w,
    clipPos.z,
    clipPos.w
  );

  // Color based on distance
  let distance = gradients[instanceIndex].x;
  output.color = getDistanceColor(distance);

  return output;
}
```

**Fragment Shader:**
```wgsl
@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Draw smooth circle within quad
  let dist = length(input.uv); // uv in [-1, 1]
  let alpha = 1.0 - smoothstep(0.8, 1.0, dist);

  return vec4f(input.color, alpha);
}
```

---

## 5. 3D Scene Example

```wgsl
fn sceneSDF(p: vec3f) -> vec4f {
  // Sphere at origin
  let sphere = sdgSphere(p - vec3f(0.0, 0.3, 0.0), 0.4);

  // Box offset
  let box = sdgBox(p - vec3f(0.5, -0.2, 0.0), vec3f(0.25, 0.25, 0.25));

  // Combine with smooth union
  return opSmoothUnion(sphere, box, 0.15);
}
```

---

## 6. Code Architecture

### New Files (3D-specific)

**Entry Point:**
- `src/main-3d.ts` - 3D version entry point

**Shaders:**
- `src/shaders-3d/compute-gradient-3d.wgsl` - 3D SDF gradient evaluation
- `src/shaders-3d/update-positions-3d.wgsl` - 3D instant projection
- `src/shaders-3d/render-splats-3d.wgsl` - Point sprite rendering

**TypeScript Classes:**
- `src/3d/Camera.ts` - Camera with view/projection matrices (using gl-matrix)
- `src/3d/OrbitCameraController.ts` - Mouse/keyboard input handling
- `src/3d/PointManager3D.ts` - Manages vec3 position buffers
- `src/3d/GradientSampler3D.ts` - 3D gradient compute pipeline
- `src/3d/PositionUpdater3D.ts` - 3D position update pipeline
- `src/3d/Renderer3D.ts` - 3D point sprite rendering

### Existing Files (Keep Separate)
- All 2D files remain unchanged
- User can switch between 2D and 3D by changing entry point

### Data Flow (Per Frame)
```
Input:
├─ Camera controller updates from mouse/keyboard
├─ Camera computes view-projection matrix
└─ Upload VP matrix + camera pos to GPU uniform buffer

GPU Pipeline:
├─ Compute Pass 1: Evaluate 3D SDF gradients at point positions
│   └─ Output: gradient buffer (vec4: distance + 3D gradient)
├─ Compute Pass 2: Project points to surface instantly
│   └─ Input: positions + gradients
│   └─ Output: updated positions (on surface)
├─ Swap ping-pong buffers
└─ Render Pass: Project 3D points to screen, render as sprites
    └─ Input: positions + gradients + VP matrix
    └─ Output: Billboarded point sprites with distance-based colors

Output:
└─ 3D visualization of splats on SDF surface
```

---

## 7. Implementation Plan

### Phase 1: Basic 3D Foundation
**Goal:** Render 3D points projected onto a sphere.

1. **Setup Dependencies**
   - Install gl-matrix: `npm install gl-matrix`
   - Install types: `npm install --save-dev @types/gl-matrix`

2. **Create Camera System**
   - `Camera.ts` with view/projection matrix generation
   - `OrbitCameraController.ts` with mouse controls
   - Test: Orbit around a fixed point

3. **3D SDF Primitives**
   - Fetch sphere + box gradient functions from IQ's article
   - Implement in `compute-gradient-3d.wgsl`
   - Create simple scene with one sphere

4. **3D Point System**
   - `PointManager3D.ts` - vec3 positions, ping-pong buffers
   - Random initialization in [-1, 1]³
   - Test: Points stored correctly on GPU

5. **Compute Pipeline**
   - `GradientSampler3D.ts` - Evaluate 3D SDF at point positions
   - `PositionUpdater3D.ts` - Instant projection to surface
   - Test: Points snap to sphere surface

6. **Basic Rendering**
   - `Renderer3D.ts` - Simple point rendering (no sprites yet)
   - Just render as GL_POINTS with distance-based color
   - Test: See colored points in 3D space

**Deliverable:** Colored dots on a sphere, orbitcamera works.

---

### Phase 2: Point Sprites & Polish
**Goal:** Nice-looking billboarded splats.

1. **Point Sprite Shader**
   - Billboarded quads in vertex shader
   - Smooth circular discs in fragment shader
   - Size attenuation with distance from camera

2. **Camera Improvements**
   - Smooth damping on rotation
   - Pan controls (right-click drag)
   - Zoom limits (min/max distance)

3. **Visual Polish**
   - Better color scheme for distance visualization
   - Depth testing for proper occlusion
   - Anti-aliased edges on sprites

**Deliverable:** Beautiful 3D point cloud on sphere.

---

### Phase 3: Box Primitive & CSG
**Goal:** Multiple primitives combined.

1. **Box SDF with Gradients**
   - Implement sdgBox from IQ's article
   - Test: Points on cube surface

2. **Union Operation**
   - Simple min-based union
   - Smooth union with blended gradients

3. **Scene Composition**
   - Multiple sphere + box combinations
   - Test different arrangements

**Deliverable:** Complex shapes from CSG operations.

---

### Phase 4: Future Enhancements (Optional)
- More primitives (torus, cylinder, capsule)
- Transformations (rotation matrices)
- Interactive editing (GUI to move primitives)
- Export point cloud (PLY format)
- Gaussian splatting upgrade
- Basic lighting (use gradients as normals)

---

## 8. Technical Specifications

### Uniform Buffer Layout (3D)
```wgsl
struct Uniforms {
  viewProjectionMatrix: mat4x4f,  // 64 bytes
  cameraPosition: vec3f,           // 12 bytes
  time: f32,                       // 4 bytes
  // Total: 80 bytes (already aligned)
}
```

### Buffer Sizes (1M points)
- Position buffers (ping-pong): 2 × 1M × 12 bytes = **24 MB**
- Gradient buffer: 1M × 16 bytes = **16 MB**
- **Total: ~40 MB** (very manageable)

### Performance Targets
- **1M points** at 60 FPS (should be achievable with indirect rendering)
- **10M points** at 30+ FPS (stretch goal)

---

## 9. Key Challenges & Solutions

### Challenge 1: Billboard Quad Generation
**Problem:** Creating screen-facing quads in vertex shader.

**Solution:**
```wgsl
// Extract camera right/up vectors from view matrix (inverse)
let right = viewMatrixInverse[0].xyz;
let up = viewMatrixInverse[1].xyz;

// Expand point to quad
let quadPos = worldPos + (right * offset.x + up * offset.y) * size;
```

### Challenge 2: Gradient Rotation (Future)
**Problem:** When rotating primitives, must also rotate gradients.

**Solution:**
```wgsl
let rotatedP = rotationMatrix * p;
let result = sdgPrimitive(rotatedP);
result.yzw = rotationMatrix * result.yzw; // Rotate gradient back to world space
```

### Challenge 3: Depth Testing
**Problem:** Overlapping transparent sprites.

**Solution:**
- Use depth testing with `depthWrite: true`
- Render opaque sprites (alpha = 1.0) or use additive blending
- For true transparency, would need depth sorting (deferred to Phase 4)

---

## 10. Dependencies

### Required
- **gl-matrix**: Matrix and vector math
  ```bash
  npm install gl-matrix
  npm install --save-dev @types/gl-matrix
  ```

### Project Structure
```
src/
├── 2d/                      # Keep existing 2D code
│   ├── main.ts
│   ├── shader.wgsl
│   └── ...
├── 3d/                      # New 3D code
│   ├── main-3d.ts
│   ├── Camera.ts
│   ├── OrbitCameraController.ts
│   ├── PointManager3D.ts
│   ├── GradientSampler3D.ts
│   ├── PositionUpdater3D.ts
│   ├── Renderer3D.ts
│   └── shaders/
│       ├── compute-gradient-3d.wgsl
│       ├── update-positions-3d.wgsl
│       └── render-splats-3d.wgsl
└── shared/                  # Shared utilities (if needed)
```

### HTML Entry Points
```html
<!-- 2D version -->
<script type="module" src="/src/2d/main.ts"></script>

<!-- 3D version -->
<script type="module" src="/src/3d/main-3d.ts"></script>
```

User can switch between them easily.

---

## Next Steps

Ready to implement **Phase 1: Basic 3D Foundation**:

1. ✅ Install gl-matrix dependency
2. ✅ Create Camera.ts class
3. ✅ Create OrbitCameraController.ts
4. ✅ Fetch sphere + box SDF gradient functions from IQ
5. ✅ Implement compute-gradient-3d.wgsl
6. ✅ Create PointManager3D.ts
7. ✅ Create GradientSampler3D.ts
8. ✅ Create PositionUpdater3D.ts
9. ✅ Create basic Renderer3D.ts
10. ✅ Wire everything together in main-3d.ts

**Ready to proceed?**
