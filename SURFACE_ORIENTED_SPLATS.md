# Phase 1: Surface-Oriented Ellipsoidal Splats

## Overview

Transform the current screen-space billboards into surface-oriented ellipsoidal splats that align with the SDF surface. This should significantly reduce gaps by making splats stretch along the surface rather than face the camera.

## Current Behavior (Renderer.ts:76-88)

- Quads are billboards created in **screen space** after projection
- All quads face the camera regardless of surface orientation
- Fixed circular shape leads to gaps on surfaces viewed at oblique angles

```wgsl
// Current: screen-space billboard
let clipPos = uniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);
let screenOffset = quadOffset[vertexIndex] * pointSize;
output.position = vec4f(
  clipPos.xy + screenOffset * clipPos.w,
  clipPos.z,
  clipPos.w
);
```

## Target Behavior

- Quads generated in **world space** before projection
- Each quad oriented tangent to the SDF surface using the gradient normal
- Ellipsoidal shape: wide along the surface, thin perpendicular to it
- Overlapping splats fill gaps while maintaining depth sorting

## Mathematical Approach

### 1. Extract Surface Normal
Already available from gradient buffer (Renderer.ts:94-100):
```wgsl
let gradientData = gradients.results[instanceIndex];
let gradient = gradientData.yzw;
let normal = normalize(gradient);
```

### 2. Construct Tangent Frame
Build an orthonormal basis (tangent, bitangent, normal) at each point:

```wgsl
// Create an arbitrary tangent vector perpendicular to normal
fn computeTangent(normal: vec3f) -> vec3f {
  // Pick an axis least aligned with normal
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
  return normalize(cross(up, normal));
}

let tangent = computeTangent(normal);
let bitangent = cross(normal, tangent);
```

This gives us a coordinate frame aligned with the surface.

### 3. Generate Ellipsoidal Quad in World Space

```wgsl
// Quad corners in 2D
let quadOffset = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
);

// Size parameters (world space)
let tangentScale = 0.02;   // Wide along surface
let bitangentScale = 0.02; // Wide along surface
let normalScale = 0.005;   // Thin perpendicular to surface

let offset2D = quadOffset[vertexIndex];

// Build 3D offset using tangent frame
let worldOffset =
  tangent * offset2D.x * tangentScale +
  bitangent * offset2D.y * bitangentScale +
  normal * 0.0; // Could add slight normal offset for thickness

let finalWorldPos = worldPos + worldOffset;

// Now project to clip space
output.position = uniforms.viewProjectionMatrix * vec4f(finalWorldPos, 1.0);
```

### 4. Overlap for Gap Filling

Increase `tangentScale` and `bitangentScale` slightly (e.g., 1.2-1.5x theoretical coverage) so adjacent splats overlap. This ensures no gaps even with slight irregularities in point distribution.

## Implementation Steps

### Step 1: Add Tangent Helper Function
Add to vertex shader in `Renderer.ts` (around line 63):
```wgsl
fn computeTangent(normal: vec3f) -> vec3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
  return normalize(cross(up, normal));
}
```

### Step 2: Modify Vertex Shader
Replace the billboard generation (lines 70-88) with world-space ellipsoid generation:

1. Extract normal from gradient (already done at line 94-100, move earlier)
2. Compute tangent and bitangent
3. Build world offset from quad corners
4. Add offset to world position
5. Project to clip space

### Step 3: Adjust Sizing Parameters
Add tunable size parameters (could eventually become uniforms):
- `tangentScale`: Controls surface coverage width
- `bitangentScale`: Controls surface coverage height
- `normalScale`: Controls splat thickness (0 for flat, small value for volume)

Start with:
```wgsl
let tangentScale = 0.025;    // ~2.5cm in world space
let bitangentScale = 0.025;
let normalScale = 0.0;        // Flat for now
```

### Step 4: Fragment Shader (Minimal Changes)
The existing circular falloff (lines 111-117) should work fine, but could be modified to:
- Use elliptical falloff matching the tangent/bitangent ratio
- Add slight depth offset based on UV to create curved surface appearance

Optional enhancement:
```wgsl
// Elliptical falloff instead of circular
let dist = length(input.uv / vec2f(1.0, 1.0)); // Could use aspect ratio here
```

### Step 5: Test & Tune
- Verify splats orient correctly by rotating camera
- Adjust scale parameters to balance coverage vs performance
- Check depth sorting works correctly
- Look for remaining gaps and increase overlap as needed

## Expected Results

**Before:** Points appear as camera-facing circles, gaps visible when surface is viewed at oblique angles

**After:** Points form oriented ellipsoids that "lie flat" on the surface, significantly reducing gaps through better coverage

## Tuning Parameters

| Parameter | Purpose | Starting Value | Notes |
|-----------|---------|----------------|-------|
| `tangentScale` | Surface width coverage | 0.025 | Increase if gaps remain |
| `bitangentScale` | Surface height coverage | 0.025 | Usually same as tangent |
| `normalScale` | Splat thickness | 0.0 | Keep at 0 for now |
| Overlap factor | How much splats overlap | 1.0-1.5x | Implicit in scale values |

## Performance Considerations

- **Same vertex count:** Still 6 vertices × numPoints instances
- **Added per-vertex work:** Cross products, normalization (minimal cost)
- **Benefit:** Better coverage may allow fewer total points for same quality

## Next Steps (Future Phases)

If gaps still exist after this phase:
- **Phase 2:** Screen-space smoothing/dilation pass
- **Phase 3:** Adaptive sizing based on curvature or point density
- **Alternative:** Point relaxation to improve distribution

## Code Files to Modify

- `src/Renderer.ts` - vertex shader function (lines 63-107)
  - Add `computeTangent()` helper
  - Replace billboard logic with oriented ellipsoid generation
  - Optionally adjust fragment shader for elliptical falloff
