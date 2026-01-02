// Uniforms
struct Uniforms {
  viewProjectionMatrix: mat4x4f,
  cameraPosition: vec3f,
  time: f32,
}

// Point data
struct PointData {
  positions: array<vec3f>,
}

struct GradientData {
  results: array<vec4f>, // (distance, gradient.x, gradient.y, gradient.z)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: PointData;
@group(0) @binding(2) var<storage, read_write> gradients: GradientData;

// SDF with gradient for a sphere
// Returns vec4(distance, gradient.x, gradient.y, gradient.z)
fn sdgSphere(p: vec3f, radius: f32) -> vec4f {
  let l = length(p);
  return vec4f(l - radius, p / l);
}

// SDF with gradient for a box
// Returns vec4(distance, gradient.x, gradient.y, gradient.z)
fn sdgBox(p: vec3f, b: vec3f, r: f32) -> vec4f {
  let w = abs(p) - (b - r);
  let g = max(w.x, max(w.y, w.z));
  let q = max(w, vec3f(0.0));
  let l = length(q);

  var f: vec4f;
  if (g > 0.0) {
    f = vec4f(l, q / l);
  } else {
    f = vec4f(
      g,
      select(0.0, 1.0, w.x == g),
      select(0.0, 1.0, w.y == g),
      select(0.0, 1.0, w.z == g)
    );
  }

  return vec4f(f.x - r, f.yzw * sign(p));
}

// Union operation (pick closest)
fn opUnion(d1: vec4f, d2: vec4f) -> vec4f {
  return select(d2, d1, d1.x < d2.x);
}

// Smooth union operation
fn opSmoothUnion(d1: vec4f, d2: vec4f, k: f32) -> vec4f {
  let h = clamp(0.5 + 0.5 * (d2.x - d1.x) / k, 0.0, 1.0);
  let d = mix(d2.x, d1.x, h) - k * h * (1.0 - h);
  let grad = normalize(mix(d2.yzw, d1.yzw, h));
  return vec4f(d, grad);
}

// Main SDF scene with gradients
fn sceneSDF(p: vec3f) -> vec4f {
  // Sphere at origin
  let sphere = sdgSphere(p - vec3f(0.0, 0.3, 0.0), 0.4);

  // Box (using rounding parameter = 0.0 for sharp edges)
  let box = sdgBox(p - vec3f(0.5, -0.2, 0.0), vec3f(0.25, 0.25, 0.25), 0.0);

  // Combine with smooth union
  return opSmoothUnion(sphere, box, 0.15);
}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let numPoints = arrayLength(&points.positions);

  if (index >= numPoints) {
    return;
  }

  // Get point position in 3D space
  let pos = points.positions[index];

  // Evaluate SDF gradient at this position
  let result = sceneSDF(pos);

  // Store result (distance, gradient.x, gradient.y, gradient.z)
  gradients.results[index] = result;
}
