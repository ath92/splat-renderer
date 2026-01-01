// Uniforms
struct Uniforms {
  resolution: vec2f,
  time: f32,
}

// Point data
struct PointData {
  positions: array<vec2f>,
}

struct GradientData {
  results: array<vec4f>, // (distance, gradient.x, gradient.y, padding)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: PointData;
@group(0) @binding(2) var<storage, read_write> gradients: GradientData;

// SDF with gradient for a circle
// Returns vec3(distance, gradient.x, gradient.y)
fn sdgCircle(p: vec2f, radius: f32) -> vec3f {
  let d = length(p);
  return vec3f(d - radius, p / d);
}

// SDF with gradient for a box
// Returns vec3(distance, gradient.x, gradient.y)
fn sdgBox(p: vec2f, b: vec2f) -> vec3f {
  let w = abs(p) - b;
  let s = vec2f(
    select(1.0, -1.0, p.x < 0.0),
    select(1.0, -1.0, p.y < 0.0)
  );
  let g = max(w.x, w.y);
  let q = max(w, vec2f(0.0));
  let l = length(q);

  let dist = select(g, l, g > 0.0);

  var grad: vec2f;
  if (g > 0.0) {
    grad = s * (q / l);
  } else {
    grad = s * select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), w.x > w.y);
  }

  return vec3f(dist, grad);
}

// Main SDF scene with gradients
fn sceneSDF(p: vec2f) -> vec3f {
  // Static circle
  let circle1 = sdgCircle(p - vec2f(0.0, 0.1), 0.15);

  // Static rectangle
  let box1 = sdgBox(p - vec2f(-0.2, -0.2), vec2f(0.2, 0.15));

  // Another circle
  let circle2 = sdgCircle(p - vec2f(0.3, -0.3), 0.1);

  // Static rectangle
  let box2 = sdgBox(p - vec2f(0.0, 0.3), vec2f(0.12, 0.12));

  // For gradient functions, we need to combine them properly
  // For now, just return the closest one (this is a simplified approach)
  var result = circle1;

  if (box1.x < result.x) {
    result = box1;
  }
  if (circle2.x < result.x) {
    result = circle2;
  }
  if (box2.x < result.x) {
    result = box2;
  }

  return result;
}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let numPoints = arrayLength(&points.positions);

  if (index >= numPoints) {
    return;
  }

  // Get point position in normalized screen space
  let pos = points.positions[index];

  // Apply aspect ratio correction
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  let correctedPos = pos * vec2f(aspect, 1.0);

  // Evaluate SDF gradient at this position
  let result = sceneSDF(correctedPos);

  // Store result (distance, gradient.x, gradient.y, padding)
  gradients.results[index] = vec4f(result, 0.0);
}
