// Vertex shader outputs
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

// Uniforms
struct Uniforms {
  resolution: vec2f,
  time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Vertex shader - creates a full screen quad
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;

  // Full screen triangle
  let x = f32((vertexIndex & 1u) << 2u);
  let y = f32((vertexIndex & 2u) << 1u);

  output.position = vec4f(x - 1.0, 1.0 - y, 0.0, 1.0);
  output.uv = vec2f(x * 0.5, y * 0.5);

  return output;
}

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

// Fragment shader
@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Convert to normalized device coordinates centered at origin
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  var uv = (input.uv * 2.0 - 1.0) * vec2f(aspect, 1.0);

  // Get distance and gradient from SDF
  let result = sceneSDF(uv);
  let d = result.x;
  let gradient = result.yz;

  // Render gradient as color
  // Map gradient from [-1, 1] to [0, 1] for RGB display
  let gradientColor = gradient * 0.5 + 0.5;

  var color: vec3f;
  if (d <= 0.0) {
    // Inside shapes: show gradient as color
    color = vec3f(gradientColor, 0.0);
  } else {
    // Outside shapes: show gradient as color with distance influence
    color = vec3f(gradientColor, clamp(d * 0.5, 0.0, 1.0));
  }

  return vec4f(color, 1.0);
}
