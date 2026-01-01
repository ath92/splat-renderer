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

// SDF for a circle
fn sdCircle(p: vec2f, radius: f32) -> f32 {
  return length(p) - radius;
}

// SDF for a rectangle (box)
fn sdBox(p: vec2f, size: vec2f) -> f32 {
  let d = abs(p) - size;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

// Smooth minimum for blending SDFs
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * h * k * (1.0 / 6.0);
}

// Main SDF scene
fn sceneSDF(p: vec2f) -> f32 {
  // Static circle
  let circle1 = sdCircle(p - vec2f(0.0, 0.1), 0.15);

  // Static rectangle
  let box1 = sdBox(p - vec2f(-0.2, -0.2), vec2f(0.2, 0.15));

  // Another circle
  let circle2 = sdCircle(p - vec2f(0.3, -0.3), 0.1);

  // Static rectangle
  let box2 = sdBox(p - vec2f(0.0, 0.3), vec2f(0.12, 0.12));

  // Combine all shapes using smooth minimum
  var d = smin(circle1, box1, 0.1);
  d = smin(d, circle2, 0.1);
  d = smin(d, box2, 0.1);

  return d;
}

// Fragment shader
@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Convert to normalized device coordinates centered at origin
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  var uv = (input.uv * 2.0 - 1.0) * vec2f(aspect, 1.0);

  // Get distance from SDF
  let d = sceneSDF(uv);

  // Base color depending on inside/outside
  var color: vec3f;
  if (d <= 0.0) {
    color = vec3f(0.2, 0.3, 0.4); // Dark blue-gray for inside shapes
  } else {
    color = vec3f(0.9, 0.9, 0.95); // Light background for outside
  }

  // Add isolines outside the primitives
  if (d > 0.0) {
    let bands = fract(d * 10.0);
    let lineWidth = 0.1;
    let lineIntensity = smoothstep(0.5 - lineWidth, 0.5, bands) *
                        smoothstep(0.5 + lineWidth, 0.5, bands);
    color = mix(color, color * 0.85, lineIntensity * 0.4);
  }

  // Strong white line at d=0 (surface boundary)
  let edgeWidth = 0.01;
  let edge = smoothstep(edgeWidth, 0.0, abs(d));
  color = mix(color, vec3f(1.0, 1.0, 1.0), edge);

  return vec4f(color, 1.0);
}
