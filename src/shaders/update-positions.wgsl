// Uniforms
struct Uniforms {
  viewProjectionMatrix: mat4x4f,
  cameraPosition: vec3f,
  time: f32,
}

// Point data
struct PositionData {
  positions: array<vec3f>,
}

struct GradientData {
  results: array<vec4f>, // (distance, gradient.x, gradient.y, gradient.z)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> currentPositions: PositionData;
@group(0) @binding(2) var<storage, read> gradients: GradientData;
@group(0) @binding(3) var<storage, read_write> nextPositions: PositionData;

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let numPoints = arrayLength(&currentPositions.positions);

  if (index >= numPoints) {
    return;
  }

  // Get current position and gradient
  let pos = currentPositions.positions[index];
  let gradientResult = gradients.results[index];
  let distance = gradientResult.x;
  let gradient = gradientResult.yzw;

  // Use small gradient descent steps for natural distribution
  // Instead of instant projection, take small steps toward the surface
  const stepSize = 0.005; // Small step size for iterative convergence
  let gradLen = length(gradient);
  var newPos = pos;

  if (gradLen > 0.0001) {
    let normalizedGrad = gradient / gradLen;
    newPos = pos - normalizedGrad * distance * stepSize;
  }

  // Write updated position
  nextPositions.positions[index] = newPos;
}
