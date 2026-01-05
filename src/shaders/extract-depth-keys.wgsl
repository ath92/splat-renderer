/**
 * Extract depth keys from ProjectedSplat buffer for radix sort
 * Converts depth values to sortable uint32 keys
 */

struct Uniforms {
  numSplats: u32,
  paddedSize: u32,
}

struct ProjectedSplat {
  screenBoundsMin: vec2f,
  screenBoundsMax: vec2f,
  depth: f32,
  screenRadius: f32,
  originalIndex: u32,
  _padding: f32,
}

struct ProjectedSplats {
  splats: array<ProjectedSplat>,
}

struct Keys {
  values: array<u32>,
}

struct Payload {
  values: array<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
@group(0) @binding(2) var<storage, read_write> keys: Keys;
@group(0) @binding(3) var<storage, read_write> payload: Payload;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;

  // Pad with maximum values to sort to end
  if (index >= uniforms.paddedSize) {
    return;
  }

  if (index >= uniforms.numSplats) {
    keys.values[index] = 0xffffffffu;
    payload.values[index] = 0xffffffffu;
    return;
  }

  let projected = projectedSplats.splats[index];
  let depth = projected.depth;

  // Convert float to sortable uint32
  // IEEE 754: sign bit flip for negatives, all bits flip for positives
  let floatBits = bitcast<u32>(depth);
  let mask = select(0x80000000u, 0xffffffffu, (floatBits >> 31u) == 1u);
  keys.values[index] = floatBits ^ mask;

  // Payload stores original index
  payload.values[index] = index;
}
