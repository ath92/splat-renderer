/**
 * Work-efficient parallel prefix sum (exclusive scan)
 * Based on Blelloch algorithm
 * Handles up to WORKGROUP_SIZE * 2 elements in a single workgroup
 */

const WORKGROUP_SIZE = 256u;

struct Uniforms {
  numElements: u32,
}

struct InputData {
  values: array<u32>,
}

struct OutputData {
  values: array<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: InputData;
@group(0) @binding(2) var<storage, read_write> output: OutputData;

// Shared memory for reduction tree
var<workgroup> temp: array<u32, WORKGROUP_SIZE * 2u>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(workgroup_id) workgroupId: vec3u
) {
  let tid = localId.x;
  let numElements = uniforms.numElements;

  // Each thread loads two elements
  let ai = tid;
  let bi = tid + WORKGROUP_SIZE;

  // Load input into shared memory with bounds checking
  if (ai < numElements) {
    temp[ai] = input.values[ai];
  } else {
    temp[ai] = 0u;
  }

  if (bi < numElements) {
    temp[bi] = input.values[bi];
  } else {
    temp[bi] = 0u;
  }

  workgroupBarrier();

  // Up-sweep (reduce) phase - build sum tree
  var offset = 1u;
  for (var d = WORKGROUP_SIZE; d > 0u; d >>= 1u) {
    if (tid < d) {
      let ai_up = offset * (2u * tid + 1u) - 1u;
      let bi_up = offset * (2u * tid + 2u) - 1u;
      temp[bi_up] += temp[ai_up];
    }
    offset <<= 1u;
    workgroupBarrier();
  }

  // Clear the last element (it contains the total sum, which we don't need for exclusive scan)
  if (tid == 0u) {
    temp[WORKGROUP_SIZE * 2u - 1u] = 0u;
  }

  workgroupBarrier();

  // Down-sweep phase - propagate sums down the tree
  for (var d = 1u; d <= WORKGROUP_SIZE; d <<= 1u) {
    offset >>= 1u;
    if (tid < d) {
      let ai_down = offset * (2u * tid + 1u) - 1u;
      let bi_down = offset * (2u * tid + 2u) - 1u;

      let t = temp[ai_down];
      temp[ai_down] = temp[bi_down];
      temp[bi_down] += t;
    }
    workgroupBarrier();
  }

  // Write results back to global memory with bounds checking
  if (ai < numElements) {
    output.values[ai] = temp[ai];
  }

  if (bi < numElements) {
    output.values[bi] = temp[bi];
  }
}
