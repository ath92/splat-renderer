/**
 * Count how many splats overlap each tile
 * Pass 1 of GPU tile binning
 */

struct Uniforms {
  numSplats: u32,
  tileSize: u32,
  numTilesX: u32,
  numTilesY: u32,
  screenWidth: f32,
  screenHeight: f32,
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

struct SortedIndices {
  indices: array<u32>,
}

struct TileCounts {
  counts: array<atomic<u32>>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
@group(0) @binding(2) var<storage, read> sortedIndices: SortedIndices;
@group(0) @binding(3) var<storage, read_write> tileCounts: TileCounts;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let sortedIdx = globalId.x;
  if (sortedIdx >= uniforms.numSplats) {
    return;
  }

  // Get the actual splat index from sorted order
  let splatIdx = sortedIndices.indices[sortedIdx];
  let projected = projectedSplats.splats[splatIdx];

  // Determine which tiles this splat overlaps
  let minTileX = u32(max(0.0, floor(projected.screenBoundsMin.x / f32(uniforms.tileSize))));
  let minTileY = u32(max(0.0, floor(projected.screenBoundsMin.y / f32(uniforms.tileSize))));
  let maxTileX = u32(min(f32(uniforms.numTilesX - 1u), floor(projected.screenBoundsMax.x / f32(uniforms.tileSize))));
  let maxTileY = u32(min(f32(uniforms.numTilesY - 1u), floor(projected.screenBoundsMax.y / f32(uniforms.tileSize))));

  // Atomically increment count for each overlapping tile
  for (var ty = minTileY; ty <= maxTileY; ty++) {
    for (var tx = minTileX; tx <= maxTileX; tx++) {
      let tileIdx = ty * uniforms.numTilesX + tx;
      atomicAdd(&tileCounts.counts[tileIdx], 1u);
    }
  }
}
