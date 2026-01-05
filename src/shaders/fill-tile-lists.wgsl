/**
 * Fill tile index lists using computed offsets
 * Pass 3 of GPU tile binning (after prefix sum)
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

struct TileOffsets {
  offsets: array<u32>,
}

struct TileCurrentOffsets {
  offsets: array<atomic<u32>>,
}

struct TileIndices {
  indices: array<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
@group(0) @binding(2) var<storage, read> sortedIndices: SortedIndices;
@group(0) @binding(3) var<storage, read> tileOffsets: TileOffsets;
@group(0) @binding(4) var<storage, read_write> tileCurrentOffsets: TileCurrentOffsets;
@group(0) @binding(5) var<storage, read_write> tileIndices: TileIndices;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let sortedIdx = globalId.x;
  if (sortedIdx >= uniforms.numSplats) {
    return;
  }

  // Get the actual splat index from sorted order
  let splatIdx = sortedIndices.indices[sortedIdx];
  let projected = projectedSplats.splats[splatIdx];

  // Determine which tiles this splat overlaps (same logic as count pass)
  let minTileX = u32(max(0.0, floor(projected.screenBoundsMin.x / f32(uniforms.tileSize))));
  let minTileY = u32(max(0.0, floor(projected.screenBoundsMin.y / f32(uniforms.tileSize))));
  let maxTileX = u32(min(f32(uniforms.numTilesX - 1u), floor(projected.screenBoundsMax.x / f32(uniforms.tileSize))));
  let maxTileY = u32(min(f32(uniforms.numTilesY - 1u), floor(projected.screenBoundsMax.y / f32(uniforms.tileSize))));

  // For each overlapping tile, atomically get next write position and write index
  for (var ty = minTileY; ty <= maxTileY; ty++) {
    for (var tx = minTileX; tx <= maxTileX; tx++) {
      let tileIdx = ty * uniforms.numTilesX + tx;

      // Atomically get and increment write position
      let writePos = atomicAdd(&tileCurrentOffsets.offsets[tileIdx], 1u);

      // Write splat index to tile list
      // Because we iterate in sorted order, indices are automatically sorted within each tile
      tileIndices.indices[writePos] = splatIdx;
    }
  }
}
