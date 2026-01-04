# Tile-Based Gaussian Splatting Rendering Plan

## The Problem

Our current approach uses **instanced rendering** with a single draw call. The GPU processes all splat instances in parallel, which means:
- Fragments from different splats can write to the same pixel simultaneously
- Blending order is **undefined** (race condition)
- Sorting order doesn't guarantee correct alpha blending
- Result: visible artifacts, incorrect transparency

## The Solution: Tile-Based Rendering

Divide the screen into tiles (e.g., 16×16 pixels) and process each tile sequentially, ensuring proper back-to-front blending within each tile.

---

## How Tile-Based Rendering Works

### Step 1: Divide Screen into Tiles
```
Screen (1920×1080)
├─ Tile (0,0): pixels [0-15, 0-15]
├─ Tile (1,0): pixels [16-31, 0-15]
├─ Tile (0,1): pixels [0-15, 16-31]
└─ ... (120 × 67.5 = 8100 tiles for 16×16 tile size)
```

### Step 2: Assign Splats to Tiles
For each splat:
- Project 2D bounding box to screen space
- Determine which tiles the splat overlaps
- Add splat index to those tiles' lists

Example:
```
Splat #42 overlaps tiles: [(5,3), (6,3), (5,4), (6,4)]
→ Add splat #42 to lists for these 4 tiles
```

### Step 3: Sort Per-Tile Lists
For each tile:
- Sort its splat list by depth (back-to-front)
- This is a **smaller sort** than sorting all splats globally

### Step 4: Render Tile-by-Tile
For each tile:
- Process splats in sorted order
- Each splat contributes to pixels within the tile
- Sequential processing ensures correct blending order

---

## Implementation Architecture

### New Components

#### 1. **TileBinning** (Compute Shader)
```typescript
class TileBinner {
  // Input: splat positions, radii
  // Output: per-tile splat lists

  computeBinning(
    positions: GPUBuffer,
    radii: GPUBuffer,
    tiles: TileGrid
  ): void;
}
```

**Compute Shader Tasks:**
- Project each splat's 2D bounding box to screen space
- Determine tile overlap using bounding box
- Write splat indices to tile buffers using atomic operations

**Buffer Structure:**
```wgsl
struct TileData {
  splatIndices: array<u32>,  // Flat array of all splat indices
  tileOffsets: array<u32>,   // Start index for each tile
  tileCounts: array<u32>,    // Number of splats per tile
}
```

#### 2. **PerTileSorter** (Compute Shader)
```typescript
class PerTileSorter {
  // Sort each tile's splat list independently
  sortTiles(tileData: TileData, depths: GPUBuffer): void;
}
```

**Approach:**
- Each workgroup sorts one tile
- Use local bitonic sort or insertion sort (tiles have fewer splats)
- Parallel across tiles, sequential within each tile

#### 3. **TileRenderer** (Render Pipeline)
```typescript
class TileRenderer {
  // Render splats tile-by-tile
  render(
    tileData: TileData,
    splatProperties: GPUBuffer,
    camera: Camera
  ): void;
}
```

**Rendering Approach:**

**Option A: Per-Tile Dispatch** (Simpler)
```typescript
for (let tileY = 0; tileY < numTilesY; tileY++) {
  for (let tileX = 0; tileX < numTilesX; tileX++) {
    renderTile(tileX, tileY, tileData);
  }
}
```
- One draw call per tile
- CPU loop over tiles
- Slower but guarantees order

**Option B: Single Dispatch with Atomic Blending** (Faster)
- Single compute shader
- Each thread processes one splat-tile pair
- Use atomics or per-pixel linked lists for correct blending
- More complex but better performance

---

## Detailed Implementation Steps

### Phase 1: Screen Space Projection (Compute Shader)

**New Component: `SplatProjector`**

```wgsl
struct ProjectedSplat {
  screenBoundsMin: vec2f,  // Screen-space AABB min
  screenBoundsMax: vec2f,  // Screen-space AABB max
  depth: f32,              // Camera-space depth
  originalIndex: u32,      // Index into splat properties
}

@compute @workgroup_size(64)
fn projectSplats(...) {
  // For each splat:
  // 1. Transform 3D position to screen space
  // 2. Compute 2D bounding box (position ± radius in screen space)
  // 3. Store projected data
}
```

**Output:** Buffer of `ProjectedSplat` structs

---

### Phase 2: Tile Binning (Compute Shader)

**New Component: `TileBinner`**

```wgsl
struct TileList {
  startOffset: atomic<u32>,  // Where this tile's data starts
  count: atomic<u32>,        // Number of splats in this tile
}

@group(0) @binding(0) var<storage, read> projectedSplats: array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read_write> tileLists: array<TileList>;
@group(0) @binding(2) var<storage, read_write> splatIndices: array<u32>;

@compute @workgroup_size(64)
fn binToTiles(@builtin(global_invocation_id) globalId: vec3u) {
  let splatIdx = globalId.x;
  let splat = projectedSplats[splatIdx];

  // Calculate which tiles this splat overlaps
  let minTileX = u32(floor(splat.screenBoundsMin.x / TILE_SIZE));
  let maxTileX = u32(ceil(splat.screenBoundsMax.x / TILE_SIZE));
  let minTileY = u32(floor(splat.screenBoundsMin.y / TILE_SIZE));
  let maxTileY = u32(ceil(splat.screenBoundsMax.y / TILE_SIZE));

  // Add this splat to each overlapping tile
  for (var ty = minTileY; ty <= maxTileY; ty++) {
    for (var tx = minTileX; tx <= maxTileX; tx++) {
      let tileIdx = ty * numTilesX + tx;

      // Atomically allocate space in the flat indices array
      let insertPos = atomicAdd(&tileLists[tileIdx].count, 1u);

      // Write splat index
      splatIndices[tileLists[tileIdx].startOffset + insertPos] = splatIdx;
    }
  }
}
```

**Challenge:** Need to pre-allocate space for `splatIndices` buffer
- Conservative estimate: `numSplats * averageOverlapCount`
- Or use 2-pass approach: count first, allocate second

---

### Phase 3: Per-Tile Sorting (Compute Shader)

**New Component: `PerTileSorter`**

```wgsl
@compute @workgroup_size(256)  // One workgroup per tile
fn sortTile(@builtin(workgroup_id) workgroupId: vec3u) {
  let tileIdx = workgroupId.x;
  let tileList = tileLists[tileIdx];

  // Load splat indices for this tile into shared memory
  var shared_indices: array<u32, MAX_SPLATS_PER_TILE>;
  var shared_depths: array<f32, MAX_SPLATS_PER_TILE>;

  // Load data
  let count = min(tileList.count, MAX_SPLATS_PER_TILE);
  for (var i = localId; i < count; i += 256) {
    let splatIdx = splatIndices[tileList.startOffset + i];
    shared_indices[i] = splatIdx;
    shared_depths[i] = projectedSplats[splatIdx].depth;
  }

  workgroupBarrier();

  // Bitonic sort in shared memory (back-to-front)
  // ... (standard bitonic sort with workgroupBarrier between stages)

  // Write sorted indices back
  for (var i = localId; i < count; i += 256) {
    splatIndices[tileList.startOffset + i] = shared_indices[i];
  }
}
```

---

### Phase 4: Tile-Based Rendering

**Option A: CPU Loop (Simpler to implement)**

```typescript
for (let tileY = 0; tileY < numTilesY; tileY++) {
  for (let tileX = 0; tileX < numTilesX; tileX++) {
    const tileIdx = tileY * numTilesX + tileX;

    // Set viewport to this tile
    renderPass.setViewport(
      tileX * TILE_SIZE, tileY * TILE_SIZE,
      TILE_SIZE, TILE_SIZE,
      0, 1
    );

    // Render splats for this tile in sorted order
    renderTileSplats(tileIdx);
  }
}
```

**Option B: Compute Shader Blending (Faster)**

```wgsl
// Compute shader that manually blends splats to an output texture
@compute @workgroup_size(16, 16)  // One thread per pixel in tile
fn renderTile(
  @builtin(workgroup_id) tileId: vec3u,
  @builtin(local_invocation_id) pixelInTile: vec3u
) {
  let tileIdx = tileId.y * numTilesX + tileId.x;
  let pixelX = tileId.x * 16u + pixelInTile.x;
  let pixelY = tileId.y * 16u + pixelInTile.y;

  var color = vec3f(0.0);  // Start with background
  var alpha = 0.0;

  // Process splats in this tile (back-to-front order)
  let tileList = tileLists[tileIdx];
  for (var i = 0u; i < tileList.count; i++) {
    let splatIdx = splatIndices[tileList.startOffset + i];

    // Evaluate Gaussian at this pixel
    let splatContribution = evaluateSplat(splatIdx, pixelX, pixelY);

    // Alpha blend (back-to-front)
    color = color * (1.0 - splatContribution.a) + splatContribution.rgb * splatContribution.a;
    alpha = alpha * (1.0 - splatContribution.a) + splatContribution.a;

    // Early exit if fully opaque
    if (alpha >= 0.99) { break; }
  }

  // Write final pixel color
  textureStore(outputTexture, vec2u(pixelX, pixelY), vec4f(color, alpha));
}
```

---

## Performance Considerations

### Buffer Sizes
```
Tile size: 16×16 pixels
Screen: 1920×1080
Number of tiles: 120 × 68 = 8,160 tiles

If average splat overlaps 4 tiles:
  Total tile-splat pairs: numSplats × 4

For 100K splats:
  splatIndices buffer: 400K entries × 4 bytes = 1.6 MB
```

### Optimizations
1. **Culling:** Don't bin splats outside screen bounds
2. **Max splats per tile:** Cap at reasonable limit (e.g., 512)
3. **Shared memory sorting:** Faster than global memory
4. **Early alpha termination:** Stop blending when pixel is opaque

---

## Alternative: Simpler Sequential Rendering

If tile-based is too complex initially, we could:

**Option: Multiple Draw Calls (One per Splat)**
```typescript
// Slow but guarantees order
for (let i = 0; i < numSplats; i++) {
  const splatIdx = sortedIndices[i];

  // Bind splat-specific data
  renderPass.setBindGroup(1, createSplatBindGroup(splatIdx));

  // Draw single splat
  renderPass.draw(6, 1, 0, 0);
}
```

**Pros:**
- Guaranteed sequential blending
- Simpler implementation
- No tile binning needed

**Cons:**
- Very slow (1000s of draw calls)
- High CPU overhead
- Not scalable

---

## Recommended Approach

### Phase 1 (Proof of Concept)
Start with **Option A: Per-Tile CPU Loop** for rendering:
1. Implement projection + tile binning
2. Per-tile sorting in compute shader
3. CPU loop over tiles with one draw call per tile
4. Validate correctness

### Phase 2 (Optimization)
Switch to **Option B: Compute Shader Blending**:
1. Single compute dispatch
2. Manual alpha blending in shader
3. Much faster than CPU loop

---

## Questions for Review

1. **Tile size:** 16×16 seems standard, but should we make it configurable?
2. **Buffer allocation:** Use conservative pre-allocation or 2-pass (count then allocate)?
3. **Start with CPU loop or jump to compute shader blending?**
4. **Max splats per tile:** Should we cap and handle overflow?
5. **Worth implementing, or explore alternative approaches first?**

---

## Estimated Complexity

**New Files:**
- `SplatProjector.ts` - Project splats to screen space
- `TileBinner.ts` - Bin splats to tiles
- `PerTileSorter.ts` - Sort within tiles
- `TileRenderer.ts` - Render tiles

**Changes to Existing:**
- `Renderer.ts` - Switch to tile-based rendering
- `main.ts` - Wire up new pipeline

**Estimated LOC:** ~800-1200 lines of new code

**Time Estimate:** 4-8 hours for Phase 1, 2-4 hours for Phase 2
