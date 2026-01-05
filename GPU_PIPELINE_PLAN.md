# GPU-Only Sorting and Binning Pipeline Plan

## Overview

Eliminate all CPU readbacks from the rendering pipeline by moving depth extraction, sorting, and tile binning entirely to GPU compute shaders. This will reduce ~90-270MB/s of GPU→CPU bandwidth to zero.

## Current Pipeline (with readbacks)

```
┌─────────────┐
│  Project    │ GPU Compute
│  Splats     │
└──────┬──────┘
       │
       ├─── READBACK (ProjectedSplat buffer, 1-4MB) ───┐
       │                                               │
       v                                               v
┌─────────────┐                                 ┌─────────────┐
│  Extract    │ CPU                             │  Radix Sort │ GPU
│  Depths     │                                 │             │
└──────┬──────┘                                 └──────┬──────┘
       │                                               │
       └──────────────┬────────────────────────────────┘
                      │
                      ├─── READBACK (Sorted indices, 120-480KB) ───┐
                      │                                             │
                      v                                             v
               ┌─────────────┐                              ┌─────────────┐
               │  Tile       │ CPU (reads projected data)   │   Render    │ GPU
               │  Binning    │                              │             │
               └──────┬──────┘                              └─────────────┘
                      │
                      └──────────────────────────────────────┘
```

**Bottlenecks:**
- 3 major readbacks per frame
- CPU tile binning is synchronous (blocks frame)
- GPU sits idle during CPU processing

## Target Pipeline (GPU-only)

```
┌─────────────┐
│  Project    │ GPU Compute (existing)
│  Splats     │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Extract    │ GPU Compute (NEW)
│  Depth Keys │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Radix Sort │ GPU Compute (existing, modified)
│             │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Count      │ GPU Compute (NEW - Pass 1)
│  Tile Hits  │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Prefix Sum │ GPU Compute (NEW - Pass 2)
│  Scan       │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Fill       │ GPU Compute (NEW - Pass 3)
│  Tile Lists │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Render     │ GPU Compute (existing)
│             │
└─────────────┘
```

**Benefits:**
- Zero CPU↔GPU transfers
- Fully asynchronous GPU pipeline
- Better parallelism
- Scalable to millions of splats

---

## Detailed Component Design

### 1. Depth Key Extractor (NEW)

**Purpose:** Extract depth values from ProjectedSplat buffer and convert to sortable uint32 keys

**Input:**
- `ProjectedSplat` buffer (read-only storage)
  - Format: `{vec2f min, vec2f max, f32 depth, f32 screenRadius, u32 originalIndex, f32 _padding}`
  - Size: numSplats × 32 bytes

**Output:**
- `Keys` buffer (storage)
  - Format: `array<u32>` (sortable depth keys)
  - Size: paddedSplats × 4 bytes
- `Payload` buffer (storage)
  - Format: `array<u32>` (original indices)
  - Size: paddedSplats × 4 bytes

**Shader:** `src/shaders/extract-depth-keys.wgsl`

```wgsl
@workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  if (index >= numSplats) {
    // Pad with max values to sort to end
    keys[index] = 0xffffffffu;
    payload[index] = 0xffffffffu;
    return;
  }

  let projected = projectedSplats.splats[index];
  let depth = projected.depth;

  // Convert float to sortable uint (flip bits for IEEE 754)
  let floatBits = bitcast<u32>(depth);
  let mask = select(0x80000000u, 0xffffffffu, (floatBits >> 31u) == 1u);
  keys[index] = floatBits ^ mask;

  payload[index] = index;
}
```

**TypeScript Class:** `DepthKeyExtractor`
- Constructor: Creates pipeline, bind group layout
- `extract(encoder, projectedBuffer, keysBuffer, payloadBuffer)`
- Workgroup count: `ceil(paddedSplats / 256)`

**Performance Estimate:** <0.1ms for 120k splats

---

### 2. Radix Sort Modifications

**Current State:** Reads back ProjectedSplat buffer to extract depths

**Changes:**
- Remove CPU depth extraction logic
- Remove readback of ProjectedSplat buffer
- Keep sorted indices in GPU memory (don't read back)
- Accept pre-filled `keysBuffer` and `payloadBuffer` as input

**Modified Method Signature:**
```typescript
// Old:
async sort(encoder: GPUCommandEncoder, projectedBuffer: GPUBuffer): Promise<Uint32Array>

// New:
sort(encoder: GPUCommandEncoder, keysBuffer: GPUBuffer, payloadBuffer: GPUBuffer): void
```

**Returns:** Nothing (indices stay on GPU)

**Performance Gain:** Eliminates ~1-4MB readback + CPU processing time

---

### 3. GPU Tile Binner (NEW)

Three-pass algorithm for exact tile assignment:

#### Pass 1: Count Tile Hits

**Purpose:** Count how many splats overlap each tile

**Input:**
- `ProjectedSplat` buffer (read-only)
- `SortedIndices` buffer (read-only, output from radix sort payload)
- Uniforms: `numTilesX`, `numTilesY`, `tileSize`, `screenWidth`, `screenHeight`

**Output:**
- `TileCounts` buffer (storage, read-write)
  - Format: `array<atomic<u32>>`
  - Size: numTiles × 4 bytes
  - Initially zeroed

**Shader:** `src/shaders/count-tile-hits.wgsl`

```wgsl
@workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let sortedIdx = globalId.x;
  if (sortedIdx >= numSplats) { return; }

  let splatIdx = sortedIndices[sortedIdx];
  let projected = projectedSplats.splats[splatIdx];

  // Determine overlapping tiles
  let minTileX = u32(max(0.0, floor(projected.screenBoundsMin.x / f32(tileSize))));
  let minTileY = u32(max(0.0, floor(projected.screenBoundsMin.y / f32(tileSize))));
  let maxTileX = u32(min(f32(numTilesX - 1), floor(projected.screenBoundsMax.x / f32(tileSize))));
  let maxTileY = u32(min(f32(numTilesY - 1), floor(projected.screenBoundsMax.y / f32(tileSize))));

  // Atomically increment count for each overlapping tile
  for (var ty = minTileY; ty <= maxTileY; ty++) {
    for (var tx = minTileX; tx <= maxTileX; tx++) {
      let tileIdx = ty * numTilesX + tx;
      atomicAdd(&tileCounts.counts[tileIdx], 1u);
    }
  }
}
```

**Performance Estimate:** ~0.5-1ms for 120k splats

**Challenge:** Atomic contention on popular tiles
**Solution:** Accept it - modern GPUs handle atomics well, and tiles are typically hit by 10-100 splats, not thousands

---

#### Pass 2: Prefix Sum (Exclusive Scan)

**Purpose:** Convert tile counts to tile offsets using parallel prefix sum

**Algorithm:** Work-Efficient Parallel Scan (Blelloch, 1990)
- Up-sweep phase: Build reduction tree
- Down-sweep phase: Propagate sums

**Input:**
- `TileCounts` buffer (read-only)

**Output:**
- `TileOffsets` buffer (storage)
  - Format: `array<u32>`
  - Size: numTiles × 4 bytes
  - `TileOffsets[i]` = sum of all counts before tile i

**Shader:** `src/shaders/prefix-sum.wgsl`

**Implementation Strategy:**
We'll use a **single-workgroup scan** for simplicity (works up to ~32k tiles):

```wgsl
// Shared memory for reduction tree
var<workgroup> temp: array<u32, WORKGROUP_SIZE * 2>;

@workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(workgroup_id) workgroupId: vec3u
) {
  let tid = localId.x;
  let offset = 1u;

  // Load input into shared memory
  let ai = tid;
  let bi = tid + WORKGROUP_SIZE;
  temp[ai] = select(0u, tileCounts.counts[ai], ai < numTiles);
  temp[bi] = select(0u, tileCounts.counts[bi], bi < numTiles);

  // Up-sweep (reduce) phase
  var d = WORKGROUP_SIZE;
  for (var d = WORKGROUP_SIZE; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (tid < d) {
      let ai = offset * (2u * tid + 1u) - 1u;
      let bi = offset * (2u * tid + 2u) - 1u;
      temp[bi] += temp[ai];
    }
    offset *= 2u;
  }

  // Clear last element (becomes total sum)
  if (tid == 0u) {
    temp[WORKGROUP_SIZE * 2u - 1u] = 0u;
  }

  // Down-sweep phase
  for (d = 1u; d <= WORKGROUP_SIZE; d *= 2u) {
    offset >>= 1u;
    workgroupBarrier();
    if (tid < d) {
      let ai = offset * (2u * tid + 1u) - 1u;
      let bi = offset * (2u * tid + 2u) - 1u;
      let t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  workgroupBarrier();

  // Write results
  if (ai < numTiles) {
    tileOffsets.offsets[ai] = temp[ai];
  }
  if (bi < numTiles) {
    tileOffsets.offsets[bi] = temp[bi];
  }
}
```

**Workgroup Size:** 256-512 (handles up to 512-1024 tiles in single pass)

**For Larger Tile Counts:** Use hierarchical scan:
1. Scan within workgroups → partial sums
2. Scan the partial sums → offsets
3. Add offsets back to original scans

**Performance Estimate:** <0.2ms for typical tile counts (100-1000 tiles)

**Alternative:** Use GPU vendor libraries if available (e.g., WebGPU doesn't have them, but we can implement our own)

---

#### Pass 3: Fill Tile Lists

**Purpose:** Fill the final tile index lists using the computed offsets

**Input:**
- `ProjectedSplat` buffer (read-only)
- `SortedIndices` buffer (read-only)
- `TileOffsets` buffer (read-only)
- `TileCounts` buffer (read-only) - need original counts to bounds-check

**Output:**
- `TileIndices` buffer (storage)
  - Format: `array<u32>` (flat array of splat indices)
  - Size: totalSplats × 4 bytes (where totalSplats = sum of all tile counts)

**Temporary:**
- `TileCurrentOffsets` buffer (storage, read-write)
  - Format: `array<atomic<u32>>`
  - Size: numTiles × 4 bytes
  - Initialized to copy of `TileOffsets` (starting positions)
  - Used to track current write position per tile

**Shader:** `src/shaders/fill-tile-lists.wgsl`

```wgsl
@workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let sortedIdx = globalId.x;
  if (sortedIdx >= numSplats) { return; }

  let splatIdx = sortedIndices[sortedIdx];
  let projected = projectedSplats.splats[splatIdx];

  // Determine overlapping tiles (same logic as Pass 1)
  let minTileX = u32(max(0.0, floor(projected.screenBoundsMin.x / f32(tileSize))));
  let minTileY = u32(max(0.0, floor(projected.screenBoundsMin.y / f32(tileSize))));
  let maxTileX = u32(min(f32(numTilesX - 1), floor(projected.screenBoundsMax.x / f32(tileSize))));
  let maxTileY = u32(min(f32(numTilesY - 1), floor(projected.screenBoundsMax.y / f32(tileSize))));

  // For each overlapping tile, atomically get next slot and write index
  for (var ty = minTileY; ty <= maxTileY; ty++) {
    for (var tx = minTileX; tx <= maxTileX; tx++) {
      let tileIdx = ty * numTilesX + tx;

      // Atomically get and increment write position
      let writePos = atomicAdd(&tileCurrentOffsets.offsets[tileIdx], 1u);

      // Write splat index to tile list
      tileIndices.indices[writePos] = splatIdx;
    }
  }
}
```

**Performance Estimate:** ~0.5-1ms for 120k splats

**Key Detail:** Because we process splats in sorted order (far to near), the indices written to each tile list are automatically in sorted order!

---

### 4. TypeScript Class: GPUTileBinner

**Purpose:** Encapsulate the 3-pass tile binning pipeline

**Constructor:**
```typescript
constructor(device: GPUDevice, maxSplats: number, tileSize: number)
```

**Buffers:**
- `tileCountsBuffer`: atomic counters (numTiles × 4 bytes)
- `tileOffsetsBuffer`: prefix sum results (numTiles × 4 bytes)
- `tileCurrentOffsetsBuffer`: working copy for fill pass (numTiles × 4 bytes)
- `tileIndicesBuffer`: flat index array (maxSplats × 4 × maxOverlap)
  - Need to estimate max size or use dynamic allocation

**Methods:**

```typescript
/**
 * Run complete 3-pass binning
 */
binSplats(
  commandEncoder: GPUCommandEncoder,
  projectedBuffer: GPUBuffer,
  sortedIndicesBuffer: GPUBuffer,
  numSplats: number,
  screenWidth: number,
  screenHeight: number
): void {
  // Calculate tile grid dimensions
  const numTilesX = Math.ceil(screenWidth / this.tileSize);
  const numTilesY = Math.ceil(screenHeight / this.tileSize);
  const numTiles = numTilesX * numTilesY;

  // Zero tile counts
  this.zeroBuffer(commandEncoder, this.tileCountsBuffer, numTiles);

  // Pass 1: Count
  this.runCountPass(commandEncoder, projectedBuffer, sortedIndicesBuffer, numSplats, numTilesX, numTilesY);

  // Pass 2: Prefix sum
  this.runPrefixSumPass(commandEncoder, numTiles);

  // Copy offsets to current offsets (for atomic increments in fill pass)
  this.copyBuffer(commandEncoder, this.tileOffsetsBuffer, this.tileCurrentOffsetsBuffer, numTiles * 4);

  // Pass 3: Fill
  this.runFillPass(commandEncoder, projectedBuffer, sortedIndicesBuffer, numSplats, numTilesX, numTilesY);
}

getTileOffsetsBuffer(): GPUBuffer;
getTileIndicesBuffer(): GPUBuffer;
getTileCountsBuffer(): GPUBuffer;
```

**Buffer Size Strategy:**

For `tileIndicesBuffer`, we need total size = sum of all tile counts. Options:

1. **Conservative Pre-allocation:** Assume worst case (all splats in all tiles)
   - Size: `numSplats × numTiles × 4` bytes
   - Too wasteful for large scenes

2. **Dynamic Allocation (RECOMMENDED):**
   - After Pass 1 (count), read back just the total sum (4 bytes!)
   - Allocate exact buffer size
   - Run Pass 2 and 3
   - One tiny readback is acceptable

3. **Heuristic:** Assume average splat overlaps ~4 tiles
   - Size: `numSplats × 4 × 4` bytes
   - Good balance for typical scenes

**Recommendation:** Use option 2 - one 4-byte readback to get exact size is negligible compared to current multi-MB readbacks.

---

## Implementation Plan

### Phase 1: Depth Key Extractor (Simplest)

1. Create `src/shaders/extract-depth-keys.wgsl`
2. Create `src/DepthKeyExtractor.ts`
3. Modify `RadixSorter.ts`:
   - Remove CPU depth extraction
   - Remove projected buffer readback
   - Accept pre-filled keys/payload buffers
   - Keep results on GPU
4. Update `main.ts` to wire up depth extraction before radix sort

**Test:** Verify sorting still works (can temporarily read back to validate)

**Expected Improvement:** -1-4MB readback, ~1-3ms faster

---

### Phase 2: Prefix Sum Shader (Foundation)

1. Create `src/shaders/prefix-sum.wgsl`
2. Create `src/PrefixSumScanner.ts`
3. Add unit tests with known inputs/outputs
4. Optimize workgroup size for target hardware

**Test:** Validate against CPU prefix sum with various array sizes

**Expected Time:** This is the most complex shader, allocate extra time

---

### Phase 3: GPU Tile Binner (Main Event)

1. Create `src/shaders/count-tile-hits.wgsl`
2. Create `src/shaders/fill-tile-lists.wgsl`
3. Create `src/GPUTileBinner.ts`
4. Implement dynamic buffer allocation (single small readback for total count)
5. Update `main.ts` to use GPU tile binner instead of CPU version

**Test:**
- Verify tile counts match CPU version
- Verify indices are in sorted order within each tile
- Visual test: rendering should be identical

**Expected Improvement:** -120-480KB readback, ~2-5ms faster (depending on CPU speed)

---

### Phase 4: Integration & Optimization

1. Remove old `TileBinner.ts` (CPU version)
2. Add performance monitoring for each GPU pass
3. Optimize workgroup sizes based on profiling
4. Consider async compute if supported (overlap with rendering)
5. Add validation mode (toggle between CPU/GPU binning for debugging)

---

## Performance Expectations

### Before (Current):
```
Projection:    0.2ms  (GPU)
Readback #1:   1.5ms  (GPU→CPU, projected data)
Extract Depth: 0.3ms  (CPU)
Radix Sort:    2.0ms  (GPU)
Readback #2:   0.5ms  (GPU→CPU, sorted indices)
Readback #3:   1.5ms  (GPU→CPU, projected data again)
Tile Binning:  3.0ms  (CPU)
Render:        2.0ms  (GPU)
────────────────────
TOTAL:        11.0ms  (90 FPS)
```

### After (GPU-only):
```
Projection:       0.2ms  (GPU)
Extract Keys:     0.1ms  (GPU) ← new
Radix Sort:       2.0ms  (GPU)
Count Hits:       0.6ms  (GPU) ← new
Prefix Sum:       0.2ms  (GPU) ← new
Fill Lists:       0.7ms  (GPU) ← new
Render:           2.0ms  (GPU)
────────────────────────
TOTAL:            5.8ms  (172 FPS)
```

**Expected Speedup:** ~2x faster
**Key Wins:**
- Eliminated ~3.5ms of readback latency
- Eliminated ~3.0ms of CPU processing
- GPU can run fully pipelined

---

## Potential Issues & Mitigations

### Issue 1: Atomic Contention on Hot Tiles

**Scenario:** Many splats overlap the same tile → atomics serialize

**Mitigation:**
- Modern GPUs handle this well up to ~100 splats/tile
- For pathological cases (1000+ splats/tile), consider hierarchical binning
- Profile first - this is rarely a bottleneck in practice

---

### Issue 2: Prefix Sum for Large Tile Counts

**Scenario:** 4K display has ~8000 tiles, single workgroup can't handle it

**Solution:** Hierarchical scan
1. Divide into chunks of 512 tiles
2. Scan each chunk → 16 partial sums
3. Scan the partial sums → 16 offsets
4. Add offsets back to chunk results

**Implementation:** Extend `PrefixSumScanner` with multi-level support

---

### Issue 3: Dynamic Buffer Allocation

**Scenario:** Need to read back total count before allocating TileIndices buffer

**Solution:**
- Single 4-byte readback is acceptable (vs 1-4MB currently)
- Could even skip readback by using conservative estimate (4× splat count)
- Modern GPUs have plenty of VRAM

---

### Issue 4: Debugging Difficulty

**Scenario:** GPU bugs are hard to debug

**Solution:**
- Keep CPU path as validation mode (toggle via flag)
- Add buffer readback utilities for debugging
- Implement visualization of intermediate buffers
- Unit test each shader independently

---

## File Structure

```
src/
├── DepthKeyExtractor.ts          (NEW)
├── PrefixSumScanner.ts            (NEW)
├── GPUTileBinner.ts               (NEW)
├── RadixSorter.ts                 (MODIFIED - remove readbacks)
├── TileBinner.ts                  (DEPRECATED - keep for validation)
├── shaders/
│   ├── extract-depth-keys.wgsl   (NEW)
│   ├── prefix-sum.wgsl           (NEW)
│   ├── count-tile-hits.wgsl      (NEW)
│   └── fill-tile-lists.wgsl      (NEW)
└── main.ts                        (MODIFIED - wire up new pipeline)
```

---

## Testing Strategy

### Unit Tests (Per Component)

1. **DepthKeyExtractor**
   - Input: Known ProjectedSplat data
   - Verify: Depth→key conversion correctness
   - Verify: Proper padding for oversized indices

2. **PrefixSumScanner**
   - Input: `[1, 2, 3, 4, 5]`
   - Expected: `[0, 1, 3, 6, 10]`
   - Test edge cases: zeros, large numbers, power-of-2 sizes

3. **GPUTileBinner - Count Pass**
   - Input: Splat overlapping known tiles
   - Verify: Correct tile counts via readback

4. **GPUTileBinner - Fill Pass**
   - Input: Pre-computed offsets
   - Verify: Indices in correct positions
   - Verify: Sorted order maintained

### Integration Tests

1. **Visual Comparison**
   - Toggle CPU/GPU binning
   - Verify identical rendering

2. **Performance Regression**
   - Profile both versions
   - Verify GPU version is faster

3. **Stress Test**
   - 1M+ splats
   - Verify no crashes, corruption

---

## Future Optimizations

### 1. Persistent Tile Buffers
- Don't reallocate every frame
- Only resize when screen dimensions change

### 2. Async Compute
- Overlap GPU passes with rendering
- Requires command buffer pipelining

### 3. Culling Before Sort
- Frustum cull in projection shader
- Reduce splat count for sort

### 4. Hierarchical Binning
- Coarse bins → fine bins
- Reduce atomic contention

### 5. Multi-Resolution Tiles
- Small tiles for dense areas
- Large tiles for sparse areas

---

## Success Criteria

✅ Zero CPU↔GPU readbacks (except optional 4-byte total count)
✅ Full GPU pipeline from projection to render
✅ 2x performance improvement over current implementation
✅ Identical visual output to CPU version
✅ Clean, maintainable code with good separation of concerns

---

## References

- **Parallel Prefix Sum (Scan):** Blelloch, "Prefix Sums and Their Applications" (1990)
- **Radix Sort:** "Fast Radix Sort on GPUs" (Merrill & Grimshaw, 2011)
- **Gaussian Splatting:** "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023)
- **GPU Gems 3 - Chapter 39:** Parallel Prefix Sum (Scan) with CUDA

---

## Timeline Estimate

- **Phase 1** (Depth extractor): 2-3 hours
- **Phase 2** (Prefix sum): 4-6 hours (most complex)
- **Phase 3** (GPU binning): 3-4 hours
- **Phase 4** (Integration): 2-3 hours
- **Testing & Debug**: 3-4 hours

**Total:** 14-20 hours for complete implementation
