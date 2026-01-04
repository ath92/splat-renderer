/**
 * Bins splats to screen tiles with dynamic allocation
 * Two-pass approach:
 * 1. Count splats per tile
 * 2. Compute prefix sum for offsets and allocate exact buffer
 * 3. Bin splats using dynamic offsets
 */
export class TileBinner {
  private device: GPUDevice;
  private numSplats: number;
  private tileSize: number;
  private numTilesX: number = 0;
  private numTilesY: number = 0;
  private numTiles: number = 0;

  // Buffers
  private tileListsBuffer: GPUBuffer | null = null;
  private tileOffsetsBuffer: GPUBuffer | null = null; // Prefix sum offsets
  private splatIndicesBuffer: GPUBuffer | null = null;
  private tempBuffers: GPUBuffer[] = [];

  // Cached counts and offsets from last frame
  private tileCounts: Uint32Array | null = null;
  private tileOffsets: Uint32Array | null = null;
  private totalSplatCount: number = 0;

  // Pipelines
  private resetPipeline: GPUComputePipeline;
  private resetBindGroupLayout: GPUBindGroupLayout;
  private countPipeline: GPUComputePipeline;
  private countBindGroupLayout: GPUBindGroupLayout;
  private binPipeline: GPUComputePipeline;
  private binBindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, numSplats: number, tileSize: number = 16) {
    this.device = device;
    this.numSplats = numSplats;
    this.tileSize = tileSize;

    // Create reset pipeline (clears tile counters)
    const resetResult = this.createResetPipeline();
    this.resetPipeline = resetResult.pipeline;
    this.resetBindGroupLayout = resetResult.bindGroupLayout;

    // Create counting pipeline
    const countResult = this.createCountPipeline();
    this.countPipeline = countResult.pipeline;
    this.countBindGroupLayout = countResult.bindGroupLayout;

    // Create binning pipeline
    const binResult = this.createBinPipeline();
    this.binPipeline = binResult.pipeline;
    this.binBindGroupLayout = binResult.bindGroupLayout;
  }

  private createResetPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct TileList {
        count: atomic<u32>,
      }

      struct TileLists {
        lists: array<TileList>,
      }

      @group(0) @binding(0) var<storage, read_write> tileLists: TileLists;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let tileIdx = globalId.x;
        if (tileIdx >= arrayLength(&tileLists.lists)) {
          return;
        }

        atomicStore(&tileLists.lists[tileIdx].count, 0u);
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Tile reset shader",
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Tile reset pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    return { pipeline, bindGroupLayout };
  }

  private createCountPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct ProjectedSplat {
        screenBoundsMin: vec2f,
        screenBoundsMax: vec2f,
        depth: f32,
        originalIndex: u32,
        _padding: vec2f,
      }

      struct ProjectedSplats {
        splats: array<ProjectedSplat>,
      }

      struct TileList {
        count: atomic<u32>,
      }

      struct TileLists {
        lists: array<TileList>,
      }

      struct CountParams {
        tileSize: u32,
        numTilesX: u32,
        numTilesY: u32,
        screenWidth: u32,
        screenHeight: u32,
      }

      @group(0) @binding(0) var<uniform> params: CountParams;
      @group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
      @group(0) @binding(2) var<storage, read_write> tileLists: TileLists;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let splatIdx = globalId.x;
        if (splatIdx >= arrayLength(&projectedSplats.splats)) {
          return;
        }

        let splat = projectedSplats.splats[splatIdx];

        // Clamp bounds to screen
        let minX = max(splat.screenBoundsMin.x, 0.0);
        let minY = max(splat.screenBoundsMin.y, 0.0);
        let maxX = min(splat.screenBoundsMax.x, f32(params.screenWidth));
        let maxY = min(splat.screenBoundsMax.y, f32(params.screenHeight));

        // Skip if completely off-screen
        if (minX >= maxX || minY >= maxY) {
          return;
        }

        // Calculate overlapping tiles
        let minTileX = u32(floor(minX / f32(params.tileSize)));
        let maxTileX = min(u32(floor(maxX / f32(params.tileSize))), params.numTilesX - 1u);
        let minTileY = u32(floor(minY / f32(params.tileSize)));
        let maxTileY = min(u32(floor(maxY / f32(params.tileSize))), params.numTilesY - 1u);

        // Count splats per tile (don't write indices yet)
        for (var ty = minTileY; ty <= maxTileY; ty++) {
          for (var tx = minTileX; tx <= maxTileX; tx++) {
            let tileIdx = ty * params.numTilesX + tx;
            atomicAdd(&tileLists.lists[tileIdx].count, 1u);
          }
        }
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Tile counting shader",
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Tile counting pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    return { pipeline, bindGroupLayout };
  }

  private createBinPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct ProjectedSplat {
        screenBoundsMin: vec2f,
        screenBoundsMax: vec2f,
        depth: f32,
        originalIndex: u32,
        _padding: vec2f,
      }

      struct ProjectedSplats {
        splats: array<ProjectedSplat>,
      }

      struct TileList {
        count: atomic<u32>,
      }

      struct TileLists {
        lists: array<TileList>,
      }

      struct TileOffsets {
        offsets: array<u32>,
      }

      struct SplatIndices {
        indices: array<u32>,
      }

      struct BinParams {
        tileSize: u32,
        numTilesX: u32,
        numTilesY: u32,
        screenWidth: u32,
        screenHeight: u32,
      }

      @group(0) @binding(0) var<uniform> params: BinParams;
      @group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
      @group(0) @binding(2) var<storage, read_write> tileLists: TileLists;
      @group(0) @binding(3) var<storage, read> tileOffsets: TileOffsets;
      @group(0) @binding(4) var<storage, read_write> splatIndices: SplatIndices;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let splatIdx = globalId.x;
        if (splatIdx >= arrayLength(&projectedSplats.splats)) {
          return;
        }

        let splat = projectedSplats.splats[splatIdx];

        // Clamp bounds to screen
        let minX = max(splat.screenBoundsMin.x, 0.0);
        let minY = max(splat.screenBoundsMin.y, 0.0);
        let maxX = min(splat.screenBoundsMax.x, f32(params.screenWidth));
        let maxY = min(splat.screenBoundsMax.y, f32(params.screenHeight));

        // Skip if completely off-screen
        if (minX >= maxX || minY >= maxY) {
          return;
        }

        // Calculate overlapping tiles
        let minTileX = u32(floor(minX / f32(params.tileSize)));
        let maxTileX = min(u32(floor(maxX / f32(params.tileSize))), params.numTilesX - 1u);
        let minTileY = u32(floor(minY / f32(params.tileSize)));
        let maxTileY = min(u32(floor(maxY / f32(params.tileSize))), params.numTilesY - 1u);

        // Bin splats using dynamic offsets
        for (var ty = minTileY; ty <= maxTileY; ty++) {
          for (var tx = minTileX; tx <= maxTileX; tx++) {
            let tileIdx = ty * params.numTilesX + tx;

            // Get base offset for this tile
            let baseOffset = tileOffsets.offsets[tileIdx];

            // Atomically get write position within tile
            let localPos = atomicAdd(&tileLists.lists[tileIdx].count, 1u);

            // Write to global buffer at: baseOffset + localPos
            let writeIdx = baseOffset + localPos;
            splatIndices.indices[writeIdx] = splatIdx;
          }
        }
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Tile binning shader",
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Tile binning pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    return { pipeline, bindGroupLayout };
  }

  private ensureBuffers(screenWidth: number, screenHeight: number): void {
    const newNumTilesX = Math.ceil(screenWidth / this.tileSize);
    const newNumTilesY = Math.ceil(screenHeight / this.tileSize);
    const newNumTiles = newNumTilesX * newNumTilesY;

    if (newNumTiles === this.numTiles) {
      return; // Buffers already correct size
    }

    this.numTilesX = newNumTilesX;
    this.numTilesY = newNumTilesY;
    this.numTiles = newNumTiles;

    // Destroy old buffers
    if (this.tileListsBuffer) this.tileListsBuffer.destroy();
    if (this.tileOffsetsBuffer) this.tileOffsetsBuffer.destroy();
    // Don't destroy splatIndicesBuffer yet - it will be recreated per-frame

    // Create tile lists buffer (counters with MAP_READ for readback)
    this.tileListsBuffer = this.device.createBuffer({
      size: this.numTiles * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Create tile offsets buffer (will be updated each frame)
    this.tileOffsetsBuffer = this.device.createBuffer({
      size: this.numTiles * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Reset cached data
    this.tileCounts = null;
    this.tileOffsets = null;
  }

  async binSorted(
    sortedIndices: Uint32Array,
    projectedBuffer: GPUBuffer,
    screenWidth: number,
    screenHeight: number
  ): Promise<void> {
    this.ensureBuffers(screenWidth, screenHeight);

    // Read back projected data to determine which tiles each splat overlaps
    const readbackBuffer = this.device.createBuffer({
      size: this.numSplats * 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      projectedBuffer,
      0,
      readbackBuffer,
      0,
      this.numSplats * 32
    );
    this.device.queue.submit([commandEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const projectedData = new Float32Array(readbackBuffer.getMappedRange());

    // Count splats per tile and compute offsets
    this.tileCounts = new Uint32Array(this.numTiles);

    // First pass: count
    for (const splatIdx of sortedIndices) {
      const offset = splatIdx * 8;
      const minX = Math.max(projectedData[offset + 0], 0);
      const minY = Math.max(projectedData[offset + 1], 0);
      const maxX = Math.min(projectedData[offset + 2], screenWidth);
      const maxY = Math.min(projectedData[offset + 3], screenHeight);

      if (minX >= maxX || minY >= maxY) continue;

      const minTileX = Math.floor(minX / this.tileSize);
      const maxTileX = Math.min(Math.floor(maxX / this.tileSize), this.numTilesX - 1);
      const minTileY = Math.floor(minY / this.tileSize);
      const maxTileY = Math.min(Math.floor(maxY / this.tileSize), this.numTilesY - 1);

      for (let ty = minTileY; ty <= maxTileY; ty++) {
        for (let tx = minTileX; tx <= maxTileX; tx++) {
          const tileIdx = ty * this.numTilesX + tx;
          this.tileCounts[tileIdx]++;
        }
      }
    }

    // Compute prefix sum
    this.tileOffsets = new Uint32Array(this.numTiles);
    let runningSum = 0;
    for (let i = 0; i < this.numTiles; i++) {
      this.tileOffsets[i] = runningSum;
      runningSum += this.tileCounts[i];
    }
    this.totalSplatCount = runningSum;

    // Allocate exact-size buffer
    if (this.splatIndicesBuffer) {
      this.splatIndicesBuffer.destroy();
    }
    this.splatIndicesBuffer = this.device.createBuffer({
      size: Math.max(this.totalSplatCount * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Second pass: fill indices in sorted order
    const indices = new Uint32Array(this.totalSplatCount);
    const currentPos = new Uint32Array(this.numTiles);
    this.tileOffsets.forEach((offset, i) => (currentPos[i] = offset));

    for (const splatIdx of sortedIndices) {
      const offset = splatIdx * 8;
      const minX = Math.max(projectedData[offset + 0], 0);
      const minY = Math.max(projectedData[offset + 1], 0);
      const maxX = Math.min(projectedData[offset + 2], screenWidth);
      const maxY = Math.min(projectedData[offset + 3], screenHeight);

      if (minX >= maxX || minY >= maxY) continue;

      const minTileX = Math.floor(minX / this.tileSize);
      const maxTileX = Math.min(Math.floor(maxX / this.tileSize), this.numTilesX - 1);
      const minTileY = Math.floor(minY / this.tileSize);
      const maxTileY = Math.min(Math.floor(maxY / this.tileSize), this.numTilesY - 1);

      for (let ty = minTileY; ty <= maxTileY; ty++) {
        for (let tx = minTileX; tx <= maxTileX; tx++) {
          const tileIdx = ty * this.numTilesX + tx;
          indices[currentPos[tileIdx]++] = splatIdx;
        }
      }
    }

    readbackBuffer.unmap();
    readbackBuffer.destroy();

    // Upload indices and tile data
    this.device.queue.writeBuffer(this.splatIndicesBuffer, 0, indices);
    this.device.queue.writeBuffer(this.tileOffsetsBuffer!, 0, this.tileOffsets.buffer);

    // Upload counts to tileListsBuffer
    const tileListsData = new Uint32Array(this.numTiles);
    tileListsData.set(this.tileCounts);
    this.device.queue.writeBuffer(this.tileListsBuffer!, 0, tileListsData);
  }

  async bin(
    commandEncoder: GPUCommandEncoder,
    projectedBuffer: GPUBuffer,
    screenWidth: number,
    screenHeight: number
  ): Promise<void> {
    this.ensureBuffers(screenWidth, screenHeight);

    // Create params buffer (without maxSplatsPerTile)
    const params = new Uint32Array([
      this.tileSize,
      this.numTilesX,
      this.numTilesY,
      screenWidth,
      screenHeight,
    ]);

    const paramsBuffer = this.device.createBuffer({
      size: params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(paramsBuffer.getMappedRange()).set(params);
    paramsBuffer.unmap();

    // Step 1: Reset tile counters
    const resetBindGroup = this.device.createBindGroup({
      layout: this.resetBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.tileListsBuffer! } }],
    });

    let computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.resetPipeline);
    computePass.setBindGroup(0, resetBindGroup);
    const resetWorkgroups = Math.ceil(this.numTiles / 64);
    computePass.dispatchWorkgroups(resetWorkgroups);
    computePass.end();

    // Step 2: Count splats per tile
    const countBindGroup = this.device.createBindGroup({
      layout: this.countBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: this.tileListsBuffer! } },
      ],
    });

    computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.countPipeline);
    computePass.setBindGroup(0, countBindGroup);
    const countWorkgroups = Math.ceil(this.numSplats / 64);
    computePass.dispatchWorkgroups(countWorkgroups);
    computePass.end();

    // Step 3: Read back counts and compute prefix sum on CPU
    const readbackBuffer = this.device.createBuffer({
      size: this.numTiles * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(
      this.tileListsBuffer!,
      0,
      readbackBuffer,
      0,
      this.numTiles * 4
    );

    // Submit and wait for counts
    this.device.queue.submit([commandEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const countsData = new Uint32Array(readbackBuffer.getMappedRange());
    this.tileCounts = new Uint32Array(countsData); // Copy data
    readbackBuffer.unmap();
    readbackBuffer.destroy();

    // Compute prefix sum (exclusive scan)
    this.tileOffsets = new Uint32Array(this.numTiles);
    let runningSum = 0;
    for (let i = 0; i < this.numTiles; i++) {
      this.tileOffsets[i] = runningSum;
      runningSum += this.tileCounts[i];
    }
    this.totalSplatCount = runningSum;

    // Allocate exact-size splat indices buffer
    if (this.splatIndicesBuffer) {
      this.splatIndicesBuffer.destroy();
    }
    this.splatIndicesBuffer = this.device.createBuffer({
      size: Math.max(this.totalSplatCount * 4, 4), // At least 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Upload offsets to GPU
    this.device.queue.writeBuffer(this.tileOffsetsBuffer!, 0, this.tileOffsets.buffer);

    // Step 4: Reset counters again (will be used as write positions)
    const commandEncoder2 = this.device.createCommandEncoder();

    computePass = commandEncoder2.beginComputePass();
    computePass.setPipeline(this.resetPipeline);
    computePass.setBindGroup(0, resetBindGroup);
    computePass.dispatchWorkgroups(resetWorkgroups);
    computePass.end();

    // Step 5: Bin splats with dynamic offsets
    const binBindGroup = this.device.createBindGroup({
      layout: this.binBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: this.tileListsBuffer! } },
        { binding: 3, resource: { buffer: this.tileOffsetsBuffer! } },
        { binding: 4, resource: { buffer: this.splatIndicesBuffer! } },
      ],
    });

    computePass = commandEncoder2.beginComputePass();
    computePass.setPipeline(this.binPipeline);
    computePass.setBindGroup(0, binBindGroup);
    computePass.dispatchWorkgroups(countWorkgroups);
    computePass.end();

    this.device.queue.submit([commandEncoder2.finish()]);

    // Save buffer for cleanup after frame
    this.tempBuffers.push(paramsBuffer);
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach(buffer => buffer.destroy());
    this.tempBuffers = [];
  }

  getTileSize(): number {
    return this.tileSize;
  }

  getNumTilesX(): number {
    return this.numTilesX;
  }

  getNumTilesY(): number {
    return this.numTilesY;
  }

  getMaxSplatsPerTile(): number {
    // Return actual max from this frame's counts
    if (!this.tileCounts) return 0;
    return Math.max(...this.tileCounts);
  }

  getTileListsBuffer(): GPUBuffer {
    return this.tileListsBuffer!;
  }

  getTileOffsetsBuffer(): GPUBuffer {
    return this.tileOffsetsBuffer!;
  }

  getSplatIndicesBuffer(): GPUBuffer {
    return this.splatIndicesBuffer!;
  }

  getTileCounts(): Uint32Array | null {
    return this.tileCounts;
  }

  getTileOffsets(): Uint32Array | null {
    return this.tileOffsets;
  }

  destroy(): void {
    if (this.tileListsBuffer) this.tileListsBuffer.destroy();
    if (this.tileOffsetsBuffer) this.tileOffsetsBuffer.destroy();
    if (this.splatIndicesBuffer) this.splatIndicesBuffer.destroy();
    this.cleanupTempBuffers();
  }
}
