/**
 * Bins splats to screen tiles
 * Each splat is assigned to all tiles its bounding box overlaps
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
  private splatIndicesBuffer: GPUBuffer | null = null;
  private tempBuffers: GPUBuffer[] = [];

  // Pipelines
  private resetPipeline: GPUComputePipeline;
  private resetBindGroupLayout: GPUBindGroupLayout;
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

      struct SplatIndices {
        indices: array<u32>,
      }

      struct BinParams {
        tileSize: u32,
        numTilesX: u32,
        numTilesY: u32,
        screenWidth: u32,
        screenHeight: u32,
        maxSplatsPerTile: u32,
      }

      @group(0) @binding(0) var<uniform> params: BinParams;
      @group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
      @group(0) @binding(2) var<storage, read_write> tileLists: TileLists;
      @group(0) @binding(3) var<storage, read_write> splatIndices: SplatIndices;

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

        // Add this splat to each overlapping tile
        for (var ty = minTileY; ty <= maxTileY; ty++) {
          for (var tx = minTileX; tx <= maxTileX; tx++) {
            let tileIdx = ty * params.numTilesX + tx;

            // Atomically increment tile counter
            let insertPos = atomicAdd(&tileLists.lists[tileIdx].count, 1u);

            // Check if we have space (conservative limit)
            if (insertPos < params.maxSplatsPerTile) {
              // Write splat index to this tile's section
              let writeIdx = tileIdx * params.maxSplatsPerTile + insertPos;
              splatIndices.indices[writeIdx] = splatIdx;
            }
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

  ensureBuffers(screenWidth: number, screenHeight: number): void {
    const newNumTilesX = Math.ceil(screenWidth / this.tileSize);
    const newNumTilesY = Math.ceil(screenHeight / this.tileSize);
    const newNumTiles = newNumTilesX * newNumTilesY;

    // Only recreate if dimensions changed
    if (
      newNumTilesX === this.numTilesX &&
      newNumTilesY === this.numTilesY &&
      this.tileListsBuffer !== null
    ) {
      return;
    }

    this.numTilesX = newNumTilesX;
    this.numTilesY = newNumTilesY;
    this.numTiles = newNumTiles;

    // Destroy old buffers
    if (this.tileListsBuffer) this.tileListsBuffer.destroy();
    if (this.splatIndicesBuffer) this.splatIndicesBuffer.destroy();

    // Create tile lists buffer (just counters)
    // Each TileList: atomic<u32> = 4 bytes
    this.tileListsBuffer = this.device.createBuffer({
      size: this.numTiles * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Conservative allocation: assume average 16 tiles per splat (can overlap many tiles)
    // But store as maxSplatsPerTile × numTiles for simple indexing
    const maxSplatsPerTile = Math.min(
      Math.ceil((this.numSplats * 16) / this.numTiles),
      2048 // Cap at 2048 splats per tile (increased from 1024)
    );

    this.splatIndicesBuffer = this.device.createBuffer({
      size: this.numTiles * maxSplatsPerTile * 4, // u32 per index
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  bin(
    commandEncoder: GPUCommandEncoder,
    projectedBuffer: GPUBuffer,
    screenWidth: number,
    screenHeight: number
  ): void {
    this.ensureBuffers(screenWidth, screenHeight);

    // Conservative max splats per tile
    const maxSplatsPerTile = Math.min(
      Math.ceil((this.numSplats * 16) / this.numTiles),
      2048
    );

    // Create params buffer
    const params = new Uint32Array([
      this.tileSize,
      this.numTilesX,
      this.numTilesY,
      screenWidth,
      screenHeight,
      maxSplatsPerTile,
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

    // Step 2: Bin splats to tiles
    const binBindGroup = this.device.createBindGroup({
      layout: this.binBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: this.tileListsBuffer! } },
        { binding: 3, resource: { buffer: this.splatIndicesBuffer! } },
      ],
    });

    computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.binPipeline);
    computePass.setBindGroup(0, binBindGroup);
    const binWorkgroups = Math.ceil(this.numSplats / 64);
    computePass.dispatchWorkgroups(binWorkgroups);
    computePass.end();

    // Save buffer for cleanup after submit
    this.tempBuffers.push(paramsBuffer);
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach(buffer => buffer.destroy());
    this.tempBuffers = [];
  }

  getTileListsBuffer(): GPUBuffer {
    return this.tileListsBuffer!;
  }

  getSplatIndicesBuffer(): GPUBuffer {
    return this.splatIndicesBuffer!;
  }

  getNumTilesX(): number {
    return this.numTilesX;
  }

  getNumTilesY(): number {
    return this.numTilesY;
  }

  getTileSize(): number {
    return this.tileSize;
  }

  getMaxSplatsPerTile(): number {
    return Math.min(
      Math.ceil((this.numSplats * 16) / this.numTiles),
      2048
    );
  }

  destroy(): void {
    this.cleanupTempBuffers();
    if (this.tileListsBuffer) this.tileListsBuffer.destroy();
    if (this.splatIndicesBuffer) this.splatIndicesBuffer.destroy();
  }
}
