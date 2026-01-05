import countShaderCode from "./shaders/count-tile-hits.wgsl?raw";
import fillShaderCode from "./shaders/fill-tile-lists.wgsl?raw";
import { PrefixSumScanner } from "./PrefixSumScanner";

/**
 * GPU-based tile binning for sorted splats
 * Three-pass algorithm: Count → Prefix Sum → Fill
 */
export class GPUTileBinner {
  private device: GPUDevice;
  private tileSize: number;

  // Pipelines
  private countPipeline: GPUComputePipeline;
  private countBindGroupLayout: GPUBindGroupLayout;
  private fillPipeline: GPUComputePipeline;
  private fillBindGroupLayout: GPUBindGroupLayout;

  // Prefix sum scanner
  private prefixSumScanner: PrefixSumScanner;

  // Persistent buffers
  private tileCountsBuffer: GPUBuffer | null = null;
  private tileOffsetsBuffer: GPUBuffer | null = null;
  private tileCurrentOffsetsBuffer: GPUBuffer | null = null;
  private tileIndicesBuffer: GPUBuffer | null = null;

  // Current state
  private currentNumTiles = 0;
  private currentTotalIndices = 0;

  // Temporary buffers for cleanup
  private tempBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, tileSize: number) {
    this.device = device;
    this.tileSize = tileSize;

    const { pipeline: countPipeline, bindGroupLayout: countBindGroupLayout } =
      this.createCountPipeline();
    this.countPipeline = countPipeline;
    this.countBindGroupLayout = countBindGroupLayout;

    const { pipeline: fillPipeline, bindGroupLayout: fillBindGroupLayout } =
      this.createFillPipeline();
    this.fillPipeline = fillPipeline;
    this.fillBindGroupLayout = fillBindGroupLayout;

    this.prefixSumScanner = new PrefixSumScanner(device);
  }

  private createCountPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderModule = this.device.createShaderModule({
      label: "Count tile hits shader",
      code: countShaderCode,
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
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Count tile hits pipeline",
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

  private createFillPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderModule = this.device.createShaderModule({
      label: "Fill tile lists shader",
      code: fillShaderCode,
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
          buffer: { type: "read-only-storage" },
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
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Fill tile lists pipeline",
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

  private ensureBuffers(numTiles: number): void {
    // Always recreate tile counts buffer to ensure it starts clean
    // (atomic operations can accumulate if buffer isn't properly cleared)
    if (this.tileCountsBuffer) this.tileCountsBuffer.destroy();

    this.tileCountsBuffer = this.device.createBuffer({
      size: numTiles * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Only recreate other buffers if size changed
    if (this.currentNumTiles !== numTiles) {
      if (this.tileOffsetsBuffer) this.tileOffsetsBuffer.destroy();
      if (this.tileCurrentOffsetsBuffer) this.tileCurrentOffsetsBuffer.destroy();

      this.tileOffsetsBuffer = this.device.createBuffer({
        size: numTiles * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      this.tileCurrentOffsetsBuffer = this.device.createBuffer({
        size: numTiles * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      this.currentNumTiles = numTiles;
    }
  }

  /**
   * Run complete 3-pass GPU tile binning
   */
  async binSplats(
    commandEncoder: GPUCommandEncoder,
    projectedBuffer: GPUBuffer,
    sortedIndicesBuffer: GPUBuffer,
    numSplats: number,
    screenWidth: number,
    screenHeight: number
  ): Promise<void> {
    const numTilesX = Math.ceil(screenWidth / this.tileSize);
    const numTilesY = Math.ceil(screenHeight / this.tileSize);
    const numTiles = numTilesX * numTilesY;

    // Ensure buffers are allocated
    this.ensureBuffers(numTiles);

    // Create uniforms
    const uniformData = new Float32Array(8);
    const u32View = new Uint32Array(uniformData.buffer);
    u32View[0] = numSplats;
    u32View[1] = this.tileSize;
    u32View[2] = numTilesX;
    u32View[3] = numTilesY;
    uniformData[4] = screenWidth;
    uniformData[5] = screenHeight;

    const uniformBuffer = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    this.tempBuffers.push(uniformBuffer);

    // Note: tileCountsBuffer is freshly created, so already zeroed
    // Pass 1: Count tile hits
    const countBindGroup = this.device.createBindGroup({
      layout: this.countBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: sortedIndicesBuffer } },
        { binding: 3, resource: { buffer: this.tileCountsBuffer! } },
      ],
    });

    const countPass = commandEncoder.beginComputePass();
    countPass.setPipeline(this.countPipeline);
    countPass.setBindGroup(0, countBindGroup);
    countPass.dispatchWorkgroups(Math.ceil(numSplats / 256));
    countPass.end();

    // Submit to ensure counts are ready
    this.device.queue.submit([commandEncoder.finish()]);

    // Read back total count to allocate exact buffer size
    const totalCountBuffer = this.device.createBuffer({
      size: numTiles * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const readbackEncoder = this.device.createCommandEncoder();
    readbackEncoder.copyBufferToBuffer(
      this.tileCountsBuffer!,
      0,
      totalCountBuffer,
      0,
      numTiles * 4
    );
    this.device.queue.submit([readbackEncoder.finish()]);

    await totalCountBuffer.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(totalCountBuffer.getMappedRange());
    const totalIndices = counts.reduce((sum, count) => sum + count, 0);
    totalCountBuffer.unmap();
    totalCountBuffer.destroy();

    // Sanity check: warn if counts seem unreasonably high (but don't cap)
    const expectedAvgOverlap = 4; // Typical splat overlaps 2-4 tiles
    const maxReasonableIndices = numSplats * 50; // Very generous upper bound
    if (totalIndices > maxReasonableIndices) {
      console.error(
        `GPUTileBinner: totalIndices (${totalIndices}) is extremely high! ` +
        `Expected ~${numSplats * expectedAvgOverlap} for ${numSplats} splats. ` +
        `This indicates a serious bug in tile counting.`
      );
      // Don't cap - let it fail so we can see the issue
    } else if (totalIndices > numSplats * 20) {
      console.warn(
        `GPUTileBinner: totalIndices (${totalIndices}) is higher than expected. ` +
        `Expected ~${numSplats * expectedAvgOverlap} for ${numSplats} splats. ` +
        `Each splat is overlapping ~${(totalIndices / numSplats).toFixed(1)} tiles on average.`
      );
    }

    // Allocate exact size needed (no capping)
    if (this.currentTotalIndices !== totalIndices) {
      if (this.tileIndicesBuffer) {
        this.tileIndicesBuffer.destroy();
      }
      this.tileIndicesBuffer = this.device.createBuffer({
        size: Math.max(4, totalIndices * 4), // At least 4 bytes
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.currentTotalIndices = totalIndices;
    }

    // Pass 2: Prefix sum
    const commandEncoder2 = this.device.createCommandEncoder();
    await this.prefixSumScanner.scan(
      commandEncoder2,
      this.tileCountsBuffer!,
      this.tileOffsetsBuffer!,
      numTiles
    );

    // Copy offsets to current offsets for fill pass
    const commandEncoder2b = this.device.createCommandEncoder();
    commandEncoder2b.copyBufferToBuffer(
      this.tileOffsetsBuffer!,
      0,
      this.tileCurrentOffsetsBuffer!,
      0,
      numTiles * 4
    );

    this.device.queue.submit([commandEncoder2b.finish()]);

    // Pass 3: Fill tile lists
    const commandEncoder3 = this.device.createCommandEncoder();

    const fillBindGroup = this.device.createBindGroup({
      layout: this.fillBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: sortedIndicesBuffer } },
        { binding: 3, resource: { buffer: this.tileOffsetsBuffer! } },
        { binding: 4, resource: { buffer: this.tileCurrentOffsetsBuffer! } },
        { binding: 5, resource: { buffer: this.tileIndicesBuffer! } },
      ],
    });

    const fillPass = commandEncoder3.beginComputePass();
    fillPass.setPipeline(this.fillPipeline);
    fillPass.setBindGroup(0, fillBindGroup);
    fillPass.dispatchWorkgroups(Math.ceil(numSplats / 256));
    fillPass.end();

    this.device.queue.submit([commandEncoder3.finish()]);
  }

  getTileOffsetsBuffer(): GPUBuffer {
    if (!this.tileOffsetsBuffer) {
      throw new Error("Tile offsets buffer not initialized");
    }
    return this.tileOffsetsBuffer;
  }

  getTileIndicesBuffer(): GPUBuffer {
    if (!this.tileIndicesBuffer) {
      throw new Error("Tile indices buffer not initialized");
    }
    return this.tileIndicesBuffer;
  }

  getTileCountsBuffer(): GPUBuffer {
    if (!this.tileCountsBuffer) {
      throw new Error("Tile counts buffer not initialized");
    }
    return this.tileCountsBuffer;
  }

  getTileSize(): number {
    return this.tileSize;
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach((buffer) => buffer.destroy());
    this.tempBuffers = [];
    this.prefixSumScanner.cleanupTempBuffers();
  }

  destroy(): void {
    if (this.tileCountsBuffer) this.tileCountsBuffer.destroy();
    if (this.tileOffsetsBuffer) this.tileOffsetsBuffer.destroy();
    if (this.tileCurrentOffsetsBuffer) this.tileCurrentOffsetsBuffer.destroy();
    if (this.tileIndicesBuffer) this.tileIndicesBuffer.destroy();
    this.cleanupTempBuffers();
  }
}
