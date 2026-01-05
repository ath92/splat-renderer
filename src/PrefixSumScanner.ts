import shaderCode from "./shaders/prefix-sum.wgsl?raw";

/**
 * GPU parallel prefix sum (exclusive scan)
 * Supports up to 512 elements in single-pass mode
 * For larger arrays, use hierarchical scan
 */
export class PrefixSumScanner {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private readonly maxSinglePassElements = 512; // WORKGROUP_SIZE * 2
  private tempBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice) {
    this.device = device;

    const { pipeline, bindGroupLayout } = this.createPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }

  private createPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderModule = this.device.createShaderModule({
      label: "Prefix sum shader",
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
      label: "Prefix sum pipeline",
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

  /**
   * Perform exclusive prefix sum on input buffer
   * Uses GPU for small arrays (<= 512), CPU for larger arrays
   * @param commandEncoder Command encoder
   * @param inputBuffer Input buffer (read-only)
   * @param outputBuffer Output buffer (write)
   * @param numElements Number of elements to scan
   */
  async scan(
    commandEncoder: GPUCommandEncoder,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    numElements: number
  ): Promise<void> {
    if (numElements <= this.maxSinglePassElements) {
      // Single-pass GPU for small arrays
      this.scanSinglePass(commandEncoder, inputBuffer, outputBuffer, numElements);
    } else {
      // CPU-based for large arrays (simpler and fast enough for ~10k elements)
      await this.scanCPU(inputBuffer, outputBuffer, numElements);
    }
  }

  /**
   * Single-pass scan for up to 512 elements
   */
  private scanSinglePass(
    commandEncoder: GPUCommandEncoder,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    numElements: number
  ): void {
    // Create uniform buffer
    const uniformData = new Uint32Array([numElements]);
    const uniformBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    });

    // Dispatch compute pass
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(1); // Single workgroup
    computePass.end();

    // Track temporary buffer for cleanup
    this.tempBuffers.push(uniformBuffer);
  }

  /**
   * CPU-based prefix sum for large arrays
   * Simple and fast enough for ~10k elements
   */
  private async scanCPU(
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    numElements: number
  ): Promise<void> {
    // Read input from GPU
    const readbackBuffer = this.device.createBuffer({
      size: numElements * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputBuffer, 0, readbackBuffer, 0, numElements * 4);
    this.device.queue.submit([encoder.finish()]);

    // Compute prefix sum on CPU
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const inputData = new Uint32Array(readbackBuffer.getMappedRange());

    const outputData = new Uint32Array(numElements);
    let sum = 0;
    for (let i = 0; i < numElements; i++) {
      outputData[i] = sum;
      sum += inputData[i];
    }

    readbackBuffer.unmap();
    readbackBuffer.destroy();

    // Write result back to GPU
    this.device.queue.writeBuffer(outputBuffer, 0, outputData);
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach((buffer) => buffer.destroy());
    this.tempBuffers = [];
  }
}
