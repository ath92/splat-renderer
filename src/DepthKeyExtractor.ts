import shaderCode from "./shaders/extract-depth-keys.wgsl?raw";

/**
 * Extracts depth keys from ProjectedSplat buffer for GPU radix sort
 */
export class DepthKeyExtractor {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
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
      label: "Depth key extraction shader",
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
      label: "Depth key extraction pipeline",
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
   * Extract depth keys from projected splats
   */
  extract(
    commandEncoder: GPUCommandEncoder,
    projectedBuffer: GPUBuffer,
    keysBuffer: GPUBuffer,
    payloadBuffer: GPUBuffer,
    numSplats: number,
    paddedSize: number
  ): void {
    // Create uniform buffer
    const uniformData = new Uint32Array([numSplats, paddedSize]);
    const uniformBuffer = this.device.createBuffer({
      size: uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: keysBuffer } },
        { binding: 3, resource: { buffer: payloadBuffer } },
      ],
    });

    // Dispatch compute pass
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);

    const workgroupCount = Math.ceil(paddedSize / 256);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // Track temporary buffer for cleanup after submission
    this.tempBuffers.push(uniformBuffer);
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach((buffer) => buffer.destroy());
    this.tempBuffers = [];
  }
}
