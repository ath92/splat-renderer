export class GradientSampler {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private gradientBuffer: GPUBuffer;
  private readbackBuffer: GPUBuffer | null = null;
  private numPoints: number;

  constructor(
    device: GPUDevice,
    computeShaderCode: string,
    numPoints: number
  ) {
    this.device = device;
    this.numPoints = numPoints;

    // Create compute shader module
    const computeShaderModule = device.createShaderModule({
      label: "Compute gradient shader",
      code: computeShaderCode,
    });

    // Create bind group layout
    this.bindGroupLayout = device.createBindGroupLayout({
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

    // Create compute pipeline
    this.computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: computeShaderModule,
        entryPoint: "computeMain",
      },
    });

    // Create gradient output buffer
    const gradientBufferSize = numPoints * 4 * 4; // vec4f per point
    this.gradientBuffer = device.createBuffer({
      size: gradientBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  evaluateGradients(
    commandEncoder: GPUCommandEncoder,
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer
  ): void {
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: this.gradientBuffer } },
      ],
    });

    // Dispatch compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(this.numPoints / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  getGradientBuffer(): GPUBuffer {
    return this.gradientBuffer;
  }

  // Optional async readback for debugging (non-blocking)
  async readbackGradientsAsync(): Promise<Float32Array | null> {
    // Create readback buffer on first use
    if (!this.readbackBuffer) {
      const gradientBufferSize = this.numPoints * 4 * 4;
      this.readbackBuffer = this.device.createBuffer({
        size: gradientBufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }

    // Check if buffer is already mapped (previous read still in progress)
    if (this.readbackBuffer.mapState !== "unmapped") {
      return null; // Skip this frame
    }

    // Copy gradient buffer to readback buffer
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.gradientBuffer,
      0,
      this.readbackBuffer,
      0,
      this.numPoints * 4 * 4
    );
    this.device.queue.submit([commandEncoder.finish()]);

    // Read back results asynchronously
    try {
      await this.readbackBuffer.mapAsync(GPUMapMode.READ);
      const arrayBuffer = this.readbackBuffer.getMappedRange();
      const gradientResults = new Float32Array(arrayBuffer).slice();
      this.readbackBuffer.unmap();
      return gradientResults;
    } catch (error) {
      console.error("Failed to read back gradients:", error);
      return null;
    }
  }

  destroy(): void {
    this.gradientBuffer.destroy();
    if (this.readbackBuffer) {
      this.readbackBuffer.destroy();
    }
  }
}
