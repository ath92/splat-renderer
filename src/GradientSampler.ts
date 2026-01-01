export class GradientSampler {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private gradientBuffer: GPUBuffer;
  private readbackBuffer: GPUBuffer;
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

    // Create readback buffer
    this.readbackBuffer = device.createBuffer({
      size: gradientBufferSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  async evaluateGradients(
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer
  ): Promise<Float32Array> {
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: this.gradientBuffer } },
      ],
    });

    // Create command encoder
    const commandEncoder = this.device.createCommandEncoder();

    // Dispatch compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(this.numPoints / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // Copy gradient buffer to readback buffer
    commandEncoder.copyBufferToBuffer(
      this.gradientBuffer,
      0,
      this.readbackBuffer,
      0,
      this.numPoints * 4 * 4
    );

    // Submit commands
    this.device.queue.submit([commandEncoder.finish()]);

    // Read back results
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = this.readbackBuffer.getMappedRange();
    const gradientResults = new Float32Array(arrayBuffer).slice();
    this.readbackBuffer.unmap();

    return gradientResults;
  }

  destroy(): void {
    this.gradientBuffer.destroy();
    this.readbackBuffer.destroy();
  }
}
