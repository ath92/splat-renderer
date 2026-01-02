export class PositionUpdater {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private numPoints: number;

  constructor(
    device: GPUDevice,
    updateShaderCode: string,
    numPoints: number
  ) {
    this.device = device;
    this.numPoints = numPoints;

    // Create compute shader module
    const computeShaderModule = device.createShaderModule({
      label: "Update positions 3D shader",
      code: updateShaderCode,
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
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
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
  }

  updatePositions(
    commandEncoder: GPUCommandEncoder,
    uniformBuffer: GPUBuffer,
    currentPositionBuffer: GPUBuffer,
    gradientBuffer: GPUBuffer,
    nextPositionBuffer: GPUBuffer
  ): void {
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: currentPositionBuffer } },
        { binding: 2, resource: { buffer: gradientBuffer } },
        { binding: 3, resource: { buffer: nextPositionBuffer } },
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
}
