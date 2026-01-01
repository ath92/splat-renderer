export class GradientSampler3D {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private gradientBuffer: GPUBuffer;
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
      label: "Compute gradient 3D shader",
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

  destroy(): void {
    this.gradientBuffer.destroy();
  }
}
