/**
 * Manages splat properties: radius, color, and opacity
 * Layout: vec4(position.xyz, radius) + vec4(color.rgb, opacity)
 * Total: 8 floats (32 bytes) per splat
 */
export class SplatPropertyManager {
  private device: GPUDevice;
  private numSplats: number;
  private propertyBuffer: GPUBuffer;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, numSplats: number) {
    this.device = device;
    this.numSplats = numSplats;

    // Create property buffer: 8 floats per splat (position+radius, color+opacity)
    const bufferSize = numSplats * 32; // 8 floats * 4 bytes
    this.propertyBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Create pipeline once
    const { pipeline, bindGroupLayout } = this.createPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;

    // Initialize with default values
    this.initializeDefaults();
  }

  private initializeDefaults(): void {
    // Initialize all splats with default properties
    const data = new Float32Array(this.numSplats * 8);

    for (let i = 0; i < this.numSplats; i++) {
      const offset = i * 8;
      // Position will be filled from position buffer (first 3 floats)
      data[offset + 0] = 0; // x (will be updated from position buffer)
      data[offset + 1] = 0; // y
      data[offset + 2] = 0; // z
      data[offset + 3] = 0.04; // radius (larger default to match compute shader)

      // Color + opacity
      data[offset + 4] = 1.0; // r (white)
      data[offset + 5] = 1.0; // g
      data[offset + 6] = 1.0; // b
      data[offset + 7] = 0.7; // opacity (higher for better coverage)
    }

    this.device.queue.writeBuffer(
      this.propertyBuffer,
      0,
      data.buffer,
      data.byteOffset,
      data.byteLength
    );
  }

  private createPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct PositionData {
        positions: array<vec4f>,
      }

      struct CurvatureData {
        values: array<vec4f>, // (normal.xyz, scaleFactor)
      }

      struct SplatProperties {
        data: array<vec4f>, // Interleaved: [pos+radius, color+opacity, ...]
      }

      @group(0) @binding(0) var<storage, read> positions: PositionData;
      @group(0) @binding(1) var<storage, read> curvature: CurvatureData;
      @group(0) @binding(2) var<storage, read_write> properties: SplatProperties;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let index = globalId.x;
        if (index >= arrayLength(&positions.positions)) {
          return;
        }

        let pos = positions.positions[index].xyz;
        let curvatureData = curvature.values[index];
        let normal = curvatureData.xyz;
        let scaleFactor = curvatureData.w;

        // TEMPORARY: Constant radius to test
        let radius = 0.04;

        // Color based on normal for visualization
        let color = abs(normal) * 0.8 + 0.2;

        // Opacity will be controlled by Gaussian falloff in fragment shader
        // This is the base opacity multiplier (set to 1.0 for full control via Gaussian)
        let opacity = 1.0;

        // Store in interleaved format
        properties.data[index * 2u + 0u] = vec4f(pos, radius);
        properties.data[index * 2u + 1u] = vec4f(color, opacity);
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Update splat properties",
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
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
      label: "Update splat properties pipeline",
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
   * Update splat properties based on curvature data
   * Uses curvature to set radius and color
   */
  updateFromCurvature(
    commandEncoder: GPUCommandEncoder,
    positionBuffer: GPUBuffer,
    curvatureBuffer: GPUBuffer
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: curvatureBuffer } },
        { binding: 2, resource: { buffer: this.propertyBuffer } },
      ],
    });

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(this.numSplats / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  getPropertyBuffer(): GPUBuffer {
    return this.propertyBuffer;
  }

  destroy(): void {
    this.propertyBuffer.destroy();
  }
}
