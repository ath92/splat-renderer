/**
 * Extracts camera-space depth for each splat for sorting
 */
export class DepthExtractor {
  private device: GPUDevice;
  private numSplats: number;
  private paddedSize: number;
  private depthBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private initialIndices: Uint32Array;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, numSplats: number) {
    this.device = device;
    this.numSplats = numSplats;

    // Pad to power-of-2 for bitonic sort compatibility
    this.paddedSize = Math.pow(2, Math.ceil(Math.log2(numSplats)));

    // Create depth buffer with padded size (f32 per splat)
    // Initialize padding with very positive values (will sort to end after negation)
    const depths = new Float32Array(this.paddedSize);
    for (let i = 0; i < this.paddedSize; i++) {
      depths[i] = 1e10; // Very positive, becomes very negative after negation in shader
    }

    this.depthBuffer = device.createBuffer({
      size: this.paddedSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(
      this.depthBuffer,
      0,
      depths.buffer,
      depths.byteOffset,
      depths.byteLength
    );

    // Create and store initial index buffer (0, 1, 2, ...)
    this.initialIndices = new Uint32Array(this.paddedSize);
    for (let i = 0; i < numSplats; i++) {
      this.initialIndices[i] = i;
    }
    // Fill padding with max values so they sort to the end
    for (let i = numSplats; i < this.paddedSize; i++) {
      this.initialIndices[i] = 0xFFFFFFFF; // Max u32 value
    }

    this.indexBuffer = device.createBuffer({
      size: this.paddedSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Upload initial indices
    device.queue.writeBuffer(
      this.indexBuffer,
      0,
      this.initialIndices.buffer,
      this.initialIndices.byteOffset,
      this.initialIndices.byteLength
    );

    // Create pipeline
    const { pipeline, bindGroupLayout } = this.createPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }

  private createPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct Uniforms {
        viewProjectionMatrix: mat4x4f,
        cameraPosition: vec3f,
        time: f32,
      }

      struct PositionData {
        positions: array<vec4f>,
      }

      struct DepthData {
        depths: array<f32>,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> positions: PositionData;
      @group(0) @binding(2) var<storage, read_write> depths: DepthData;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let index = globalId.x;
        if (index >= arrayLength(&positions.positions)) {
          return;
        }

        let worldPos = positions.positions[index].xyz;

        // Compute distance from camera
        let depth = distance(worldPos, uniforms.cameraPosition);

        // Store NEGATIVE depth so ascending sort gives back-to-front order
        // (farther = more negative, sorts first; closer = less negative, sorts last)
        depths.depths[index] = -depth;
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Depth extraction shader",
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
      label: "Depth extraction pipeline",
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

  extractDepths(
    commandEncoder: GPUCommandEncoder,
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer
  ): void {
    // IMPORTANT: Reset index buffer to [0, 1, 2, ...] before each sort
    // Otherwise indices from previous frame will be re-sorted, causing corruption
    this.device.queue.writeBuffer(
      this.indexBuffer,
      0,
      this.initialIndices.buffer,
      this.initialIndices.byteOffset,
      this.initialIndices.byteLength
    );

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: this.depthBuffer } },
      ],
    });

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(this.numSplats / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  getDepthBuffer(): GPUBuffer {
    return this.depthBuffer;
  }

  getIndexBuffer(): GPUBuffer {
    return this.indexBuffer;
  }

  destroy(): void {
    this.depthBuffer.destroy();
    this.indexBuffer.destroy();
  }
}
