/**
 * GPU Bitonic Sort for sorting splats by depth
 * Sorts both depth values and corresponding indices
 */
export class BitonicSorter {
  private device: GPUDevice;
  private paddedSize: number;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private uniformBuffer: GPUBuffer;

  constructor(device: GPUDevice, numElements: number) {
    this.device = device;

    // Bitonic sort requires power-of-2 size, so pad if needed
    this.paddedSize = this.nextPowerOf2(numElements);

    const { pipeline, bindGroupLayout } = this.createPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;

    // Create uniform buffer for sort parameters (j, k)
    this.uniformBuffer = device.createBuffer({
      size: 8, // 2 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private nextPowerOf2(n: number): number {
    return Math.pow(2, Math.ceil(Math.log2(n)));
  }

  private createPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct SortParams {
        j: u32,
        k: u32,
      }

      struct DepthData {
        depths: array<f32>,
      }

      struct IndexData {
        indices: array<u32>,
      }

      @group(0) @binding(0) var<uniform> params: SortParams;
      @group(0) @binding(1) var<storage, read_write> depths: DepthData;
      @group(0) @binding(2) var<storage, read_write> indices: IndexData;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let i = globalId.x;
        let arraySize = arrayLength(&depths.depths);

        if (i >= arraySize) {
          return;
        }

        let j = params.j;
        let k = params.k;

        // XOR-based bitonic sort indexing
        let i_XOR_j = i ^ j;

        // Only process if this is the lower index of the pair
        if (i_XOR_j <= i) {
          return;
        }

        // Get values
        let depth1 = depths.depths[i];
        let depth2 = depths.depths[i_XOR_j];

        // Determine swap direction based on k
        // For ascending order (negative depths for back-to-front):
        // swap0: if (i & k) == 0 and depth1 > depth2
        // swap1: if (i & k) != 0 and depth1 < depth2
        let swap0 = (i & k) == 0u && depth1 > depth2;
        let swap1 = (i & k) != 0u && depth1 < depth2;

        if (swap0 || swap1) {
          // Swap depths
          depths.depths[i] = depth2;
          depths.depths[i_XOR_j] = depth1;

          // Swap indices
          let tempIdx = indices.indices[i];
          indices.indices[i] = indices.indices[i_XOR_j];
          indices.indices[i_XOR_j] = tempIdx;
        }
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Bitonic sort shader",
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
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Bitonic sort pipeline",
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
   * Sort depth buffer and index buffer using XOR-based bitonic sort
   * Ascending order with negative depths gives back-to-front rendering
   */
  sort(
    commandEncoder: GPUCommandEncoder,
    depthBuffer: GPUBuffer,
    indexBuffer: GPUBuffer
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: depthBuffer } },
        { binding: 2, resource: { buffer: indexBuffer } },
      ],
    });

    // XOR-based bitonic sort algorithm
    // Outer loop: k doubles each iteration (2, 4, 8, ...)
    for (let k = 2; k <= this.paddedSize; k *= 2) {
      // Inner loop: j starts at k/2 and halves each iteration
      for (let j = k / 2; j > 0; j /= 2) {
        // Update uniform buffer with j and k parameters
        const params = new Uint32Array([j, k]);

        this.device.queue.writeBuffer(
          this.uniformBuffer,
          0,
          params.buffer,
          params.byteOffset,
          params.byteLength
        );

        // Dispatch compute pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.pipeline);
        computePass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(this.paddedSize / 64);
        computePass.dispatchWorkgroups(workgroupCount);
        computePass.end();
      }
    }
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}
