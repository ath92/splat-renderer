/**
 * Sorts splats within each tile by depth (back-to-front)
 * Uses simple insertion sort in shared memory since tiles have relatively few splats
 */
export class PerTileSorter {
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
    const shaderCode = `
      struct ProjectedSplat {
        screenBoundsMin: vec2f,
        screenBoundsMax: vec2f,
        depth: f32,
        originalIndex: u32,
        _padding: vec2f,
      }

      struct ProjectedSplats {
        splats: array<ProjectedSplat>,
      }

      struct TileList {
        count: atomic<u32>,
      }

      struct TileLists {
        lists: array<TileList>,
      }

      struct TileOffsets {
        offsets: array<u32>,
      }

      struct SplatIndices {
        indices: array<u32>,
      }

      struct SortParams {
        maxSplatsPerTile: u32,
      }

      @group(0) @binding(0) var<uniform> params: SortParams;
      @group(0) @binding(1) var<storage, read> projectedSplats: ProjectedSplats;
      @group(0) @binding(2) var<storage, read_write> tileLists: TileLists;
      @group(0) @binding(3) var<storage, read> tileOffsets: TileOffsets;
      @group(0) @binding(4) var<storage, read_write> splatIndices: SplatIndices;

      // Shared memory for sorting (max 2048 splats per tile)
      var<workgroup> sharedIndices: array<u32, 2048>;
      var<workgroup> sharedDepths: array<f32, 2048>;

      @compute @workgroup_size(256)
      fn main(
        @builtin(workgroup_id) workgroupId: vec3u,
        @builtin(local_invocation_id) localId: vec3u
      ) {
        let tileIdx = workgroupId.x;
        let localThreadIdx = localId.x;

        // Get offset and count for this tile
        let tileOffset = tileOffsets.offsets[tileIdx];
        let count = atomicLoad(&tileLists.lists[tileIdx].count);
        let numSplats = min(count, params.maxSplatsPerTile);

        // Load data into shared memory from packed buffer
        // Use maxSplatsPerTile for uniform control flow
        for (var i = localThreadIdx; i < params.maxSplatsPerTile; i += 256u) {
          if (i < numSplats) {
            let globalIdx = tileOffset + i;
            let splatIdx = splatIndices.indices[globalIdx];
            sharedIndices[i] = splatIdx;
            sharedDepths[i] = projectedSplats.splats[splatIdx].depth;
          } else {
            // Fill with sentinel values for padding
            sharedIndices[i] = 0u;
            sharedDepths[i] = -1e10; // Very small so they sort to end in descending order
          }
        }

        workgroupBarrier();

        // Simple parallel bubble sort
        // Use maxSplatsPerTile for uniform control flow
        for (var iteration = 0u; iteration < params.maxSplatsPerTile; iteration++) {
          for (var i = localThreadIdx; i < params.maxSplatsPerTile - 1u; i += 256u) {
            // Compare-and-swap adjacent elements if out of order
            // Back-to-front: larger depth (further) should come first
            if (sharedDepths[i] < sharedDepths[i + 1u]) {
              // Swap depths
              let tempDepth = sharedDepths[i];
              sharedDepths[i] = sharedDepths[i + 1u];
              sharedDepths[i + 1u] = tempDepth;

              // Swap indices
              let tempIdx = sharedIndices[i];
              sharedIndices[i] = sharedIndices[i + 1u];
              sharedIndices[i + 1u] = tempIdx;
            }
          }
          workgroupBarrier();
        }

        // Write sorted data back to packed buffer
        for (var i = localThreadIdx; i < params.maxSplatsPerTile; i += 256u) {
          let globalIdx = tileOffset + i;
          splatIndices.indices[globalIdx] = sharedIndices[i];
        }
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Per-tile sort shader",
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
          buffer: { type: "storage" }, // Must be read_write for atomics
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" }, // tileOffsets
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }, // splatIndices
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Per-tile sort pipeline",
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

  sort(
    commandEncoder: GPUCommandEncoder,
    projectedBuffer: GPUBuffer,
    tileListsBuffer: GPUBuffer,
    tileOffsetsBuffer: GPUBuffer,
    splatIndicesBuffer: GPUBuffer,
    numTiles: number,
    maxSplatsPerTile: number
  ): void {
    // Create params buffer
    const params = new Uint32Array([maxSplatsPerTile]);
    const paramsBuffer = this.device.createBuffer({
      size: params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(paramsBuffer.getMappedRange()).set(params);
    paramsBuffer.unmap();

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: projectedBuffer } },
        { binding: 2, resource: { buffer: tileListsBuffer } },
        { binding: 3, resource: { buffer: tileOffsetsBuffer } },
        { binding: 4, resource: { buffer: splatIndicesBuffer } },
      ],
    });

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    // One workgroup per tile
    computePass.dispatchWorkgroups(numTiles);
    computePass.end();

    // Save buffer for cleanup after submit
    this.tempBuffers.push(paramsBuffer);
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach(buffer => buffer.destroy());
    this.tempBuffers = [];
  }

  destroy(): void {
    this.cleanupTempBuffers();
  }
}
