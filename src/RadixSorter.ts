import radixSortCode from "./shaders/radix-sort.wgsl?raw";

/**
 * GPU Radix Sort for sorting splats globally by depth
 * Based on high-performance radix sort implementation
 */
export class RadixSorter {
  private device: GPUDevice;
  private numSplats: number;

  // Radix sort configuration
  private readonly RADIX_LOG2 = 8; // 8 bits per pass = 4 passes for 32-bit keys
  private readonly RADIX_SIZE = 256; // 2^8
  private readonly HISTOGRAM_WG_SIZE = 256;
  private readonly HISTOGRAM_SG_SIZE = 32; // Subgroup size
  private readonly HISTOGRAM_BLOCK_ROWS = 15;
  private readonly SCATTER_BLOCK_ROWS = 15;
  private readonly SCATTER_WG_SIZE = 256;
  private readonly PREFIX_WG_SIZE = 128;

  // Buffers
  private keysBuffer: GPUBuffer;
  private keysBBuffer: GPUBuffer;
  private payloadBuffer: GPUBuffer; // Original indices
  private payloadBBuffer: GPUBuffer;
  private histogramsBuffer: GPUBuffer;
  private infosBuffer: GPUBuffer;

  // Pipelines
  private zeroHistogramsPipeline!: GPUComputePipeline;
  private calculateHistogramPipeline!: GPUComputePipeline;
  private prefixHistogramPipeline!: GPUComputePipeline;
  private scatterEvenPipeline!: GPUComputePipeline;
  private scatterOddPipeline!: GPUComputePipeline;

  private bindGroupLayout!: GPUBindGroupLayout;
  private tempBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, numSplats: number) {
    this.device = device;
    this.numSplats = numSplats;

    // Pad to multiple of block size
    const blockKvs = this.HISTOGRAM_WG_SIZE * this.HISTOGRAM_BLOCK_ROWS;
    const paddedSize = Math.ceil(numSplats / blockKvs) * blockKvs;

    // Create buffers
    this.keysBuffer = device.createBuffer({
      size: paddedSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.keysBBuffer = device.createBuffer({
      size: paddedSize * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.payloadBuffer = device.createBuffer({
      size: paddedSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.payloadBBuffer = device.createBuffer({
      size: paddedSize * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Calculate histogram buffer size
    const scatterBlockKvs = this.SCATTER_WG_SIZE * this.SCATTER_BLOCK_ROWS;
    const scatterBlocksRu = Math.ceil(numSplats / scatterBlockKvs);
    const keyvalSize = 4; // 4 passes for 32-bit keys
    const histoSize = this.RADIX_SIZE;
    const histogramsSize =
      (keyvalSize + scatterBlocksRu - 1) * histoSize * 4;

    this.histogramsBuffer = device.createBuffer({
      size: histogramsSize,
      usage: GPUBufferUsage.STORAGE,
    });

    // Info buffer
    const infoData = new Uint32Array([
      numSplats, // keys_size
      paddedSize, // padded_size
      4, // passes (32 bits / 8 bits per pass)
      0, // even_pass
      1, // odd_pass
    ]);

    this.infosBuffer = device.createBuffer({
      size: infoData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(this.infosBuffer.getMappedRange()).set(infoData);
    this.infosBuffer.unmap();

    // Create pipelines
    this.createPipelines();
  }

  private createPipelines(): void {
    // Calculate rs_mem_dwords for scatter shader
    const rsPrefixSweeps = 3;
    const rsPrefixSize =
      ((this.RADIX_SIZE / this.PREFIX_WG_SIZE) >> rsPrefixSweeps) + 1;
    const rsMemDwords =
      this.RADIX_SIZE +
      this.RADIX_SIZE +
      rsPrefixSize +
      this.SCATTER_WG_SIZE * this.SCATTER_BLOCK_ROWS;
    const rsMemSweep0Offset = 0;
    const rsMemSweep1Offset = rsPrefixSize >> 2;
    const rsMemSweep2Offset = rsMemSweep1Offset + (rsPrefixSize >> 4);

    // Replace placeholders in shader code
    let shaderCode = radixSortCode
      .replace(/{histogram_sg_size}/g, this.HISTOGRAM_SG_SIZE.toString())
      .replace(/{histogram_wg_size}/g, this.HISTOGRAM_WG_SIZE.toString())
      .replace(/{scatter_wg_size}/g, this.SCATTER_WG_SIZE.toString())
      .replace(/{prefix_wg_size}/g, this.PREFIX_WG_SIZE.toString());

    // Prepend constants
    const constants = `
const rs_radix_log2 = ${this.RADIX_LOG2}u;
const rs_radix_size = ${this.RADIX_SIZE}u;
const rs_keyval_size = 4u;
const rs_histogram_block_rows = ${this.HISTOGRAM_BLOCK_ROWS}u;
const rs_scatter_block_rows = ${this.SCATTER_BLOCK_ROWS}u;
const histogram_sg_size = ${this.HISTOGRAM_SG_SIZE}u;
const histogram_wg_size = ${this.HISTOGRAM_WG_SIZE}u;
const rs_mem_dwords = ${rsMemDwords}u;
const rs_mem_sweep_0_offset = ${rsMemSweep0Offset}u;
const rs_mem_sweep_1_offset = ${rsMemSweep1Offset}u;
const rs_mem_sweep_2_offset = ${rsMemSweep2Offset}u;
`;

    shaderCode = constants + shaderCode;

    const shaderModule = this.device.createShaderModule({
      label: "Radix sort shader",
      code: shaderCode,
    });

    // Create bind group layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Create pipelines
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.zeroHistogramsPipeline = this.device.createComputePipeline({
      label: "Zero histograms",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "zero_histograms" },
    });

    this.calculateHistogramPipeline = this.device.createComputePipeline({
      label: "Calculate histogram",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "calculate_histogram" },
    });

    this.prefixHistogramPipeline = this.device.createComputePipeline({
      label: "Prefix histogram",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "prefix_histogram" },
    });

    this.scatterEvenPipeline = this.device.createComputePipeline({
      label: "Scatter even",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "scatter_even" },
    });

    this.scatterOddPipeline = this.device.createComputePipeline({
      label: "Scatter odd",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "scatter_odd" },
    });
  }

  /**
   * Sort projected splats by depth (far to near)
   * @param commandEncoder Command encoder
   * @param projectedBuffer Buffer containing ProjectedSplat data
   */
  async sort(
    commandEncoder: GPUCommandEncoder,
    projectedBuffer: GPUBuffer
  ): Promise<Uint32Array> {
    // Copy depths from projected buffer to keys buffer
    // We need to read back the depths first
    const readbackBuffer = this.device.createBuffer({
      size: this.numSplats * 32, // ProjectedSplat size
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(
      projectedBuffer,
      0,
      readbackBuffer,
      0,
      this.numSplats * 32
    );

    this.device.queue.submit([commandEncoder.finish()]);

    // Read back projected data
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const projectedData = new Float32Array(readbackBuffer.getMappedRange());

    // Extract depths and convert to sortable integers
    // We want far-to-near (descending), so negate depths
    const keys = new Uint32Array(this.numSplats);
    const payload = new Uint32Array(this.numSplats);

    for (let i = 0; i < this.numSplats; i++) {
      const depth = projectedData[i * 8 + 4]; // depth is at offset 4 in ProjectedSplat
      // Convert float to sortable uint (flip bits for negative, flip sign bit for positive)
      const floatBits = new Uint32Array(new Float32Array([depth]).buffer)[0];
      const mask = (floatBits >> 31) === 1 ? 0xffffffff : 0x80000000;
      keys[i] = floatBits ^ mask;
      payload[i] = i; // Original index
    }

    readbackBuffer.unmap();
    readbackBuffer.destroy();

    // Upload keys and payload
    this.device.queue.writeBuffer(this.keysBuffer, 0, keys);
    this.device.queue.writeBuffer(this.payloadBuffer, 0, payload);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.infosBuffer } },
        { binding: 1, resource: { buffer: this.histogramsBuffer } },
        { binding: 2, resource: { buffer: this.keysBuffer } },
        { binding: 3, resource: { buffer: this.keysBBuffer } },
        { binding: 4, resource: { buffer: this.payloadBuffer } },
        { binding: 5, resource: { buffer: this.payloadBBuffer } },
      ],
    });

    const blockKvs = this.HISTOGRAM_WG_SIZE * this.HISTOGRAM_BLOCK_ROWS;
    const numBlocks = Math.ceil(this.numSplats / blockKvs);

    // Run radix sort passes
    const commandEncoder2 = this.device.createCommandEncoder();

    // Zero histograms
    const computePass1 = commandEncoder2.beginComputePass();
    computePass1.setPipeline(this.zeroHistogramsPipeline);
    computePass1.setBindGroup(0, bindGroup);
    computePass1.dispatchWorkgroups(Math.max(1, numBlocks));
    computePass1.end();

    // Calculate histogram
    const computePass2 = commandEncoder2.beginComputePass();
    computePass2.setPipeline(this.calculateHistogramPipeline);
    computePass2.setBindGroup(0, bindGroup);
    computePass2.dispatchWorkgroups(numBlocks);
    computePass2.end();

    // Prefix histogram (4 passes for 32-bit keys)
    const computePass3 = commandEncoder2.beginComputePass();
    computePass3.setPipeline(this.prefixHistogramPipeline);
    computePass3.setBindGroup(0, bindGroup);
    computePass3.dispatchWorkgroups(4); // 4 passes
    computePass3.end();

    this.device.queue.submit([commandEncoder2.finish()]);

    // Scatter passes (alternating even/odd)
    const scatterBlockKvs = this.SCATTER_WG_SIZE * this.SCATTER_BLOCK_ROWS;
    const scatterBlocks = Math.ceil(this.numSplats / scatterBlockKvs);

    for (let pass = 0; pass < 4; pass++) {
      const commandEncoder3 = this.device.createCommandEncoder();
      const computePass = commandEncoder3.beginComputePass();

      if (pass % 2 === 0) {
        computePass.setPipeline(this.scatterEvenPipeline);
      } else {
        computePass.setPipeline(this.scatterOddPipeline);
      }

      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(scatterBlocks);
      computePass.end();

      this.device.queue.submit([commandEncoder3.finish()]);
    }

    // Read back sorted payload (original indices in sorted order)
    const resultBuffer = this.device.createBuffer({
      size: this.numSplats * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder4 = this.device.createCommandEncoder();
    commandEncoder4.copyBufferToBuffer(
      this.payloadBuffer,
      0,
      resultBuffer,
      0,
      this.numSplats * 4
    );
    this.device.queue.submit([commandEncoder4.finish()]);

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const sortedIndices = new Uint32Array(resultBuffer.getMappedRange()).slice();
    resultBuffer.unmap();
    resultBuffer.destroy();

    return sortedIndices;
  }

  cleanupTempBuffers(): void {
    this.tempBuffers.forEach(buffer => buffer.destroy());
    this.tempBuffers = [];
  }

  destroy(): void {
    this.keysBuffer.destroy();
    this.keysBBuffer.destroy();
    this.payloadBuffer.destroy();
    this.payloadBBuffer.destroy();
    this.histogramsBuffer.destroy();
    this.infosBuffer.destroy();
    this.cleanupTempBuffers();
  }
}
