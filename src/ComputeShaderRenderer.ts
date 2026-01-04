/**
 * Compute shader renderer for tile-based Gaussian splatting
 * Single compute dispatch with manual alpha blending
 */
export class ComputeShaderRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;

  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private outputTexture: GPUTexture | null = null;

  // Blit pipeline to copy rgba8unorm to presentation format
  private blitPipeline: GPURenderPipeline;
  private blitBindGroupLayout: GPUBindGroupLayout;

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
  ) {
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;

    const { pipeline, bindGroupLayout } = this.createPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;

    const blitResult = this.createBlitPipeline();
    this.blitPipeline = blitResult.pipeline;
    this.blitBindGroupLayout = blitResult.bindGroupLayout;
  }

  private createPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    // Always use rgba8unorm for compute shader output (bgra8unorm doesn't support storage)
    const shaderCode = `
      struct Uniforms {
        viewProjectionMatrix: mat4x4f,
        cameraPosition: vec3f,
        time: f32,
        tileSize: u32,
        numTilesX: u32,
        maxSplatsPerTile: u32,
      }

      struct SplatProperties {
        data: array<vec4f>, // Interleaved: [pos+radius, color+opacity, ...]
      }

      struct SplatIndices {
        indices: array<u32>,
      }

      struct CurvatureData {
        values: array<vec4f>, // (normal.xyz, scaleFactor)
      }

      struct TileList {
        count: atomic<u32>,
      }

      struct TileLists {
        lists: array<TileList>,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> splatProperties: SplatProperties;
      @group(0) @binding(2) var<storage, read> splatIndices: SplatIndices;
      @group(0) @binding(3) var<storage, read> curvatureData: CurvatureData;
      @group(0) @binding(4) var<storage, read_write> tileLists: TileLists;
      @group(0) @binding(5) var outputTexture: texture_storage_2d<rgba8unorm, write>;

      fn computeTangent(normal: vec3f) -> vec3f {
        let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
        return normalize(cross(up, normal));
      }

      fn evaluateSplat(
        splatIndex: u32,
        pixelPos: vec2f,
        viewProj: mat4x4f
      ) -> vec4f {
        // Read splat properties
        let posRadius = splatProperties.data[splatIndex * 2u + 0u];
        let colorOpacity = splatProperties.data[splatIndex * 2u + 1u];

        let worldPos = posRadius.xyz;
        let radius = posRadius.w;
        let color = colorOpacity.xyz;
        let opacity = colorOpacity.w;

        // Get normal from curvature data
        let curvature = curvatureData.values[splatIndex];
        let normal = curvature.xyz;

        // Project splat center to screen space
        let clipPos = viewProj * vec4f(worldPos, 1.0);

        // Behind camera check
        if (clipPos.w <= 0.0) {
          return vec4f(0.0);
        }

        let ndc = clipPos.xyz / clipPos.w;

        // Get texture dimensions
        let texSize = vec2f(textureDimensions(outputTexture));
        let screenCenter = vec2f(
          (ndc.x + 1.0) * 0.5 * texSize.x,
          (1.0 - ndc.y) * 0.5 * texSize.y
        );

        // Estimate screen-space radius (simple approach)
        // Project a point offset by radius
        let offsetWorld = worldPos + vec3f(radius, 0.0, 0.0);
        let offsetClip = viewProj * vec4f(offsetWorld, 1.0);
        let offsetNDC = offsetClip.xyz / offsetClip.w;
        let offsetScreen = vec2f(
          (offsetNDC.x + 1.0) * 0.5 * texSize.x,
          (1.0 - offsetNDC.y) * 0.5 * texSize.y
        );
        let screenRadius = distance(screenCenter, offsetScreen);

        if (screenRadius < 0.5) {
          return vec4f(0.0); // Too small
        }

        // Vector from splat center to pixel
        let pixelOffset = pixelPos - screenCenter;
        let pixelDist = length(pixelOffset);

        // Simple circular Gaussian with smooth falloff
        let normalizedDist = pixelDist / screenRadius;

        // Limit splat extent to avoid excessive overlap
        // Use 2 sigma cutoff (covers ~95% of Gaussian)
        if (normalizedDist > 2.0) {
          return vec4f(0.0);
        }

        // Gaussian with moderate falloff
        let sigma = 0.5;
        let gaussian = exp(-0.5 * normalizedDist * normalizedDist / (sigma * sigma));

        // Lighting
        let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
        let diffuse = max(dot(normal, lightDir), 0.0);
        let litColor = color * (0.85 + 0.15 * diffuse);

        return vec4f(litColor, gaussian);
      }

      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let pixelCoord = globalId.xy;
        let texSize = textureDimensions(outputTexture);

        // Bounds check
        if (pixelCoord.x >= texSize.x || pixelCoord.y >= texSize.y) {
          return;
        }

        // Determine which tile this pixel belongs to
        let tileX = pixelCoord.x / uniforms.tileSize;
        let tileY = pixelCoord.y / uniforms.tileSize;
        let tileIdx = tileY * uniforms.numTilesX + tileX;

        // Start with transparent black background
        var color = vec3f(0.0, 0.0, 0.0);
        var alpha = 0.0;

        let pixelPos = vec2f(f32(pixelCoord.x) + 0.5, f32(pixelCoord.y) + 0.5);

        // Get actual count for this tile
        let tileCount = atomicLoad(&tileLists.lists[tileIdx].count);
        let numSplats = min(tileCount, uniforms.maxSplatsPerTile);

        // Process splats in this tile (back-to-front order)
        let tileStartOffset = tileIdx * uniforms.maxSplatsPerTile;

        for (var i = 0u; i < numSplats; i++) {
          let splatIdx = splatIndices.indices[tileStartOffset + i];

          // Evaluate splat at this pixel
          let splatContribution = evaluateSplat(splatIdx, pixelPos, uniforms.viewProjectionMatrix);

          // Back-to-front alpha blending
          let splatAlpha = splatContribution.a;
          color = color * (1.0 - splatAlpha) + splatContribution.rgb * splatAlpha;
          alpha = alpha * (1.0 - splatAlpha) + splatAlpha;

          // Early exit if fully opaque
          if (alpha >= 0.99) {
            break;
          }
        }

        // Composite with background color to produce fully opaque result
        let backgroundColor = vec3f(0.05, 0.05, 0.1);
        let finalColor = color + backgroundColor * (1.0 - alpha);

        textureStore(outputTexture, pixelCoord, vec4f(finalColor, 1.0));
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Compute shader renderer",
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
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: "write-only",
            format: "rgba8unorm",
          },
        },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: "Compute shader renderer pipeline",
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

  private createBlitPipeline(): {
    pipeline: GPURenderPipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
      @group(0) @binding(1) var sourceSampler: sampler;

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
      }

      @vertex
      fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        var output: VertexOutput;
        // Fullscreen triangle
        let x = f32((vertexIndex & 1u) << 2u) - 1.0;
        let y = f32((vertexIndex & 2u) << 1u) - 1.0;
        output.position = vec4f(x, y, 0.0, 1.0);
        output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        return textureSample(sourceTexture, sourceSampler, input.uv);
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Blit shader",
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {},
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {},
        },
      ],
    });

    const pipeline = this.device.createRenderPipeline({
      label: "Blit pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    return { pipeline, bindGroupLayout };
  }

  private ensureOutputTexture(width: number, height: number): void {
    if (
      !this.outputTexture ||
      this.outputTexture.width !== width ||
      this.outputTexture.height !== height
    ) {
      if (this.outputTexture) {
        this.outputTexture.destroy();
      }

      this.outputTexture = this.device.createTexture({
        size: { width, height },
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }

  render(
    uniformData: Float32Array,
    splatPropertyBuffer: GPUBuffer,
    splatIndicesBuffer: GPUBuffer,
    curvatureBuffer: GPUBuffer,
    tileListsBuffer: GPUBuffer,
    tileSize: number,
    numTilesX: number,
    maxSplatsPerTile: number,
    width: number,
    height: number,
  ): void {
    this.ensureOutputTexture(width, height);

    // Create uniform buffer with tile parameters
    const computeUniformData = new ArrayBuffer(96);
    const f32View = new Float32Array(computeUniformData);
    const u32View = new Uint32Array(computeUniformData);

    // Copy viewProjection matrix (16 floats = 64 bytes)
    f32View.set(uniformData.subarray(0, 16), 0);
    // Copy camera position (3 floats)
    f32View.set(uniformData.subarray(16, 19), 16);
    // Copy time (1 float)
    f32View[19] = uniformData[19];
    // Set tile parameters
    u32View[20] = tileSize;
    u32View[21] = numTilesX;
    u32View[22] = maxSplatsPerTile;

    const computeUniformBuffer = this.device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(computeUniformBuffer, 0, computeUniformData);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: computeUniformBuffer } },
        { binding: 1, resource: { buffer: splatPropertyBuffer } },
        { binding: 2, resource: { buffer: splatIndicesBuffer } },
        { binding: 3, resource: { buffer: curvatureBuffer } },
        { binding: 4, resource: { buffer: tileListsBuffer } },
        { binding: 5, resource: this.outputTexture!.createView() },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);

    // Dispatch one thread per 8x8 tile
    const workgroupsX = Math.ceil(width / 8);
    const workgroupsY = Math.ceil(height / 8);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

    // Blit rgba8unorm output to canvas (presentation format)
    const canvasTexture = this.context.getCurrentTexture();

    // Create sampler
    const sampler = this.device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });

    const blitBindGroup = this.device.createBindGroup({
      layout: this.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: this.outputTexture!.createView() },
        { binding: 1, resource: sampler },
      ],
    });

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });

    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, blitBindGroup);
    renderPass.draw(3, 1, 0, 0); // Fullscreen triangle
    renderPass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Clean up temporary buffer
    computeUniformBuffer.destroy();
  }

  destroy(): void {
    if (this.outputTexture) {
      this.outputTexture.destroy();
    }
  }
}
