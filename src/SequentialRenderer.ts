/**
 * Simple sequential renderer - one draw call per splat
 * Guarantees perfect back-to-front ordering but slower than tile-based
 */
export class SequentialRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private numSplats: number;

  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private depthTexture: GPUTexture | null = null;

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
    numSplats: number
  ) {
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;
    this.numSplats = numSplats;

    const { pipeline, bindGroupLayout } = this.createPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }

  private createPipeline(): {
    pipeline: GPURenderPipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const shaderCode = `
      struct Uniforms {
        viewProjectionMatrix: mat4x4f,
        cameraPosition: vec3f,
        time: f32,
        splatIndex: u32, // Which splat to render
      }

      struct SplatProperties {
        data: array<vec4f>, // Interleaved: [pos+radius, color+opacity, ...]
      }

      struct SortedIndices {
        indices: array<u32>,
      }

      struct CurvatureData {
        values: array<vec4f>, // (normal.xyz, scaleFactor)
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> splatProperties: SplatProperties;
      @group(0) @binding(2) var<storage, read> sortedIndices: SortedIndices;
      @group(0) @binding(3) var<storage, read> curvatureData: CurvatureData;

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
        @location(1) uv: vec2f,
        @location(2) opacity: f32,
        @location(3) normal: vec3f,
      }

      fn computeTangent(normal: vec3f) -> vec3f {
        let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
        return normalize(cross(up, normal));
      }

      @vertex
      fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32
      ) -> VertexOutput {
        var output: VertexOutput;

        // Get the splat index from uniform (which splat we're rendering)
        let splatIndex = sortedIndices.indices[uniforms.splatIndex];

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

        // Construct tangent frame
        let tangent = computeTangent(normal);
        let bitangent = cross(normal, tangent);

        // Quad corners
        let quadOffset = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );

        let offset2D = quadOffset[vertexIndex];

        // Build 3D offset using tangent frame
        let worldOffset =
          tangent * offset2D.x * radius +
          bitangent * offset2D.y * radius;

        let finalWorldPos = worldPos + worldOffset;

        // Project to clip space
        output.position = uniforms.viewProjectionMatrix * vec4f(finalWorldPos, 1.0);
        output.uv = offset2D;
        output.color = color;
        output.opacity = opacity;
        output.normal = normal;

        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        let dist2 = dot(input.uv, input.uv);

        if (dist2 > 1.0) {
          discard;
        }

        let sigma = 0.4;
        let gaussian = exp(-0.5 * dist2 / (sigma * sigma));

        let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
        let diffuse = max(dot(input.normal, lightDir), 0.0);
        let litColor = input.color * (0.85 + 0.15 * diffuse);

        let finalAlpha = gaussian;

        return vec4f(litColor, finalAlpha);
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Sequential renderer shader",
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    const pipeline = this.device.createRenderPipeline({
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
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "always", // Always pass depth test for transparent objects
      },
    });

    return { pipeline, bindGroupLayout };
  }

  private ensureDepthTexture(width: number, height: number): void {
    if (
      !this.depthTexture ||
      this.depthTexture.width !== width ||
      this.depthTexture.height !== height
    ) {
      if (this.depthTexture) {
        this.depthTexture.destroy();
      }

      this.depthTexture = this.device.createTexture({
        size: { width, height },
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }

  render(
    uniformData: Float32Array,
    splatPropertyBuffer: GPUBuffer,
    sortedIndexBuffer: GPUBuffer,
    curvatureBuffer: GPUBuffer,
    width: number,
    height: number
  ): void {
    this.ensureDepthTexture(width, height);

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture!.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(this.pipeline);

    // Collect buffers to destroy after submit
    const buffersToDestroy: GPUBuffer[] = [];

    // Render each splat sequentially in back-to-front order
    for (let i = 0; i < this.numSplats; i++) {
      // Update uniform with current splat index
      const splatUniformData = new ArrayBuffer(96);
      const f32View = new Float32Array(splatUniformData);
      const u32View = new Uint32Array(splatUniformData);

      // Copy viewProjection matrix (16 floats = 64 bytes)
      f32View.set(uniformData.subarray(0, 16), 0);
      // Copy camera position (3 floats)
      f32View.set(uniformData.subarray(16, 19), 16);
      // Copy time (1 float)
      f32View[19] = uniformData[19];
      // Set splat index (u32 at byte offset 80)
      u32View[20] = i;

      const splatUniformBuffer = this.device.createBuffer({
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint32Array(splatUniformBuffer.getMappedRange()).set(new Uint32Array(splatUniformData));
      splatUniformBuffer.unmap();

      const bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: splatUniformBuffer } },
          { binding: 1, resource: { buffer: splatPropertyBuffer } },
          { binding: 2, resource: { buffer: sortedIndexBuffer } },
          { binding: 3, resource: { buffer: curvatureBuffer } },
        ],
      });

      renderPass.setBindGroup(0, bindGroup);
      renderPass.draw(6, 1, 0, 0); // 6 vertices, 1 instance

      // Mark for cleanup after submit
      buffersToDestroy.push(splatUniformBuffer);
    }

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Clean up uniform buffers after submit
    buffersToDestroy.forEach(buffer => buffer.destroy());
  }

  destroy(): void {
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }
  }
}
