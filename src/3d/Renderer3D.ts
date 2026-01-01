export class Renderer3D {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private numPoints: number;

  // Point rendering pipeline
  private pointPipeline: GPURenderPipeline;
  private pointBindGroupLayout: GPUBindGroupLayout;
  private indirectBuffer: GPUBuffer;
  private depthTexture: GPUTexture | null = null;

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
    numPoints: number
  ) {
    this.numPoints = numPoints;
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;

    // Create point visualization shader and pipeline
    const { pipeline, bindGroupLayout } = this.createPointPipeline();
    this.pointPipeline = pipeline;
    this.pointBindGroupLayout = bindGroupLayout;

    // Create indirect draw buffer
    this.indirectBuffer = this.createIndirectBuffer();
  }

  private createPointPipeline(): {
    pipeline: GPURenderPipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const pointShaderCode = `
      struct PositionData {
        positions: array<vec3f>,
      }

      struct GradientData {
        results: array<vec4f>, // (distance, gradient.x, gradient.y, gradient.z)
      }

      struct Uniforms {
        viewProjectionMatrix: mat4x4f,
        cameraPosition: vec3f,
        time: f32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> positions: PositionData;
      @group(0) @binding(2) var<storage, read> gradients: GradientData;

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
        @location(1) uv: vec2f,
      }

      @vertex
      fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        @builtin(instance_index) instanceIndex: u32
      ) -> VertexOutput {
        var output: VertexOutput;

        // Get 3D position from storage buffer
        let worldPos = positions.positions[instanceIndex];

        // Project to clip space
        let clipPos = uniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);

        // Create billboard quad in screen space
        let quadOffset = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );
        let pointSize = 0.01; // Screen-space size
        let screenOffset = quadOffset[vertexIndex] * pointSize;

        output.position = vec4f(
          clipPos.xy + screenOffset * clipPos.w,
          clipPos.z,
          clipPos.w
        );

        // UV coordinates for fragment shader
        output.uv = quadOffset[vertexIndex];

        // Color based on distance
        let distance = gradients.results[instanceIndex].x;

        if (distance > 0.05) {
          output.color = vec3f(1.0, 0.0, 0.0); // Red: far from surface
        } else if (abs(distance) < 0.01) {
          output.color = vec3f(0.0, 0.0, 1.0); // Blue: on surface
        } else {
          output.color = vec3f(0.0, 1.0, 0.0); // Green: close to surface
        }

        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        // Draw smooth circle within quad
        let dist = length(input.uv);
        let alpha = 1.0 - smoothstep(0.8, 1.0, dist);

        if (alpha < 0.1) {
          discard;
        }

        return vec4f(input.color, alpha);
      }
    `;

    const pointShaderModule = this.device.createShaderModule({
      label: "Point 3D shader",
      code: pointShaderCode,
    });

    // Create bind group layout for point rendering
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
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
      ],
    });

    const pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: pointShaderModule,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: pointShaderModule,
        entryPoint: "fragmentMain",
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    return { pipeline, bindGroupLayout };
  }

  private createIndirectBuffer(): GPUBuffer {
    // Indirect draw arguments: vertexCount, instanceCount, firstVertex, firstInstance
    const indirectArray = new Uint32Array([
      6, // vertexCount (6 vertices per quad)
      this.numPoints, // instanceCount
      0, // firstVertex
      0, // firstInstance
    ]);

    const buffer = this.device.createBuffer({
      size: indirectArray.byteLength,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(
      buffer,
      0,
      indirectArray.buffer,
      indirectArray.byteOffset,
      indirectArray.byteLength
    );

    return buffer;
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
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer,
    gradientBuffer: GPUBuffer,
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

    // Render points using indirect rendering
    this.renderPoints(renderPass, uniformBuffer, positionBuffer, gradientBuffer);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  private renderPoints(
    renderPass: GPURenderPassEncoder,
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer,
    gradientBuffer: GPUBuffer
  ): void {
    // Create bind group for point rendering
    const pointBindGroup = this.device.createBindGroup({
      layout: this.pointBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: gradientBuffer } },
      ],
    });

    // Draw points using indirect rendering
    renderPass.setPipeline(this.pointPipeline);
    renderPass.setBindGroup(0, pointBindGroup);
    renderPass.drawIndirect(this.indirectBuffer, 0);
  }

  destroy(): void {
    this.indirectBuffer.destroy();
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }
  }
}
