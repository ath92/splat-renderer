export class Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private numPoints: number;

  // SDF pipelines
  private sdfPipeline: GPURenderPipeline;
  private sdfGradientPipeline: GPURenderPipeline;

  // Point rendering pipeline
  private pointPipeline: GPURenderPipeline;
  private pointBindGroupLayout: GPUBindGroupLayout;
  private indirectBuffer: GPUBuffer;

  // Bind groups
  private sdfBindGroup: GPUBindGroup;

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
    sdfShaderCode: string,
    sdfGradientShaderCode: string,
    uniformBuffer: GPUBuffer,
    numPoints: number
  ) {
    this.numPoints = numPoints;
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;

    // Create shader modules
    const sdfShaderModule = device.createShaderModule({
      label: "SDF shader",
      code: sdfShaderCode,
    });

    const sdfGradientShaderModule = device.createShaderModule({
      label: "SDF gradient shader",
      code: sdfGradientShaderCode,
    });

    // Create bind group layout for SDF
    const sdfBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    // Create bind group for SDF
    this.sdfBindGroup = device.createBindGroup({
      layout: sdfBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    // Create pipeline layout
    const sdfPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [sdfBindGroupLayout],
    });

    // Create SDF render pipeline
    this.sdfPipeline = device.createRenderPipeline({
      layout: sdfPipelineLayout,
      vertex: {
        module: sdfShaderModule,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: sdfShaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    // Create SDF gradient render pipeline
    this.sdfGradientPipeline = device.createRenderPipeline({
      layout: sdfPipelineLayout,
      vertex: {
        module: sdfGradientShaderModule,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: sdfGradientShaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

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
        positions: array<vec2f>,
      }

      struct GradientData {
        results: array<vec4f>, // (distance, gradient.x, gradient.y, padding)
      }

      @group(0) @binding(0) var<storage, read> positions: PositionData;
      @group(0) @binding(1) var<storage, read> gradients: GradientData;

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
      }

      @vertex
      fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        @builtin(instance_index) instanceIndex: u32
      ) -> VertexOutput {
        var output: VertexOutput;

        // Get point data from storage buffers
        let center = positions.positions[instanceIndex];
        let distance = gradients.results[instanceIndex].x;

        // Create a small quad for each point
        let size = 0.01;
        let offset = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );

        let localPos = offset[vertexIndex] * size;
        // Flip y to match SDF coordinate system
        let flippedCenter = vec2f(center.x, -center.y);
        output.position = vec4f(flippedCenter + localPos, 0.0, 1.0);

        // Color based on distance (red = far, green = close, blue = on surface)
        if (distance > 0.01) {
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
        return vec4f(input.color, 1.0);
      }
    `;

    const pointShaderModule = this.device.createShaderModule({
      label: "Point shader",
      code: pointShaderCode,
    });

    // Create bind group layout for point rendering
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
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

  render(
    useGradientMode: boolean,
    positionBuffer: GPUBuffer,
    gradientBuffer: GPUBuffer
  ): void {
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    // Render SDF
    renderPass.setPipeline(
      useGradientMode ? this.sdfGradientPipeline : this.sdfPipeline
    );
    renderPass.setBindGroup(0, this.sdfBindGroup);
    renderPass.draw(3);

    // Render points using indirect rendering
    this.renderPoints(renderPass, positionBuffer, gradientBuffer);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  private renderPoints(
    renderPass: GPURenderPassEncoder,
    positionBuffer: GPUBuffer,
    gradientBuffer: GPUBuffer
  ): void {
    // Create bind group for point rendering
    const pointBindGroup = this.device.createBindGroup({
      layout: this.pointBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: gradientBuffer } },
      ],
    });

    // Draw points using indirect rendering
    renderPass.setPipeline(this.pointPipeline);
    renderPass.setBindGroup(0, pointBindGroup);
    renderPass.drawIndirect(this.indirectBuffer, 0);
  }

  destroy(): void {
    this.indirectBuffer.destroy();
  }
}
