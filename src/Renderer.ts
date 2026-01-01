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
  private pointVertexBuffer: GPUBuffer;

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
    this.pointPipeline = this.createPointPipeline();
    this.pointVertexBuffer = this.createPointVertexBuffer();
  }

  private createPointPipeline(): GPURenderPipeline {
    const pointShaderCode = `
      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
      }

      struct PointData {
        @location(0) center: vec2f,
        @location(1) distance: f32,
      }

      @vertex
      fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        point: PointData
      ) -> VertexOutput {
        var output: VertexOutput;

        // Create a small quad for each point
        let size = 0.01;
        let offset = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );

        let localPos = offset[vertexIndex] * size;
        // Flip y to match SDF coordinate system
        let flippedCenter = vec2f(point.center.x, -point.center.y);
        output.position = vec4f(flippedCenter + localPos, 0.0, 1.0);

        // Color based on distance (red = far, green = close, blue = on surface)
        let normalizedDist = clamp(abs(point.distance) * 5.0, 0.0, 1.0);
        if (point.distance > 0.01) {
          output.color = vec3f(1.0, 0.0, 0.0); // Red: far from surface
        } else if (abs(point.distance) < 0.01) {
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

    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: pointShaderModule,
        entryPoint: "vertexMain",
        buffers: [
          {
            arrayStride: 3 * 4, // vec2f center + f32 distance
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x2" }, // center
              { shaderLocation: 1, offset: 8, format: "float32" }, // distance
            ],
          },
        ],
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
  }

  private createPointVertexBuffer(): GPUBuffer {
    // This will be updated each frame
    return this.device.createBuffer({
      size: this.numPoints * 3 * 4, // 3 floats per point (x, y, distance)
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  render(
    useGradientMode: boolean,
    points: Float32Array,
    gradientResults: Float32Array,
    numPoints: number
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

    // Render points
    this.renderPoints(renderPass, points, gradientResults, numPoints);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  private renderPoints(
    renderPass: GPURenderPassEncoder,
    points: Float32Array,
    gradientResults: Float32Array,
    numPoints: number
  ): void {
    // Prepare point instance data (center.x, center.y, distance)
    const pointData = new Float32Array(numPoints * 3);
    for (let i = 0; i < numPoints; i++) {
      pointData[i * 3] = points[i * 2]; // x
      pointData[i * 3 + 1] = points[i * 2 + 1]; // y
      pointData[i * 3 + 2] = gradientResults[i * 4]; // distance
    }

    // Upload point data
    this.device.queue.writeBuffer(this.pointVertexBuffer, 0, pointData);

    // Draw points
    renderPass.setPipeline(this.pointPipeline);
    renderPass.setVertexBuffer(0, this.pointVertexBuffer);
    renderPass.draw(6, numPoints); // 6 vertices per quad, N instances
  }

  destroy(): void {
    this.pointVertexBuffer.destroy();
  }
}
