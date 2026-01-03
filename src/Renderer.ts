export class Renderer {
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
        positions: array<vec4f>, // vec4 for proper alignment in storage buffers
      }

      struct GradientData {
        results: array<vec4f>, // (distance, gradient.x, gradient.y, gradient.z)
      }

      struct ScaleFactors {
        values: array<f32>,
      }

      struct Uniforms {
        viewProjectionMatrix: mat4x4f,
        cameraPosition: vec3f,
        time: f32,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> positions: PositionData;
      @group(0) @binding(2) var<storage, read> gradients: GradientData;
      @group(0) @binding(3) var<storage, read> scaleFactors: ScaleFactors;

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
        @location(1) uv: vec2f,
        @location(2) normal: vec3f,
      }

      fn computeTangent(normal: vec3f) -> vec3f {
        // Pick an axis least aligned with normal to avoid degeneracy
        let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
        return normalize(cross(up, normal));
      }

      @vertex
      fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        @builtin(instance_index) instanceIndex: u32
      ) -> VertexOutput {
        var output: VertexOutput;

        // Get 3D position from storage buffer (vec4, use .xyz)
        let worldPos = positions.positions[instanceIndex].xyz;

        // Extract surface normal from gradient
        let gradientData = gradients.results[instanceIndex];
        let distance = gradientData.x;
        let gradient = gradientData.yzw;
        let normal = normalize(gradient);

        // Construct tangent frame aligned with surface
        let tangent = computeTangent(normal);
        let bitangent = cross(normal, tangent);

        // Quad corners in 2D
        let quadOffset = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );

        // Get curvature-based scale factor for this point
        let scaleFactor = scaleFactors.values[instanceIndex];

        // Size parameters in world space
        let tangentScale = 0.025 * scaleFactor;    // Width along surface
        let bitangentScale = 0.025 * scaleFactor;  // Height along surface
        let normalScale = 0.0;                     // Thickness (0 = flat)

        let offset2D = quadOffset[vertexIndex];

        // Build 3D offset in world space using tangent frame
        let worldOffset =
          tangent * offset2D.x * tangentScale +
          bitangent * offset2D.y * bitangentScale +
          normal * normalScale;

        let finalWorldPos = worldPos + worldOffset;

        // Project to clip space
        output.position = uniforms.viewProjectionMatrix * vec4f(finalWorldPos, 1.0);

        // UV coordinates for fragment shader
        output.uv = offset2D;

        // Pass normal for lighting
        output.normal = normal;

        // Color based on surface normal direction
        output.color = normal * 0.5 + 0.5;

        return output;
      }

      @fragment
      fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        // Simple directional lighting based on normal
        let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
        let diffuse = max(dot(input.normal, lightDir), 0.0);

        // Combine normal color with lighting
        let baseColor = input.color;
        let litColor = baseColor * (0.3 + 0.7 * diffuse); // Ambient + diffuse

        return vec4f(litColor, 1.0);
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
        module: pointShaderModule,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: pointShaderModule,
        entryPoint: "fragmentMain",
        targets: [
          {
            format: this.presentationFormat,
            // No blending - opaque surface splats
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
    scaleFactorsBuffer: GPUBuffer,
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
    this.renderPoints(renderPass, uniformBuffer, positionBuffer, gradientBuffer, scaleFactorsBuffer);

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  private renderPoints(
    renderPass: GPURenderPassEncoder,
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer,
    gradientBuffer: GPUBuffer,
    scaleFactorsBuffer: GPUBuffer
  ): void {
    // Create bind group for point rendering
    const pointBindGroup = this.device.createBindGroup({
      layout: this.pointBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: gradientBuffer } },
        { binding: 3, resource: { buffer: scaleFactorsBuffer } },
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
