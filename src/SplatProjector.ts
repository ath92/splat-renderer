/**
 * Projects splats to screen space and computes 2D bounding boxes
 * Output: ProjectedSplat data (screen bounds, depth, original index)
 */
export class SplatProjector {
  private device: GPUDevice;
  private numSplats: number;
  private projectedBuffer: GPUBuffer;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, numSplats: number) {
    this.device = device;
    this.numSplats = numSplats;

    // Create projected splat buffer
    // Each ProjectedSplat: vec2f min, vec2f max, f32 depth, u32 index = 6 floats + 1 u32 = 28 bytes
    // Pad to 32 bytes for alignment
    const bufferSize = numSplats * 32;
    this.projectedBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

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
        screenWidth: f32,
        screenHeight: f32,
      }

      struct SplatProperties {
        data: array<vec4f>, // Interleaved: [pos+radius, color+opacity, ...]
      }

      struct ProjectedSplat {
        screenBoundsMin: vec2f,
        screenBoundsMax: vec2f,
        depth: f32,
        screenRadius: f32,  // Actual screen-space radius (not padded)
        originalIndex: u32,
        _padding: f32, // Align to 32 bytes
      }

      struct ProjectedSplats {
        splats: array<ProjectedSplat>,
      }

      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> splatProperties: SplatProperties;
      @group(0) @binding(2) var<storage, read_write> projectedSplats: ProjectedSplats;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) globalId: vec3u) {
        let index = globalId.x;
        if (index >= arrayLength(&splatProperties.data) / 2u) {
          return;
        }

        // Read splat properties
        let posRadius = splatProperties.data[index * 2u];
        let worldPos = posRadius.xyz;
        let radius = posRadius.w;

        // Compute camera-space depth
        let depth = distance(worldPos, uniforms.cameraPosition);

        // Project to clip space
        let clipPos = uniforms.viewProjectionMatrix * vec4f(worldPos, 1.0);

        // Perspective divide to NDC [-1, 1]
        let ndc = clipPos.xyz / clipPos.w;

        // Convert NDC to screen space [0, width/height]
        let screenCenter = vec2f(
          (ndc.x + 1.0) * 0.5 * uniforms.screenWidth,
          (1.0 - ndc.y) * 0.5 * uniforms.screenHeight
        );

        // Estimate screen-space radius by checking multiple directions
        // Project offsets in X, Y, and Z directions to find maximum screen-space extent
        let offsets = array<vec3f, 6>(
          vec3f(radius, 0.0, 0.0),
          vec3f(-radius, 0.0, 0.0),
          vec3f(0.0, radius, 0.0),
          vec3f(0.0, -radius, 0.0),
          vec3f(0.0, 0.0, radius),
          vec3f(0.0, 0.0, -radius)
        );

        var maxScreenRadius = 0.0;
        for (var i = 0u; i < 6u; i++) {
          let offsetPos = worldPos + offsets[i];
          let offsetClip = uniforms.viewProjectionMatrix * vec4f(offsetPos, 1.0);
          let offsetNDC = offsetClip.xyz / offsetClip.w;
          let offsetScreen = vec2f(
            (offsetNDC.x + 1.0) * 0.5 * uniforms.screenWidth,
            (1.0 - offsetNDC.y) * 0.5 * uniforms.screenHeight
          );
          let dist = distance(screenCenter, offsetScreen);
          maxScreenRadius = max(maxScreenRadius, dist);
        }

        // Compute 2D bounding box with safety margin for conservative tile binning
        // Gaussian (sigma=0.5) drops to ~1% at normalized distance of 3.0
        // But normalizedDist = pixelDist / screenRadius, so we need bounds = 3.0 * screenRadius
        // However, that's for full accuracy. For performance, use 1.5x (99% coverage)
        let paddedRadius = maxScreenRadius * 1.5;
        let boundsMin = screenCenter - vec2f(paddedRadius);
        let boundsMax = screenCenter + vec2f(paddedRadius);

        // Store projected splat data
        projectedSplats.splats[index] = ProjectedSplat(
          boundsMin,
          boundsMax,
          depth,
          maxScreenRadius,  // Actual screen-space radius (for rendering)
          index,
          0.0 // padding
        );
      }
    `;

    const shaderModule = this.device.createShaderModule({
      label: "Splat projection shader",
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
      label: "Splat projection pipeline",
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

  project(
    commandEncoder: GPUCommandEncoder,
    uniformBuffer: GPUBuffer,
    splatPropertyBuffer: GPUBuffer
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: splatPropertyBuffer } },
        { binding: 2, resource: { buffer: this.projectedBuffer } },
      ],
    });

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(this.numSplats / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  getProjectedBuffer(): GPUBuffer {
    return this.projectedBuffer;
  }

  destroy(): void {
    this.projectedBuffer.destroy();
  }
}
