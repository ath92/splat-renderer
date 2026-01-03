import { SDFScene } from "./sdf/Scene";
import { WGSLCodeGenerator } from "./sdf/CodeGenerator";
import { ParameterEncoder } from "./sdf/ParameterEncoder";

export class CurvatureSampler {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private scaleFactorsBuffer: GPUBuffer;
  private sceneParamsBuffer: GPUBuffer;
  private numPoints: number;
  private scene: SDFScene;
  private currentStructureHash: string;

  constructor(device: GPUDevice, scene: SDFScene, numPoints: number) {
    this.device = device;
    this.numPoints = numPoints;
    this.scene = scene;
    this.currentStructureHash = "";

    // Create scale factors output buffer (vec4f per point: avgNormal.xyz + scaleFactor)
    const scaleFactorsBufferSize = numPoints * 16; // vec4f per point
    this.scaleFactorsBuffer = device.createBuffer({
      size: scaleFactorsBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create scene parameters buffer (initially empty)
    const bufferSize = Math.max(16, ParameterEncoder.getBufferSize(scene));
    this.sceneParamsBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group layout
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" }, // positions
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }, // scale factors output
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" }, // scene params
        },
      ],
    });

    // Build initial pipeline
    this.computePipeline = this.buildPipeline();
    this.currentStructureHash = scene.getStructureHash();

    // Upload initial scene parameters
    this.updateSceneParameters();
  }

  /**
   * Build/rebuild the compute pipeline with current scene structure
   */
  private buildPipeline(): GPUComputePipeline {
    // Generate the sceneSDF function from the scene
    const sdfFunctionCode = WGSLCodeGenerator.generateSDFFunction(this.scene);

    // Create shader code that samples normals with jitter
    const shaderCode = `
      ${sdfFunctionCode}

      struct CurvatureData {
        values: array<vec4f>, // (avgNormal.xyz, scaleFactor)
      }

      struct PositionData {
        positions: array<vec4f>,
      }

      @group(0) @binding(0) var<storage, read> positions: PositionData;
      @group(0) @binding(1) var<storage, read_write> curvatureData: CurvatureData;
      @group(0) @binding(2) var<uniform> sceneParams: SceneParams;

      @compute @workgroup_size(64)
      fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
        let index = globalId.x;

        // Bounds check
        if (index >= arrayLength(&positions.positions)) {
          return;
        }

        let centerPos = positions.positions[index].xyz;

        // Sample normals at jittered positions around the point
        let sampleRadius = 0.02; // World space jitter distance

        // Define jitter offsets (using fixed pattern for consistency)
        let offsets = array<vec3f, 6>(
          vec3f(sampleRadius, 0.0, 0.0),
          vec3f(-sampleRadius, 0.0, 0.0),
          vec3f(0.0, sampleRadius, 0.0),
          vec3f(0.0, -sampleRadius, 0.0),
          vec3f(0.0, 0.0, sampleRadius),
          vec3f(0.0, 0.0, -sampleRadius)
        );

        // Sample center normal
        let centerResult = sceneSDF(centerPos);
        let centerNormal = normalize(centerResult.yzw);

        // Sample normals at offset positions, accumulate for averaging and measure variation
        var totalNormal = centerNormal;
        var totalVariation = 0.0;

        for (var i = 0u; i < 6u; i++) {
          let samplePos = centerPos + offsets[i];
          let sampleResult = sceneSDF(samplePos);
          let sampleNormal = normalize(sampleResult.yzw);

          // Accumulate normals
          totalNormal += sampleNormal;

          // Measure angular difference (1 - dot product)
          let angularDiff = 1.0 - dot(centerNormal, sampleNormal);
          totalVariation += angularDiff;
        }

        // Average normal (7 samples: 1 center + 6 offsets)
        let avgNormal = normalize(totalNormal / 7.0);

        // Average variation
        let avgVariation = totalVariation / 6.0;

        // Map variation to scale factor
        // Low variation (flat surface) -> scale = 1.0
        // High variation (corner/edge) -> scale = 0.1
        let scaleFactor = 1.0 - smoothstep(0.0, 0.5, avgVariation);
        let finalScale = mix(0.1, 1.0, scaleFactor);

        // Store average normal and scale factor
        curvatureData.values[index] = vec4f(avgNormal, finalScale);
      }
    `;

    // Create compute shader module
    const computeShaderModule = this.device.createShaderModule({
      label: "Curvature sampler shader (generated)",
      code: shaderCode,
    });

    // Create compute pipeline
    return this.device.createComputePipeline({
      label: "Curvature compute pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: computeShaderModule,
        entryPoint: "computeMain",
      },
    });
  }

  /**
   * Update scene parameters in the uniform buffer
   */
  updateSceneParameters(): void {
    const params = ParameterEncoder.encodeParameters(this.scene);
    this.device.queue.writeBuffer(
      this.sceneParamsBuffer,
      0,
      params.buffer,
      params.byteOffset,
      params.byteLength,
    );
  }

  /**
   * Check if scene structure has changed and rebuild pipeline if needed
   */
  rebuildIfNeeded(): void {
    const newHash = this.scene.getStructureHash();
    if (newHash !== this.currentStructureHash) {
      console.log("Scene structure changed, rebuilding curvature pipeline...");

      // Rebuild pipeline
      this.computePipeline = this.buildPipeline();
      this.currentStructureHash = newHash;

      // Check if we need to resize the parameters buffer
      const requiredSize = ParameterEncoder.getBufferSize(this.scene);
      if (requiredSize > this.sceneParamsBuffer.size) {
        this.sceneParamsBuffer.destroy();
        this.sceneParamsBuffer = this.device.createBuffer({
          size: requiredSize,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      }

      // Update parameters
      this.updateSceneParameters();
    }
  }

  /**
   * Compute curvature-based scale factors for all points
   */
  computeScaleFactors(
    commandEncoder: GPUCommandEncoder,
    positionBuffer: GPUBuffer,
  ): void {
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: this.scaleFactorsBuffer } },
        { binding: 2, resource: { buffer: this.sceneParamsBuffer } },
      ],
    });

    // Dispatch compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(this.numPoints / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  getScaleFactorsBuffer(): GPUBuffer {
    return this.scaleFactorsBuffer;
  }

  destroy(): void {
    this.scaleFactorsBuffer.destroy();
    this.sceneParamsBuffer.destroy();
  }
}
