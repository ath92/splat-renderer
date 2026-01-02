import { SDFScene } from "./sdf/Scene";
import { WGSLCodeGenerator } from "./sdf/CodeGenerator";
import { ParameterEncoder } from "./sdf/ParameterEncoder";

export class GradientSampler {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private gradientBuffer: GPUBuffer;
  private sceneParamsBuffer: GPUBuffer;
  private numPoints: number;
  private scene: SDFScene;
  private currentStructureHash: string;

  constructor(device: GPUDevice, scene: SDFScene, numPoints: number) {
    this.device = device;
    this.numPoints = numPoints;
    this.scene = scene;
    this.currentStructureHash = "";

    // Create gradient output buffer
    const gradientBufferSize = numPoints * 4 * 4; // vec4f per point
    this.gradientBuffer = device.createBuffer({
      size: gradientBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create scene parameters buffer (initially empty)
    const bufferSize = Math.max(16, ParameterEncoder.getBufferSize(scene));
    this.sceneParamsBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group layout with scene params
    this.bindGroupLayout = device.createBindGroupLayout({
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
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
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
    // Generate shader code from scene
    const shaderCode = WGSLCodeGenerator.generateComputeShader(this.scene);

    console.log("Generated shader code:");
    console.log(shaderCode);

    // Create compute shader module
    const computeShaderModule = this.device.createShaderModule({
      label: "Compute gradient shader (generated)",
      code: shaderCode,
    });

    // Create compute pipeline
    return this.device.createComputePipeline({
      label: "Gradient compute pipeline",
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
   * Call this every frame to animate primitives
   */
  updateSceneParameters(): void {
    const params = ParameterEncoder.encodeParameters(this.scene);
    this.device.queue.writeBuffer(this.sceneParamsBuffer, 0, params.buffer, params.byteOffset, params.byteLength);
  }

  /**
   * Check if scene structure has changed and rebuild pipeline if needed
   * Call this before rendering if you've modified the scene structure
   */
  rebuildIfNeeded(): void {
    const newHash = this.scene.getStructureHash();
    if (newHash !== this.currentStructureHash) {
      console.log("Scene structure changed, rebuilding pipeline...");

      // Rebuild pipeline
      this.computePipeline = this.buildPipeline();
      this.currentStructureHash = newHash;

      // Check if we need to resize the parameters buffer
      const requiredSize = ParameterEncoder.getBufferSize(this.scene);
      if (requiredSize > this.sceneParamsBuffer.size) {
        console.log(
          `Resizing scene params buffer: ${this.sceneParamsBuffer.size} -> ${requiredSize}`
        );
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

  evaluateGradients(
    commandEncoder: GPUCommandEncoder,
    uniformBuffer: GPUBuffer,
    positionBuffer: GPUBuffer
  ): void {
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: this.gradientBuffer } },
        { binding: 3, resource: { buffer: this.sceneParamsBuffer } },
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

  getGradientBuffer(): GPUBuffer {
    return this.gradientBuffer;
  }

  getScene(): SDFScene {
    return this.scene;
  }

  destroy(): void {
    this.gradientBuffer.destroy();
    this.sceneParamsBuffer.destroy();
  }
}
