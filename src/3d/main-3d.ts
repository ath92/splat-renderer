import computeGradientCode from "./shaders/compute-gradient-3d.wgsl?raw";
import updatePositionsCode from "./shaders/update-positions-3d.wgsl?raw";
import { Camera } from "./Camera";
import { OrbitCameraController } from "./OrbitCameraController";
import { PointManager3D } from "./PointManager3D";
import { GradientSampler3D } from "./GradientSampler3D";
import { PositionUpdater3D } from "./PositionUpdater3D";
import { Renderer3D } from "./Renderer3D";

async function initWebGPU() {
  // Check WebGPU support
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  // Get canvas
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  if (!canvas) {
    throw new Error("Canvas not found");
  }

  // Get WebGPU adapter and device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Failed to get WebGPU context");
  }

  // Configure canvas
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
  });

  // Create uniform buffer
  // mat4x4f (64 bytes) + vec3f (12 bytes) + f32 (4 bytes) = 80 bytes
  const uniformBufferSize = 80;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize components
  const numPoints = 100000;

  const camera = new Camera();
  new OrbitCameraController(camera, canvas);

  const pointManager = new PointManager3D(device, numPoints);
  const gradientSampler = new GradientSampler3D(
    device,
    computeGradientCode,
    numPoints
  );
  const positionUpdater = new PositionUpdater3D(
    device,
    updatePositionsCode,
    numPoints
  );
  const renderer = new Renderer3D(
    device,
    context,
    presentationFormat,
    numPoints
  );

  // Resize canvas to fill window
  function resizeCanvas() {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    camera.setAspect(canvas.width / canvas.height);
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // Render loop
  let startTime = Date.now();

  function render() {
    const currentTime = (Date.now() - startTime) / 1000;

    // Get camera matrices
    const vpMatrix = camera.getViewProjectionMatrix();
    const cameraPos = camera.getPosition();

    // Update uniforms
    // mat4x4f (64 bytes) + vec3f (12 bytes) + f32 (4 bytes)
    const uniformData = new Float32Array(20); // 80 bytes / 4 = 20 floats

    // Copy view-projection matrix (16 floats)
    for (let i = 0; i < 16; i++) {
      uniformData[i] = vpMatrix[i];
    }

    // Copy camera position (3 floats)
    uniformData[16] = cameraPos[0];
    uniformData[17] = cameraPos[1];
    uniformData[18] = cameraPos[2];

    // Copy time (1 float)
    uniformData[19] = currentTime;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create single command encoder for all GPU work
    const commandEncoder = device.createCommandEncoder();

    // 1. Evaluate gradients at current point positions (compute pass)
    gradientSampler.evaluateGradients(
      commandEncoder,
      uniformBuffer,
      pointManager.getCurrentPositionBuffer()
    );

    // 2. Update positions based on gradients (compute pass)
    positionUpdater.updatePositions(
      commandEncoder,
      uniformBuffer,
      pointManager.getCurrentPositionBuffer(),
      gradientSampler.getGradientBuffer(),
      pointManager.getNextPositionBuffer()
    );

    // Submit compute work
    device.queue.submit([commandEncoder.finish()]);

    // Swap buffers for next frame
    pointManager.swap();

    // 3. Render scene (separate command encoder for render pass)
    renderer.render(
      uniformBuffer,
      pointManager.getCurrentPositionBuffer(),
      gradientSampler.getGradientBuffer(),
      canvas.width,
      canvas.height
    );

    requestAnimationFrame(render);
  }

  render();
}

// Initialize the app
initWebGPU().catch((error) => {
  console.error("Failed to initialize WebGPU:", error);
  document.body.innerHTML = `
    <div style="color: white; padding: 20px; font-family: sans-serif;">
      <h2>WebGPU Error</h2>
      <p>${error.message}</p>
      <p>Please make sure you're using a browser that supports WebGPU (Chrome 113+, Edge 113+, or recent Chrome Canary).</p>
    </div>
  `;
});
