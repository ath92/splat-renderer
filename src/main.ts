import updatePositionsCode from "./shaders/update-positions.wgsl?raw";
import { Camera } from "./Camera";
import { OrbitCameraController } from "./OrbitCameraController";
import { PointManager } from "./PointManager";
import { GradientSampler } from "./GradientSampler";
import { PositionUpdater } from "./PositionUpdater";
import { Renderer } from "./Renderer";
import { SDFScene, smoothUnion } from "./sdf/Scene";
import { Sphere } from "./sdf/Primitive";
import { Box } from "./sdf/Primitive";
import { vec3 } from "gl-matrix";

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
  const camera = new Camera();
  new OrbitCameraController(camera, canvas);

  // Create SDF scene
  const scene = new SDFScene();

  // Add a sphere
  const sphere1 = new Sphere({
    id: "sphere1",
    position: vec3.fromValues(0, 0, 0),
    radius: 0.5,
  });

  // Add a box
  const box1 = new Box({
    id: "box1",
    position: vec3.fromValues(0.6, 0, 0),
    size: vec3.fromValues(0.3, 0.3, 0.3),
  });

  // Add another sphere
  const sphere2 = new Sphere({
    id: "sphere2",
    position: vec3.fromValues(0, 0.6, 0),
    radius: 0.25,
  });

  // Build scene graph: (sphere1 ∪ box1) ∪ sphere2
  scene.setRoot(smoothUnion(0.1, smoothUnion(0.15, sphere1, box1), sphere2));

  // Initialize point manager (calculates point count dynamically)
  const pointManager = new PointManager(device, scene);
  const numPoints = pointManager.getNumPoints();

  const gradientSampler = new GradientSampler(device, scene, numPoints);
  const positionUpdater = new PositionUpdater(
    device,
    updatePositionsCode,
    numPoints,
  );
  const renderer = new Renderer(device, context, presentationFormat, numPoints);

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

    // Animate scene parameters
    sphere1.position[0] = Math.sin(currentTime) * 0.3;
    sphere1.position[1] = Math.cos(currentTime * 0.7) * 0.2;
    sphere2.radius = 0.25 + 0.1 * Math.sin(currentTime * 2);

    // Update scene parameters in GPU buffer
    gradientSampler.updateSceneParameters();

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

    // Update uniforms
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Reinitialize points to random positions each frame
    pointManager.reinitialize();

    for (let i = 0; i < 5; i++) {
      // Create single command encoder for all GPU work
      const gradientDescentCommandEncoder = device.createCommandEncoder();

      // 1. Evaluate gradients at current point positions (compute pass)
      gradientSampler.evaluateGradients(
        gradientDescentCommandEncoder,
        uniformBuffer,
        pointManager.getCurrentPositionBuffer(),
      );

      // 2. Update positions based on gradients (compute pass)
      positionUpdater.updatePositions(
        gradientDescentCommandEncoder,
        uniformBuffer,
        pointManager.getCurrentPositionBuffer(),
        gradientSampler.getGradientBuffer(),
        pointManager.getNextPositionBuffer(),
      );

      device.queue.submit([gradientDescentCommandEncoder.finish()]);
      // Swap buffers for next frame
      pointManager.swap();
    }

    // 3. Render scene (separate command encoder for render pass)
    renderer.render(
      uniformBuffer,
      pointManager.getCurrentPositionBuffer(),
      gradientSampler.getGradientBuffer(),
      canvas.width,
      canvas.height,
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
