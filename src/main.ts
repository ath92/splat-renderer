import shaderCode from "./shader.wgsl?raw";
import shaderGradientCode from "./shader-gradient.wgsl?raw";
import computeGradientCode from "./compute-gradient.wgsl?raw";
import { PointManager } from "./PointManager";
import { GradientSampler } from "./GradientSampler";
import { Renderer } from "./Renderer";

let useGradientMode = false;

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
  const uniformBufferSize = 4 * 4; // vec2f (8 bytes) + f32 (4 bytes) + padding (4 bytes)
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize components
  const numPoints = 10000;
  const pointManager = new PointManager(device, numPoints);
  const gradientSampler = new GradientSampler(
    device,
    computeGradientCode,
    numPoints,
  );
  const renderer = new Renderer(
    device,
    context,
    presentationFormat,
    shaderCode,
    shaderGradientCode,
    uniformBuffer,
    numPoints,
  );

  // Resize canvas to fill window
  function resizeCanvas() {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // Toggle gradient mode with 'G' key
  window.addEventListener("keydown", (event) => {
    if (event.key === "g" || event.key === "G") {
      useGradientMode = !useGradientMode;
      console.log(`Gradient mode: ${useGradientMode ? "ON" : "OFF"}`);
    }
  });

  // Render loop
  let startTime = Date.now();

  async function render() {
    const currentTime = (Date.now() - startTime) / 1000;

    // Update uniforms
    const uniformData = new Float32Array([
      canvas.width,
      canvas.height,
      currentTime,
      0, // padding
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Evaluate gradients at point positions (compute shader)
    const gradientResults = await gradientSampler.evaluateGradients(
      uniformBuffer,
      pointManager.getPositionBuffer(),
    );

    // Update point positions based on gradients (CPU)
    pointManager.updatePositions(device, gradientResults);

    // Render scene
    renderer.render(
      useGradientMode,
      pointManager.getPositions(),
      gradientResults,
      numPoints,
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
