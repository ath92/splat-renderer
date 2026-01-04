import updatePositionsCode from "./shaders/update-positions.wgsl?raw";
import { Camera } from "./Camera";
import { OrbitCameraController } from "./OrbitCameraController";
import { PointManager } from "./PointManager";
import { GradientSampler } from "./GradientSampler";
import { CurvatureSampler } from "./CurvatureSampler";
import { PositionUpdater } from "./PositionUpdater";
import { SplatPropertyManager } from "./SplatPropertyManager";
import { SplatProjector } from "./SplatProjector";
import { TileBinner } from "./TileBinner";
import { RadixSorter } from "./RadixSorter";
import { ComputeShaderRenderer } from "./ComputeShaderRenderer";
import { PerformanceMonitor } from "./PerformanceMonitor";
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

  // Request device with higher limits for radix sort and timestamp queries
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("timestamp-query")
      ? ["timestamp-query" as GPUFeatureName]
      : [],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: 32768, // Required for radix sort (uses 18448 bytes)
    },
  });
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
  // mat4x4f (64 bytes) + vec3f (12 bytes) + f32 (4 bytes) + f32 screenWidth + f32 screenHeight = 88 bytes
  // Pad to 96 bytes for alignment
  const uniformBufferSize = 96;
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

  // Add another box positioned along the z-axis
  const box2 = new Box({
    id: "box2",
    position: vec3.fromValues(0, 0, -1),
    size: vec3.fromValues(0.4, 0.4, 0.4),
  });

  // Build scene graph: ((sphere1 ∪ box1) ∪ sphere2) ∪ box2
  scene.setRoot(smoothUnion(0.1, smoothUnion(0.1, smoothUnion(0.15, sphere1, box1), sphere2), box2));

  // Initialize point manager (calculates point count dynamically)
  const pointManager = new PointManager(device, scene);
  const numPoints = pointManager.getNumPoints();

  const gradientSampler = new GradientSampler(device, scene, numPoints);
  const curvatureSampler = new CurvatureSampler(device, scene, numPoints);
  const positionUpdater = new PositionUpdater(
    device,
    updatePositionsCode,
    numPoints,
  );

  // Gaussian splatting components
  const splatPropertyManager = new SplatPropertyManager(device, numPoints);
  const splatProjector = new SplatProjector(device, numPoints);
  const radixSorter = new RadixSorter(device, numPoints);
  const tileBinner = new TileBinner(device, numPoints, 16); // 16x16 tiles
  const computeRenderer = new ComputeShaderRenderer(device, context, presentationFormat);
  const performanceMonitor = new PerformanceMonitor(device);

  // Resize canvas to fill window
  function resizeCanvas() {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    camera.setAspect(canvas.width / canvas.height);
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // Fit splats to scene - call this whenever the scene changes
  function fitSplatsToScene() {
    console.log("Fitting splats to scene...");

    // Reinitialize points to random positions on AABB surfaces
    pointManager.reinitialize();

    // Run gradient descent to project points onto surface
    for (let i = 0; i < 5; i++) {
      const gradientDescentCommandEncoder = device.createCommandEncoder();

      // 1. Evaluate gradients at current point positions
      gradientSampler.evaluateGradients(
        gradientDescentCommandEncoder,
        uniformBuffer,
        pointManager.getCurrentPositionBuffer(),
      );

      // 2. Update positions based on gradients
      positionUpdater.updatePositions(
        gradientDescentCommandEncoder,
        uniformBuffer,
        pointManager.getCurrentPositionBuffer(),
        gradientSampler.getGradientBuffer(),
        pointManager.getNextPositionBuffer(),
      );

      device.queue.submit([gradientDescentCommandEncoder.finish()]);
      pointManager.swap();
    }

    // Compute curvature-based scale factors after points have settled
    const curvatureCommandEncoder = device.createCommandEncoder();
    curvatureSampler.computeScaleFactors(
      curvatureCommandEncoder,
      pointManager.getCurrentPositionBuffer()
    );
    device.queue.submit([curvatureCommandEncoder.finish()]);

    // Update splat properties (radius, color, opacity) from curvature data
    const splatCommandEncoder = device.createCommandEncoder();
    splatPropertyManager.updateFromCurvature(
      splatCommandEncoder,
      pointManager.getCurrentPositionBuffer(),
      curvatureSampler.getScaleFactorsBuffer()
    );
    device.queue.submit([splatCommandEncoder.finish()]);

    console.log("Splat fitting complete!");
  }

  // Initial fit
  fitSplatsToScene();

  // Get stats display element
  const statsElement = document.getElementById("stats") as HTMLDivElement;

  // FPS tracking
  let frameCount = 0;
  let lastFpsUpdate = performance.now();
  let fps = 0;

  // Render loop - radix sort + tile-based rendering
  async function render() {
    const frameStart = performance.now();

    // Get camera matrices
    const vpMatrix = camera.getViewProjectionMatrix();
    const cameraPos = camera.getPosition();

    // Update uniforms (camera and screen dimensions)
    const uniformData = new Float32Array(24); // 96 bytes / 4 = 24 floats

    // Copy view-projection matrix (16 floats)
    for (let i = 0; i < 16; i++) {
      uniformData[i] = vpMatrix[i];
    }

    // Copy camera position (3 floats)
    uniformData[16] = cameraPos[0];
    uniformData[17] = cameraPos[1];
    uniformData[18] = cameraPos[2];

    // Time (unused but kept for compatibility)
    uniformData[19] = 0;

    // Screen dimensions (2 floats)
    uniformData[20] = canvas.width;
    uniformData[21] = canvas.height;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Rendering pipeline with radix sort
    const commandEncoder = device.createCommandEncoder();

    // Step 1: Project splats to screen space
    const projectionStart = performance.now();
    splatProjector.project(
      commandEncoder,
      uniformBuffer,
      splatPropertyManager.getPropertyBuffer()
    );
    const projectionTime = performance.now() - projectionStart;

    // Step 2: Globally sort by depth using radix sort (far to near)
    const sortStart = performance.now();
    const sortedIndices = await radixSorter.sort(
      commandEncoder,
      splatProjector.getProjectedBuffer()
    );
    const sortTime = performance.now() - sortStart;

    // Step 3: Bin sorted splats to tiles (CPU-based)
    const binStart = performance.now();
    await tileBinner.binSorted(
      sortedIndices,
      splatProjector.getProjectedBuffer(),
      canvas.width,
      canvas.height
    );
    const binTime = performance.now() - binStart;

    // Clean up temporary buffers
    tileBinner.cleanupTempBuffers();
    radixSorter.cleanupTempBuffers();

    // Step 4: Render using compute shader (manual blending)
    const renderStart = performance.now();
    computeRenderer.render(
      uniformData,
      splatPropertyManager.getPropertyBuffer(),
      tileBinner.getSplatIndicesBuffer(),
      curvatureSampler.getScaleFactorsBuffer(),
      splatProjector.getProjectedBuffer(),
      tileBinner.getTileListsBuffer(),
      tileBinner.getTileOffsetsBuffer(),
      tileBinner.getTileSize(),
      tileBinner.getNumTilesX(),
      canvas.width,
      canvas.height
    );
    const renderTime = performance.now() - renderStart;

    const frameTime = performance.now() - frameStart;

    // Update FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsUpdate >= 1000) {
      fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
      frameCount = 0;
      lastFpsUpdate = now;
    }

    // Update stats display
    if (statsElement) {
      statsElement.textContent = `FPS: ${fps}
Frame: ${frameTime.toFixed(2)}ms
Projection: ${projectionTime.toFixed(2)}ms
Radix Sort: ${sortTime.toFixed(2)}ms
Tile Binning: ${binTime.toFixed(2)}ms
Render: ${renderTime.toFixed(2)}ms
Splats: ${numPoints.toLocaleString()}`;
    }

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
