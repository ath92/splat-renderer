import { Camera } from "./Camera";

export class OrbitCameraController {
  private camera: Camera;
  private canvas: HTMLCanvasElement;

  private isDragging: boolean = false;
  private dragButton: number = -1;
  private lastMouseX: number = 0;
  private lastMouseY: number = 0;

  private rotationSpeed: number = 0.005;
  private panSpeed: number = 0.002;
  private zoomSpeed: number = 0.001;

  constructor(camera: Camera, canvas: HTMLCanvasElement) {
    this.camera = camera;
    this.canvas = canvas;

    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.canvas.addEventListener("mouseup", this.onMouseUp.bind(this));
    this.canvas.addEventListener("wheel", this.onWheel.bind(this), {
      passive: false,
    });

    // Prevent context menu on right-click
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  private onMouseDown(event: MouseEvent): void {
    this.isDragging = true;
    this.dragButton = event.button;
    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.isDragging) return;

    const dx = event.clientX - this.lastMouseX;
    const dy = event.clientY - this.lastMouseY;

    if (this.dragButton === 0) {
      // Left button: rotate
      this.camera.rotate(dx * this.rotationSpeed, -dy * this.rotationSpeed);
    } else if (this.dragButton === 1 || this.dragButton === 2) {
      // Middle or right button: pan
      this.camera.pan(-dx * this.panSpeed, dy * this.panSpeed);
    }

    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;
  }

  private onMouseUp(_event: MouseEvent): void {
    this.isDragging = false;
    this.dragButton = -1;
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();

    const delta = event.deltaY * this.zoomSpeed;
    this.camera.zoom(delta);
  }

  destroy(): void {
    // Remove event listeners if needed
  }
}
