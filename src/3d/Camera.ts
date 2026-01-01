import { mat4, vec3 } from "gl-matrix";

export class Camera {
  // Orbit camera parameters
  target: vec3;
  distance: number;
  azimuth: number; // Horizontal rotation (radians)
  elevation: number; // Vertical rotation (radians)

  // Projection parameters
  fov: number; // Field of view in degrees
  aspect: number;
  near: number;
  far: number;

  // Cached matrices
  private viewMatrix: mat4;
  private projectionMatrix: mat4;
  private viewProjectionMatrix: mat4;
  private cameraPosition: vec3;
  private isDirty: boolean = true;

  constructor() {
    this.target = vec3.fromValues(0, 0, 0);
    this.distance = 3.0;
    this.azimuth = 0.5;
    this.elevation = 0.5;
    this.fov = 45;
    this.aspect = 1.0;
    this.near = 0.1;
    this.far = 100.0;

    this.viewMatrix = mat4.create();
    this.projectionMatrix = mat4.create();
    this.viewProjectionMatrix = mat4.create();
    this.cameraPosition = vec3.create();
  }

  setAspect(aspect: number): void {
    this.aspect = aspect;
    this.isDirty = true;
  }

  rotate(deltaAzimuth: number, deltaElevation: number): void {
    this.azimuth += deltaAzimuth;
    this.elevation += deltaElevation;

    // Clamp elevation to prevent gimbal lock
    const maxElevation = Math.PI / 2 - 0.01;
    this.elevation = Math.max(-maxElevation, Math.min(maxElevation, this.elevation));

    this.isDirty = true;
  }

  zoom(deltaDistance: number): void {
    this.distance += deltaDistance;
    this.distance = Math.max(0.5, Math.min(20.0, this.distance));
    this.isDirty = true;
  }

  pan(deltaX: number, deltaY: number): void {
    // Calculate camera right and up vectors
    const position = this.getCameraPosition();
    const forward = vec3.create();
    vec3.subtract(forward, this.target, position);
    vec3.normalize(forward, forward);

    const right = vec3.create();
    vec3.cross(right, forward, vec3.fromValues(0, 1, 0));
    vec3.normalize(right, right);

    const up = vec3.create();
    vec3.cross(up, right, forward);
    vec3.normalize(up, up);

    // Move target
    const offset = vec3.create();
    vec3.scaleAndAdd(offset, offset, right, deltaX);
    vec3.scaleAndAdd(offset, offset, up, deltaY);
    vec3.add(this.target, this.target, offset);

    this.isDirty = true;
  }

  private getCameraPosition(): vec3 {
    const x = this.distance * Math.cos(this.elevation) * Math.sin(this.azimuth);
    const y = this.distance * Math.sin(this.elevation);
    const z = this.distance * Math.cos(this.elevation) * Math.cos(this.azimuth);

    return vec3.fromValues(
      this.target[0] + x,
      this.target[1] + y,
      this.target[2] + z
    );
  }

  private updateMatrices(): void {
    if (!this.isDirty) return;

    // Calculate camera position
    this.cameraPosition = this.getCameraPosition();

    // Create view matrix
    mat4.lookAt(
      this.viewMatrix,
      this.cameraPosition,
      this.target,
      vec3.fromValues(0, 1, 0)
    );

    // Create projection matrix
    mat4.perspective(
      this.projectionMatrix,
      (this.fov * Math.PI) / 180,
      this.aspect,
      this.near,
      this.far
    );

    // Combine into view-projection matrix
    mat4.multiply(
      this.viewProjectionMatrix,
      this.projectionMatrix,
      this.viewMatrix
    );

    this.isDirty = false;
  }

  getViewProjectionMatrix(): mat4 {
    this.updateMatrices();
    return this.viewProjectionMatrix;
  }

  getPosition(): vec3 {
    this.updateMatrices();
    return this.cameraPosition;
  }
}
