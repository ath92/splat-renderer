import { scaleAABB, Primitive } from "./sdf/Primitive";
import type { AABB } from "./sdf/Primitive";
import { SDFScene } from "./sdf/Scene";
import { vec3 } from "gl-matrix";

function rand_range(d = 0.0) {
  return Math.random() * (1 - d * 2) + d;
}

export class PointManager {
  private numPoints: number;
  private positionBufferA: GPUBuffer;
  private positionBufferB: GPUBuffer;
  private currentBufferIsA: boolean = true;
  private device: GPUDevice;
  private scene: SDFScene;

  /**
   * Calculate dynamic point count based on scene
   * Base count per primitive with scaling by surface area
   */
  private static calculatePointCount(scene: SDFScene): number {
    const primitives = scene.getPrimitives();
    if (primitives.length === 0) {
      return 500; // Default fallback
    }

    const basePointsPerPrimitive = 3000;
    let totalPoints = 0;

    for (const prim of primitives) {
      // Scale by surface area (normalized)
      const surfaceArea = prim.getSurfaceArea();
      const scaleFactor = Math.sqrt(surfaceArea);
      totalPoints += Math.floor(basePointsPerPrimitive * scaleFactor);
    }

    return Math.max(1000, Math.min(totalPoints, 20000)); // Clamp to reasonable range
  }

  constructor(device: GPUDevice, scene: SDFScene) {
    this.device = device;
    this.scene = scene;
    const primitives = scene.getPrimitives();

    if (primitives.length === 0) {
      throw new Error("Scene must have at least one primitive");
    }

    // Calculate dynamic point count
    this.numPoints = PointManager.calculatePointCount(scene);

    console.log(
      `Initializing ${this.numPoints} points for ${primitives.length} primitive(s)`,
    );

    // Log global AABB info
    const globalAABB = this.computeGlobalAABB(primitives);
    console.log(
      `  Global AABB: [${globalAABB.min[0].toFixed(2)}, ${globalAABB.min[1].toFixed(2)}, ${globalAABB.min[2].toFixed(2)}] to [${globalAABB.max[0].toFixed(2)}, ${globalAABB.max[1].toFixed(2)}, ${globalAABB.max[2].toFixed(2)}]`,
    );

    // Generate initial positions
    const positions = this.generateRandomPositions();

    // Create ping-pong GPU buffers for positions
    const bufferSize = positions.byteLength;
    const bufferUsage =
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC;

    this.positionBufferA = device.createBuffer({
      size: bufferSize,
      usage: bufferUsage,
    });

    this.positionBufferB = device.createBuffer({
      size: bufferSize,
      usage: bufferUsage,
    });

    // Upload initial positions to buffer A
    device.queue.writeBuffer(
      this.positionBufferA,
      0,
      positions.buffer,
      positions.byteOffset,
      positions.byteLength,
    );
  }

  /**
   * Generate random positions on AABB surfaces
   * Note: Uses vec4 alignment (4 floats per position) for WGSL storage buffer compatibility
   */
  private generateRandomPositions(): Float32Array {
    const primitives = this.scene.getPrimitives();
    const positions = new Float32Array(this.numPoints * 4); // vec4 alignment for WGSL

    // Use a single global AABB instead of per-primitive AABBs
    // This prevents overlapping AABB surfaces from creating point clusters
    const globalAABB = this.computeGlobalAABB(primitives);

    const minX = globalAABB.min[0];
    const minY = globalAABB.min[1];
    const minZ = globalAABB.min[2];
    const maxX = globalAABB.max[0];
    const maxY = globalAABB.max[1];
    const maxZ = globalAABB.max[2];

    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;

    const faceAreas = [
      dy * dz, // -X face
      dy * dz, // +X face
      dx * dz, // -Y face
      dx * dz, // +Y face
      dx * dy, // -Z face
      dx * dy, // +Z face
    ];

    const totalFaceArea = faceAreas.reduce((sum, area) => sum + area, 0);

    // Distribute all points on the global AABB surface
    for (let i = 0; i < this.numPoints; i++) {
      // Select face with probability proportional to its surface area
      const rand = Math.random() * totalFaceArea;
      let cumulativeArea = 0;
      let face = 0;
      for (let f = 0; f < 6; f++) {
        cumulativeArea += faceAreas[f];
        if (rand < cumulativeArea) {
          face = f;
          break;
        }
      }

      let x: number, y: number, z: number;

      switch (face) {
        case 0: // -X face (left)
          x = minX;
          y = minY + rand_range() * dy;
          z = minZ + rand_range() * dz;
          break;
        case 1: // +X face (right)
          x = maxX;
          y = minY + rand_range() * dy;
          z = minZ + rand_range() * dz;
          break;
        case 2: // -Y face (bottom)
          x = minX + rand_range() * dx;
          y = minY;
          z = minZ + rand_range() * dz;
          break;
        case 3: // +Y face (top)
          x = minX + rand_range() * dx;
          y = maxY;
          z = minZ + rand_range() * dz;
          break;
        case 4: // -Z face (back)
          x = minX + rand_range() * dx;
          y = minY + rand_range() * dy;
          z = minZ;
          break;
        case 5: // +Z face (front)
          x = minX + rand_range() * dx;
          y = minY + rand_range() * dy;
          z = maxZ;
          break;
        default:
          x = minX;
          y = minY;
          z = minZ;
      }

      positions[i * 4 + 0] = x;
      positions[i * 4 + 1] = y;
      positions[i * 4 + 2] = z;
      positions[i * 4 + 3] = 0; // padding for vec4 alignment
    }

    return positions;
  }

  /**
   * Compute a single global AABB that encompasses all primitives
   */
  private computeGlobalAABB(primitives: Primitive[]): AABB {
    if (primitives.length === 0) {
      return {
        min: vec3.fromValues(-1, -1, -1),
        max: vec3.fromValues(1, 1, 1),
      };
    }

    // Start with first primitive's AABB (unscaled)
    const firstAABB = primitives[0].getAABB();
    const min = vec3.clone(firstAABB.min);
    const max = vec3.clone(firstAABB.max);

    // Expand to include all primitives (unscaled)
    for (let i = 1; i < primitives.length; i++) {
      const aabb = primitives[i].getAABB();
      vec3.min(min, min, aabb.min);
      vec3.max(max, max, aabb.max);
    }

    // Now scale the global AABB by 1.5x to give points some room
    return scaleAABB({ min, max }, 1.5);
  }

  /**
   * Reinitialize points to random positions on AABB surfaces
   * Call this each frame before gradient descent to start fresh
   */
  reinitialize(): void {
    const positions = this.generateRandomPositions();

    // Write to current buffer
    this.device.queue.writeBuffer(
      this.getCurrentPositionBuffer(),
      0,
      positions.buffer,
      positions.byteOffset,
      positions.byteLength,
    );
  }

  getCurrentPositionBuffer(): GPUBuffer {
    return this.currentBufferIsA ? this.positionBufferA : this.positionBufferB;
  }

  getNextPositionBuffer(): GPUBuffer {
    return this.currentBufferIsA ? this.positionBufferB : this.positionBufferA;
  }

  swap(): void {
    this.currentBufferIsA = !this.currentBufferIsA;
  }

  getNumPoints(): number {
    return this.numPoints;
  }

  destroy(): void {
    this.positionBufferA.destroy();
    this.positionBufferB.destroy();
  }
}
