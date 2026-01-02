import { scaleAABB } from "./sdf/Primitive";
import { SDFScene } from "./sdf/Scene";

export class PointManager {
  private numPoints: number;
  private positionBufferA: GPUBuffer;
  private positionBufferB: GPUBuffer;
  private currentBufferIsA: boolean = true;

  /**
   * Calculate dynamic point count based on scene
   * Base count per primitive with scaling by surface area
   */
  private static calculatePointCount(scene: SDFScene): number {
    const primitives = scene.getPrimitives();
    if (primitives.length === 0) {
      return 10000; // Default fallback
    }

    const basePointsPerPrimitive = 30000;
    let totalPoints = 0;

    for (const prim of primitives) {
      // Scale by surface area (normalized)
      const surfaceArea = prim.getSurfaceArea();
      const scaleFactor = Math.sqrt(surfaceArea);
      totalPoints += Math.floor(basePointsPerPrimitive * scaleFactor);
    }

    return Math.max(10000, Math.min(totalPoints, 200000)); // Clamp to reasonable range
  }

  constructor(device: GPUDevice, scene: SDFScene) {
    const primitives = scene.getPrimitives();

    if (primitives.length === 0) {
      throw new Error("Scene must have at least one primitive");
    }

    // Calculate dynamic point count
    this.numPoints = PointManager.calculatePointCount(scene);

    console.log(
      `Initializing ${this.numPoints} points for ${primitives.length} primitive(s)`,
    );

    // Initialize positions distributed across primitive AABBs
    const positions = new Float32Array(this.numPoints * 3);

    // Calculate total surface area for proportional distribution
    const totalSurfaceArea = primitives.reduce(
      (sum, prim) => sum + prim.getSurfaceArea(),
      0,
    );

    let pointIndex = 0;

    for (const prim of primitives) {
      const aabb = scaleAABB(prim.getAABB(), 2);
      const surfaceArea = prim.getSurfaceArea();

      // Number of points for this primitive (proportional to surface area)
      const primPointCount = Math.floor(
        (surfaceArea / totalSurfaceArea) * this.numPoints,
      );

      console.log(
        `  ${prim.id}: ${primPointCount} points, AABB: [${aabb.min[0].toFixed(2)}, ${aabb.min[1].toFixed(2)}, ${aabb.min[2].toFixed(2)}] to [${aabb.max[0].toFixed(2)}, ${aabb.max[1].toFixed(2)}, ${aabb.max[2].toFixed(2)}]`,
      );

      // Distribute points on AABB surface
      for (let i = 0; i < primPointCount && pointIndex < this.numPoints; i++) {
        // Randomly choose one of 6 faces
        const face = Math.floor(Math.random() * 6);
        let x: number, y: number, z: number;

        const minX = aabb.min[0];
        const minY = aabb.min[1];
        const minZ = aabb.min[2];
        const maxX = aabb.max[0];
        const maxY = aabb.max[1];
        const maxZ = aabb.max[2];

        switch (face) {
          case 0: // -X face (left)
            x = minX;
            y = minY + Math.random() * (maxY - minY);
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 1: // +X face (right)
            x = maxX;
            y = minY + Math.random() * (maxY - minY);
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 2: // -Y face (bottom)
            x = minX + Math.random() * (maxX - minX);
            y = minY;
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 3: // +Y face (top)
            x = minX + Math.random() * (maxX - minX);
            y = maxY;
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 4: // -Z face (back)
            x = minX + Math.random() * (maxX - minX);
            y = minY + Math.random() * (maxY - minY);
            z = minZ;
            break;
          case 5: // +Z face (front)
            x = minX + Math.random() * (maxX - minX);
            y = minY + Math.random() * (maxY - minY);
            z = maxZ;
            break;
          default:
            x = minX;
            y = minY;
            z = minZ;
        }

        positions[pointIndex * 3 + 0] = x;
        positions[pointIndex * 3 + 1] = y;
        positions[pointIndex * 3 + 2] = z;

        pointIndex++;
      }
    }

    // Fill remaining points if any (due to rounding) with last primitive's AABB surface
    if (pointIndex < this.numPoints) {
      const lastPrim = primitives[primitives.length - 1];
      const aabb = lastPrim.getAABB();

      for (let i = pointIndex; i < this.numPoints; i++) {
        const face = Math.floor(Math.random() * 6);
        let x: number, y: number, z: number;

        const minX = aabb.min[0];
        const minY = aabb.min[1];
        const minZ = aabb.min[2];
        const maxX = aabb.max[0];
        const maxY = aabb.max[1];
        const maxZ = aabb.max[2];

        switch (face) {
          case 0:
            x = minX;
            y = minY + Math.random() * (maxY - minY);
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 1:
            x = maxX;
            y = minY + Math.random() * (maxY - minY);
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 2:
            x = minX + Math.random() * (maxX - minX);
            y = minY;
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 3:
            x = minX + Math.random() * (maxX - minX);
            y = maxY;
            z = minZ + Math.random() * (maxZ - minZ);
            break;
          case 4:
            x = minX + Math.random() * (maxX - minX);
            y = minY + Math.random() * (maxY - minY);
            z = minZ;
            break;
          case 5:
            x = minX + Math.random() * (maxX - minX);
            y = minY + Math.random() * (maxY - minY);
            z = maxZ;
            break;
          default:
            x = minX;
            y = minY;
            z = minZ;
        }

        positions[i * 3 + 0] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
      }
    }

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
