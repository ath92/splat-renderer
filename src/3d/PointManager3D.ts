export class PointManager3D {
  private numPoints: number;
  private positionBufferA: GPUBuffer;
  private positionBufferB: GPUBuffer;
  private currentBufferIsA: boolean = true;

  constructor(device: GPUDevice, numPoints: number) {
    this.numPoints = numPoints;

    // Initialize random positions in 3D bounding volume [-1, 1]³
    const positions = new Float32Array(numPoints * 3);
    for (let i = 0; i < numPoints; i++) {
      positions[i * 3 + 0] = Math.random() * 2 - 1; // x
      positions[i * 3 + 1] = Math.random() * 2 - 1; // y
      positions[i * 3 + 2] = Math.random() * 2 - 1; // z
    }

    // Create ping-pong GPU buffers for positions
    const bufferSize = positions.byteLength;
    const bufferUsage =
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

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
      positions.byteLength
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
