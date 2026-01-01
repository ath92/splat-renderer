export class PointManager {
  private numPoints: number;
  private positionBufferA: GPUBuffer;
  private positionBufferB: GPUBuffer;
  private currentBufferIsA: boolean = true;

  constructor(device: GPUDevice, numPoints: number) {
    this.numPoints = numPoints;

    // Initialize random positions in normalized screen space [-1, 1]
    const positions = new Float32Array(numPoints * 2);
    for (let i = 0; i < numPoints; i++) {
      positions[i * 2] = Math.random() * 2 - 1; // x
      positions[i * 2 + 1] = Math.random() * 2 - 1; // y
    }

    // Create ping-pong GPU buffers for positions
    const bufferSize = positions.byteLength;
    const bufferUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

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
