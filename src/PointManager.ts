export class PointManager {
  private numPoints: number;
  private positions: Float32Array;
  private positionBuffer: GPUBuffer;
  private stepSize: number = 0.01;

  constructor(device: GPUDevice, numPoints: number) {
    this.numPoints = numPoints;
    this.positions = new Float32Array(numPoints * 2);

    // Initialize random positions in normalized screen space [-1, 1]
    for (let i = 0; i < numPoints; i++) {
      this.positions[i * 2] = Math.random() * 2 - 1; // x
      this.positions[i * 2 + 1] = Math.random() * 2 - 1; // y
    }

    // Create GPU buffer for positions
    this.positionBuffer = device.createBuffer({
      size: this.positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Upload initial positions
    device.queue.writeBuffer(
      this.positionBuffer,
      0,
      this.positions.buffer,
      this.positions.byteOffset,
      this.positions.byteLength
    );
  }

  getPositionBuffer(): GPUBuffer {
    return this.positionBuffer;
  }

  getPositions(): Float32Array {
    return this.positions;
  }

  getNumPoints(): number {
    return this.numPoints;
  }

  updatePositions(
    device: GPUDevice,
    gradientResults: Float32Array
  ): void {
    // Update positions based on gradient descent
    // gradientResults format: [distance, gradient.x, gradient.y, padding] for each point
    for (let i = 0; i < this.numPoints; i++) {
      const distance = gradientResults[i * 4];
      const gradX = gradientResults[i * 4 + 1];
      const gradY = gradientResults[i * 4 + 2];

      // Distance-aware gradient descent
      // Move towards surface: pos -= normalize(gradient) * distance * stepSize
      const gradLen = Math.sqrt(gradX * gradX + gradY * gradY);
      if (gradLen > 0.0001) {
        const normalizedGradX = gradX / gradLen;
        const normalizedGradY = gradY / gradLen;

        this.positions[i * 2] -= normalizedGradX * distance * this.stepSize;
        this.positions[i * 2 + 1] -= normalizedGradY * distance * this.stepSize;
      }
    }

    // Upload updated positions to GPU
    device.queue.writeBuffer(
      this.positionBuffer,
      0,
      this.positions.buffer,
      this.positions.byteOffset,
      this.positions.byteLength
    );
  }

  destroy(): void {
    this.positionBuffer.destroy();
  }
}
