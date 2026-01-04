/**
 * Performance monitoring using WebGPU timestamp queries
 * Tracks timing for each pipeline stage
 */
export class PerformanceMonitor {
  private device: GPUDevice;
  private querySet: GPUQuerySet | null = null;
  private resolveBuffer: GPUBuffer | null = null;
  private resultBuffer: GPUBuffer | null = null;
  private timestampsSupported: boolean = false;
  private numQueries = 10; // 5 stages × 2 (start/end)

  // Rolling average for smoothing
  private timings: Map<string, number[]> = new Map();
  private readonly HISTORY_SIZE = 30;

  constructor(device: GPUDevice) {
    this.device = device;
    this.initializeTimestamps();
  }

  private initializeTimestamps(): void {
    // Check if timestamp queries are supported
    if (!this.device.features.has("timestamp-query")) {
      console.warn("Timestamp queries not supported on this device");
      return;
    }

    this.timestampsSupported = true;

    // Create query set for timestamps
    this.querySet = this.device.createQuerySet({
      type: "timestamp",
      count: this.numQueries,
    });

    // Buffer to resolve query results
    this.resolveBuffer = this.device.createBuffer({
      size: this.numQueries * 8, // 8 bytes per timestamp
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    // Buffer to read back results
    this.resultBuffer = this.device.createBuffer({
      size: this.numQueries * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /**
   * Write timestamp at the beginning of a compute pass
   */
  writeTimestamp(
    passEncoder: GPUComputePassEncoder,
    queryIndex: number
  ): void {
    if (!this.timestampsSupported || !this.querySet) return;
    passEncoder.writeTimestamp(this.querySet, queryIndex);
  }

  /**
   * Begin a timed compute pass
   */
  beginComputePass(
    commandEncoder: GPUCommandEncoder,
    queryIndex: number,
    label?: string
  ): GPUComputePassEncoder {
    if (!this.timestampsSupported || !this.querySet) {
      return commandEncoder.beginComputePass({ label });
    }

    return commandEncoder.beginComputePass({
      label,
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassTimestampIndex: queryIndex,
        endOfPassTimestampIndex: queryIndex + 1,
      },
    });
  }

  /**
   * Resolve and read back all timestamps
   */
  async resolveTimestamps(commandEncoder: GPUCommandEncoder): Promise<void> {
    if (!this.timestampsSupported || !this.querySet || !this.resolveBuffer || !this.resultBuffer) {
      return;
    }

    // Resolve queries to buffer
    commandEncoder.resolveQuerySet(
      this.querySet,
      0,
      this.numQueries,
      this.resolveBuffer,
      0
    );

    // Copy to readable buffer
    commandEncoder.copyBufferToBuffer(
      this.resolveBuffer,
      0,
      this.resultBuffer,
      0,
      this.numQueries * 8
    );
  }

  /**
   * Read back and calculate timings
   */
  async readTimings(): Promise<Map<string, number>> {
    if (!this.timestampsSupported || !this.resultBuffer) {
      return new Map();
    }

    await this.resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigUint64Array(this.resultBuffer.getMappedRange());

    const stages = new Map<string, number>();

    // Calculate durations (in milliseconds)
    const projection = Number(times[1] - times[0]) / 1_000_000;
    const sort = Number(times[3] - times[2]) / 1_000_000;
    const render = Number(times[5] - times[4]) / 1_000_000;

    this.resultBuffer.unmap();

    // Update rolling averages
    this.updateAverage("Projection", projection);
    this.updateAverage("Radix Sort", sort);
    this.updateAverage("Render", render);

    // Return smoothed values
    stages.set("Projection", this.getAverage("Projection"));
    stages.set("Radix Sort", this.getAverage("Sort"));
    stages.set("Render", this.getAverage("Render"));
    stages.set("Total GPU", this.getAverage("Projection") + this.getAverage("Sort") + this.getAverage("Render"));

    return stages;
  }

  /**
   * Update rolling average for a timing
   */
  private updateAverage(name: string, value: number): void {
    if (!this.timings.has(name)) {
      this.timings.set(name, []);
    }

    const history = this.timings.get(name)!;
    history.push(value);

    if (history.length > this.HISTORY_SIZE) {
      history.shift();
    }
  }

  /**
   * Get smoothed average for a timing
   */
  private getAverage(name: string): number {
    const history = this.timings.get(name);
    if (!history || history.length === 0) return 0;

    const sum = history.reduce((a, b) => a + b, 0);
    return sum / history.length;
  }

  /**
   * Format timings as a string for display
   */
  formatTimings(timings: Map<string, number>, cpuTime?: number): string {
    const lines: string[] = [];

    if (cpuTime !== undefined) {
      lines.push(`CPU (Tile Binning): ${cpuTime.toFixed(2)}ms`);
    }

    for (const [stage, time] of timings) {
      lines.push(`${stage}: ${time.toFixed(2)}ms`);
    }

    return lines.join("\n");
  }

  destroy(): void {
    if (this.querySet) {
      this.querySet.destroy();
    }
    if (this.resolveBuffer) {
      this.resolveBuffer.destroy();
    }
    if (this.resultBuffer) {
      this.resultBuffer.destroy();
    }
  }
}
