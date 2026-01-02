import { SDFScene } from "./Scene";

export class ParameterEncoder {
  /**
   * Encode scene parameters into a Float32Array suitable for uniform buffer
   * This must match WGSL struct alignment rules:
   * - vec3f has alignment 16 bytes (size 12, but next field starts at +16)
   * - f32 has alignment 4 bytes
   * - vec2f has alignment 8 bytes
   */
  static encodeParameters(scene: SDFScene): Float32Array {
    const primitives = scene.getPrimitives();
    const operations = scene.getOperations();

    if (primitives.length === 0) {
      return new Float32Array([0.0]); // Dummy value
    }

    const values: number[] = [];
    let byteOffset = 0;

    // Helper to align offset to boundary
    const alignTo = (offset: number, alignment: number): number => {
      return Math.ceil(offset / alignment) * alignment;
    };

    // Helper to add padding
    const padTo = (alignment: number) => {
      const currentBytes = byteOffset;
      const alignedBytes = alignTo(currentBytes, alignment);
      const paddingFloats = (alignedBytes - currentBytes) / 4;
      for (let i = 0; i < paddingFloats; i++) {
        values.push(0.0);
      }
      byteOffset = alignedBytes;
    };

    // Encode primitive parameters with proper WGSL alignment
    for (const prim of primitives) {
      const paramNames = prim.getParamNames();
      const paramValues = prim.getParamValues();
      let valueIndex = 0;

      for (const name of paramNames) {
        if (name.endsWith("_center")) {
          // vec3f: align to 16 bytes, size is 12 bytes (3 floats)
          padTo(16);
          values.push(paramValues[valueIndex++]); // x
          values.push(paramValues[valueIndex++]); // y
          values.push(paramValues[valueIndex++]); // z
          byteOffset += 12;

          // Add padding float to reach next 16-byte boundary
          values.push(0.0); // padding
          byteOffset += 4;

          // Skip the padding in paramValues if it exists
          if (valueIndex < paramValues.length && paramValues[valueIndex] === 0) {
            valueIndex++;
          }
        } else if (name.endsWith("_size")) {
          // vec3f: align to 16 bytes
          padTo(16);
          values.push(paramValues[valueIndex++]); // x
          values.push(paramValues[valueIndex++]); // y
          values.push(paramValues[valueIndex++]); // z
          byteOffset += 12;

          // Add padding float
          values.push(0.0); // padding
          byteOffset += 4;

          // Skip padding in paramValues
          if (valueIndex < paramValues.length && paramValues[valueIndex] === 0) {
            valueIndex++;
          }
        } else if (name.endsWith("_radii") || name.endsWith("_params")) {
          // vec2f: align to 8 bytes
          padTo(8);
          values.push(paramValues[valueIndex++]);
          values.push(paramValues[valueIndex++]);
          byteOffset += 8;

          // Add padding to next vec4 boundary if needed
          values.push(0.0);
          values.push(0.0);
          byteOffset += 8;

          // Skip padding in paramValues
          valueIndex += 2;
        } else if (name.endsWith("_radius")) {
          // f32: align to 4 bytes
          padTo(4);
          values.push(paramValues[valueIndex++]);
          byteOffset += 4;
        }
      }
    }

    // Encode operation parameters
    for (const op of operations) {
      const paramValues = op.getParamValues();
      for (const value of paramValues) {
        // f32: align to 4 bytes
        padTo(4);
        values.push(value);
        byteOffset += 4;
      }
    }

    // Final padding to 16-byte boundary
    padTo(16);

    return new Float32Array(values);
  }

  /**
   * Calculate the required uniform buffer size
   */
  static getBufferSize(scene: SDFScene): number {
    const params = this.encodeParameters(scene);
    // Already aligned to 16 bytes in encodeParameters
    return params.byteLength;
  }
}
