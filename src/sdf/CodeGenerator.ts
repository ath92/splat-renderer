import { SDFScene, type SceneNode } from "./Scene";
import { PrimitiveType } from "./Primitive";
import { OperationType } from "./Operation";

export class WGSLCodeGenerator {
  /**
   * Generate complete WGSL code for the compute gradient shader
   */
  static generateComputeShader(scene: SDFScene): string {
    const primitiveLibrary = this.generatePrimitiveLibrary();
    const operationLibrary = this.generateOperationLibrary();
    const paramsStruct = this.generateParamsStruct(scene);
    const sceneSDF = this.generateSceneSDF(scene);

    return `
// Primitive SDF gradient functions
${primitiveLibrary}

// Operation functions
${operationLibrary}

// Uniforms
struct Uniforms {
  viewProj: mat4x4f,
  cameraPos: vec3f,
  time: f32,
}

// Scene parameters
${paramsStruct}

struct PointData {
  positions: array<vec3f>,
}

struct GradientData {
  results: array<vec4f>, // (distance, gradient.xyz)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: PointData;
@group(0) @binding(2) var<storage, read_write> gradients: GradientData;
@group(0) @binding(3) var<uniform> sceneParams: SceneParams;

// Scene evaluation
${sceneSDF}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let numPoints = arrayLength(&points.positions);

  if (index >= numPoints) {
    return;
  }

  // Get point position in world space
  let pos = points.positions[index];

  // Evaluate SDF gradient at this position
  let result = sceneSDF(pos);

  // Store result (distance, gradient.xyz)
  gradients.results[index] = result;
}
`;
  }

  /**
   * Generate WGSL primitive gradient function library
   */
  private static generatePrimitiveLibrary(): string {
    return `
// Sphere SDF with gradient
fn sdgSphere(p: vec3f, r: f32) -> vec4f {
  let d = length(p);
  let dist = d - r;
  let grad = p / max(d, 0.0001);
  return vec4f(dist, grad);
}

// Box SDF with gradient
fn sdgBox(p: vec3f, b: vec3f) -> vec4f {
  let q = abs(p) - b;
  let dist = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);

  // Gradient computation
  let s = sign(p);
  let w = max(q, vec3f(0.0));
  let g = max(q.x, max(q.y, q.z));

  var grad: vec3f;
  if (g > 0.0) {
    grad = s * normalize(w);
  } else {
    // Inside the box - gradient points to nearest face
    if (q.x > q.y && q.x > q.z) {
      grad = vec3f(s.x, 0.0, 0.0);
    } else if (q.y > q.z) {
      grad = vec3f(0.0, s.y, 0.0);
    } else {
      grad = vec3f(0.0, 0.0, s.z);
    }
  }

  return vec4f(dist, grad);
}

// Torus SDF with gradient
fn sdgTorus(p: vec3f, t: vec2f) -> vec4f {
  let q = vec2f(length(p.xz) - t.x, p.y);
  let dist = length(q) - t.y;

  // Gradient computation
  let px = p.xz;
  let lxz = length(px);
  let dir = vec2f(lxz - t.x, p.y);
  let ldir = length(dir);

  var grad: vec3f;
  if (lxz > 0.0001 && ldir > 0.0001) {
    let dxz = px / lxz;
    let dd = dir / ldir;
    grad = vec3f(dxz.x * dd.x, dd.y, dxz.y * dd.x);
  } else {
    grad = vec3f(0.0, 1.0, 0.0);
  }

  return vec4f(dist, grad);
}

// Capsule SDF with gradient
fn sdgCapsule(p: vec3f, h: f32, r: f32) -> vec4f {
  let halfH = h * 0.5;
  let py = clamp(p.y, -halfH, halfH);
  let q = p - vec3f(0.0, py, 0.0);
  let d = length(q);
  let dist = d - r;

  var grad: vec3f;
  if (d > 0.0001) {
    grad = q / d;
  } else {
    grad = vec3f(0.0, sign(p.y), 0.0);
  }

  return vec4f(dist, grad);
}
`;
  }

  /**
   * Generate operation functions (union, intersection, etc.)
   */
  private static generateOperationLibrary(): string {
    return `
// Union (min)
fn opUnion(a: vec4f, b: vec4f) -> vec4f {
  if (a.x < b.x) {
    return a;
  } else {
    return b;
  }
}

// Intersection (max)
fn opIntersection(a: vec4f, b: vec4f) -> vec4f {
  if (a.x > b.x) {
    return a;
  } else {
    return b;
  }
}

// Subtraction
fn opSubtraction(a: vec4f, b: vec4f) -> vec4f {
  let negB = vec4f(-b.x, -b.yzw);
  return opIntersection(a, negB);
}

// Smooth minimum with gradient (approximation)
fn opSmoothUnion(a: vec4f, b: vec4f, k: f32) -> vec4f {
  let h = max(k - abs(a.x - b.x), 0.0) / k;
  let m = h * h * h * 0.5;
  let s = m * k * (1.0 / 6.0);

  // Blend distance
  let dist = min(a.x, b.x) - s;

  // Blend gradient (weighted by distance)
  let t = clamp(0.5 + 0.5 * (b.x - a.x) / k, 0.0, 1.0);
  let grad = mix(a.yzw, b.yzw, t);

  return vec4f(dist, grad);
}
`;
  }

  /**
   * Generate the SceneParams struct
   */
  private static generateParamsStruct(scene: SDFScene): string {
    const primitives = scene.getPrimitives();
    const operations = scene.getOperations();

    if (primitives.length === 0) {
      return "struct SceneParams {\n  _dummy: f32,\n}";
    }

    const lines: string[] = ["struct SceneParams {"];

    // Add primitive parameters
    for (const prim of primitives) {
      const paramNames = prim.getParamNames();

      for (const name of paramNames) {
        if (name.endsWith("_center")) {
          lines.push(`  ${name}: vec3f,`);
          lines.push(`  _pad_${name}: f32,`); // alignment padding
        } else if (name.endsWith("_size")) {
          lines.push(`  ${name}: vec3f,`);
          lines.push(`  _pad_${name}: f32,`);
        } else if (name.endsWith("_radii")) {
          lines.push(`  ${name}: vec2f,`);
          lines.push(`  _pad_${name}: vec2f,`);
        } else if (name.endsWith("_params")) {
          lines.push(`  ${name}: vec2f,`);
          lines.push(`  _pad_${name}: vec2f,`);
        } else if (name.endsWith("_radius")) {
          lines.push(`  ${name}: f32,`);
        }
      }
    }

    // Add operation parameters
    for (const op of operations) {
      const paramNames = op.getParamNames();
      for (const name of paramNames) {
        lines.push(`  ${name}: f32,`);
      }
    }

    lines.push("}");
    return lines.join("\n");
  }

  /**
   * Generate the sceneSDF function
   */
  private static generateSceneSDF(scene: SDFScene): string {
    const root = scene.getRoot();

    if (!root) {
      return `
fn sceneSDF(p: vec3f) -> vec4f {
  return vec4f(1000.0, 0.0, 1.0, 0.0); // Empty scene
}`;
    }

    let varCounter = 0;
    const getVarName = () => `result_${varCounter++}`;

    const lines: string[] = ["fn sceneSDF(p: vec3f) -> vec4f {"];

    const traverse = (node: SceneNode): string => {
      if (node.type === "primitive") {
        const prim = node.primitive;
        const varName = getVarName();
        const primType = prim.getType();

        let functionCall = "";

        switch (primType) {
          case PrimitiveType.Sphere:
            functionCall = `sdgSphere(p - sceneParams.${prim.id}_center, sceneParams.${prim.id}_radius)`;
            break;
          case PrimitiveType.Box:
            functionCall = `sdgBox(p - sceneParams.${prim.id}_center, sceneParams.${prim.id}_size)`;
            break;
          case PrimitiveType.Torus:
            functionCall = `sdgTorus(p - sceneParams.${prim.id}_center, sceneParams.${prim.id}_radii)`;
            break;
          case PrimitiveType.Capsule:
            functionCall = `sdgCapsule(p - sceneParams.${prim.id}_center, sceneParams.${prim.id}_params.x, sceneParams.${prim.id}_params.y)`;
            break;
        }

        lines.push(`  let ${varName} = ${functionCall};`);
        return varName;
      } else {
        // Operation node
        const childVars = node.children.map(traverse);
        const varName = getVarName();
        const opType = node.operation.getType();

        let operationCall = "";

        switch (opType) {
          case OperationType.Union:
            operationCall = `opUnion(${childVars[0]}, ${childVars[1]})`;
            break;
          case OperationType.Intersection:
            operationCall = `opIntersection(${childVars[0]}, ${childVars[1]})`;
            break;
          case OperationType.Subtraction:
            operationCall = `opSubtraction(${childVars[0]}, ${childVars[1]})`;
            break;
          case OperationType.SmoothUnion:
            const paramName = node.operation.getParamNames()[0];
            operationCall = `opSmoothUnion(${childVars[0]}, ${childVars[1]}, sceneParams.${paramName})`;
            break;
        }

        lines.push(`  let ${varName} = ${operationCall};`);
        return varName;
      }
    };

    const finalVar = traverse(root);
    lines.push(`  return ${finalVar};`);
    lines.push("}");

    return lines.join("\n");
  }
}
