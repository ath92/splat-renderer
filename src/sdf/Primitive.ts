import { vec3 } from "gl-matrix";

export const PrimitiveType = {
  Sphere: "sphere",
  Box: "box",
  Torus: "torus",
  Capsule: "capsule",
} as const;

export type PrimitiveType = (typeof PrimitiveType)[keyof typeof PrimitiveType];

export interface PrimitiveParams {
  id?: string;
  position?: vec3;
}

export interface SphereParams extends PrimitiveParams {
  radius?: number;
}

export interface BoxParams extends PrimitiveParams {
  size?: vec3;
}

export interface TorusParams extends PrimitiveParams {
  majorRadius?: number;
  minorRadius?: number;
}

export interface CapsuleParams extends PrimitiveParams {
  height?: number;
  radius?: number;
}

export interface AABB {
  min: vec3;
  max: vec3;
}

export abstract class Primitive {
  id: string;
  position: vec3;

  private static nextId = 0;

  constructor(id?: string) {
    this.id = id || `prim_${Primitive.nextId++}`;
    this.position = vec3.fromValues(0, 0, 0);
  }

  abstract getType(): PrimitiveType;
  abstract getParamNames(): string[];
  abstract getParamValues(): number[];
  abstract clone(): Primitive;
  abstract getAABB(): AABB;
  abstract getSurfaceArea(): number;
}

export class Sphere extends Primitive {
  radius: number;

  constructor(params: SphereParams = {}) {
    super(params.id);
    this.position = params.position
      ? vec3.clone(params.position)
      : vec3.fromValues(0, 0, 0);
    this.radius = params.radius ?? 0.5;
  }

  getType(): PrimitiveType {
    return PrimitiveType.Sphere;
  }

  getParamNames(): string[] {
    return [`${this.id}_center`, `${this.id}_radius`];
  }

  getParamValues(): number[] {
    return [...this.position, this.radius];
  }

  clone(): Sphere {
    return new Sphere({
      id: this.id,
      position: vec3.clone(this.position),
      radius: this.radius,
    });
  }

  getAABB(): AABB {
    const r = this.radius;
    return {
      min: vec3.fromValues(
        this.position[0] - r,
        this.position[1] - r,
        this.position[2] - r,
      ),
      max: vec3.fromValues(
        this.position[0] + r,
        this.position[1] + r,
        this.position[2] + r,
      ),
    };
  }

  getSurfaceArea(): number {
    return 4 * Math.PI * this.radius * this.radius;
  }
}

export class Box extends Primitive {
  size: vec3;

  constructor(params: BoxParams = {}) {
    super(params.id);
    this.position = params.position
      ? vec3.clone(params.position)
      : vec3.fromValues(0, 0, 0);
    this.size = params.size
      ? vec3.clone(params.size)
      : vec3.fromValues(0.5, 0.5, 0.5);
  }

  getType(): PrimitiveType {
    return PrimitiveType.Box;
  }

  getParamNames(): string[] {
    return [`${this.id}_center`, `${this.id}_size`];
  }

  getParamValues(): number[] {
    return [...this.position, 0, ...this.size, 0]; // padding for alignment
  }

  clone(): Box {
    return new Box({
      id: this.id,
      position: vec3.clone(this.position),
      size: vec3.clone(this.size),
    });
  }

  getAABB(): AABB {
    return {
      min: vec3.fromValues(
        this.position[0] - this.size[0],
        this.position[1] - this.size[1],
        this.position[2] - this.size[2],
      ),
      max: vec3.fromValues(
        this.position[0] + this.size[0],
        this.position[1] + this.size[1],
        this.position[2] + this.size[2],
      ),
    };
  }

  getSurfaceArea(): number {
    const w = this.size[0] * 2;
    const h = this.size[1] * 2;
    const d = this.size[2] * 2;
    return 2 * (w * h + w * d + h * d);
  }
}

export class Torus extends Primitive {
  majorRadius: number;
  minorRadius: number;

  constructor(params: TorusParams = {}) {
    super(params.id);
    this.position = params.position
      ? vec3.clone(params.position)
      : vec3.fromValues(0, 0, 0);
    this.majorRadius = params.majorRadius ?? 0.5;
    this.minorRadius = params.minorRadius ?? 0.2;
  }

  getType(): PrimitiveType {
    return PrimitiveType.Torus;
  }

  getParamNames(): string[] {
    return [`${this.id}_center`, `${this.id}_radii`];
  }

  getParamValues(): number[] {
    return [...this.position, 0, this.majorRadius, this.minorRadius, 0, 0]; // padding
  }

  clone(): Torus {
    return new Torus({
      id: this.id,
      position: vec3.clone(this.position),
      majorRadius: this.majorRadius,
      minorRadius: this.minorRadius,
    });
  }

  getAABB(): AABB {
    const outerRadius = this.majorRadius + this.minorRadius;
    return {
      min: vec3.fromValues(
        this.position[0] - outerRadius,
        this.position[1] - this.minorRadius,
        this.position[2] - outerRadius,
      ),
      max: vec3.fromValues(
        this.position[0] + outerRadius,
        this.position[1] + this.minorRadius,
        this.position[2] + outerRadius,
      ),
    };
  }

  getSurfaceArea(): number {
    return 4 * Math.PI * Math.PI * this.majorRadius * this.minorRadius;
  }
}

export class Capsule extends Primitive {
  height: number;
  radius: number;

  constructor(params: CapsuleParams = {}) {
    super(params.id);
    this.position = params.position
      ? vec3.clone(params.position)
      : vec3.fromValues(0, 0, 0);
    this.height = params.height ?? 1.0;
    this.radius = params.radius ?? 0.3;
  }

  getType(): PrimitiveType {
    return PrimitiveType.Capsule;
  }

  getParamNames(): string[] {
    return [`${this.id}_center`, `${this.id}_params`];
  }

  getParamValues(): number[] {
    return [...this.position, 0, this.height, this.radius, 0, 0]; // padding
  }

  clone(): Capsule {
    return new Capsule({
      id: this.id,
      position: vec3.clone(this.position),
      height: this.height,
      radius: this.radius,
    });
  }

  getAABB(): AABB {
    const halfH = this.height / 2;
    return {
      min: vec3.fromValues(
        this.position[0] - this.radius,
        this.position[1] - halfH - this.radius,
        this.position[2] - this.radius,
      ),
      max: vec3.fromValues(
        this.position[0] + this.radius,
        this.position[1] + halfH + this.radius,
        this.position[2] + this.radius,
      ),
    };
  }

  getSurfaceArea(): number {
    // Cylinder + two hemisphere caps
    return (
      2 * Math.PI * this.radius * this.height +
      4 * Math.PI * this.radius * this.radius
    );
  }
}

const _center: number[] = [];
const _currentScale: number[] = [];
export function scaleAABB(aabb: AABB, scale: number) {
  const center = vec3.scaleAndAdd(_center, aabb.min, aabb.max, 1 / 2);
  const currentScale = vec3.sub(_currentScale, aabb.max, aabb.min);
  return {
    min: vec3.scaleAndAdd([], center, currentScale, -scale / 2),
    max: vec3.scaleAndAdd([], center, currentScale, scale / 2),
  };
}
