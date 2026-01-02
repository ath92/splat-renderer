export const OperationType = {
  Union: "union",
  Intersection: "intersection",
  Subtraction: "subtraction",
  SmoothUnion: "smooth_union",
} as const;

export type OperationType = typeof OperationType[keyof typeof OperationType];

export abstract class Operation {
  abstract getType(): OperationType;
  abstract getParamNames(): string[];
  abstract getParamValues(): number[];
}

export class Union extends Operation {
  getType(): OperationType {
    return OperationType.Union;
  }

  getParamNames(): string[] {
    return [];
  }

  getParamValues(): number[] {
    return [];
  }
}

export class Intersection extends Operation {
  getType(): OperationType {
    return OperationType.Intersection;
  }

  getParamNames(): string[] {
    return [];
  }

  getParamValues(): number[] {
    return [];
  }
}

export class Subtraction extends Operation {
  getType(): OperationType {
    return OperationType.Subtraction;
  }

  getParamNames(): string[] {
    return [];
  }

  getParamValues(): number[] {
    return [];
  }
}

export class SmoothUnion extends Operation {
  k: number;
  private static nextId = 0;
  id: string;

  constructor(k: number = 0.1) {
    super();
    this.k = k;
    this.id = `smin_${SmoothUnion.nextId++}`;
  }

  getType(): OperationType {
    return OperationType.SmoothUnion;
  }

  getParamNames(): string[] {
    return [`${this.id}_k`];
  }

  getParamValues(): number[] {
    return [this.k];
  }
}
