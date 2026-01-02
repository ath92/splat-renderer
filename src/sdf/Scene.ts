import { Primitive } from "./Primitive";
import { Operation, Union, Intersection, Subtraction, SmoothUnion } from "./Operation";

export type SceneNode = PrimitiveNode | OperationNode;

export interface PrimitiveNode {
  type: "primitive";
  primitive: Primitive;
}

export interface OperationNode {
  type: "operation";
  operation: Operation;
  children: SceneNode[];
}

export class SDFScene {
  private root: SceneNode | null = null;
  private primitiveMap: Map<string, Primitive> = new Map();

  /**
   * Add a primitive to the scene. If this is the first element, it becomes the root.
   * Otherwise, it will be combined with the existing scene using the next operation.
   */
  add(primitive: Primitive): this {
    this.primitiveMap.set(primitive.id, primitive);

    const node: PrimitiveNode = {
      type: "primitive",
      primitive,
    };

    if (!this.root) {
      this.root = node;
    } else {
      // Store for next operation
      this.pendingNode = node;
    }

    return this;
  }

  private pendingNode: SceneNode | null = null;

  /**
   * Combine the previous elements with union (min operation)
   */
  union(): this {
    return this.applyOperation(new Union());
  }

  /**
   * Combine the previous elements with intersection (max operation)
   */
  intersection(): this {
    return this.applyOperation(new Intersection());
  }

  /**
   * Subtract the last added primitive from the previous scene
   */
  subtraction(): this {
    return this.applyOperation(new Subtraction());
  }

  /**
   * Combine the previous elements with smooth union
   */
  smoothUnion(k: number = 0.1): this {
    return this.applyOperation(new SmoothUnion(k));
  }

  private applyOperation(operation: Operation): this {
    if (!this.root || !this.pendingNode) {
      throw new Error("Need at least two elements to apply an operation");
    }

    const opNode: OperationNode = {
      type: "operation",
      operation,
      children: [this.root, this.pendingNode],
    };

    this.root = opNode;
    this.pendingNode = null;

    return this;
  }

  /**
   * Get a primitive by its ID to modify parameters
   */
  get(id: string): Primitive | undefined {
    return this.primitiveMap.get(id);
  }

  /**
   * Get all primitives in the scene
   */
  getPrimitives(): Primitive[] {
    return Array.from(this.primitiveMap.values());
  }

  /**
   * Get the root scene node
   */
  getRoot(): SceneNode | null {
    return this.root;
  }

  /**
   * Get all operations in the scene
   */
  getOperations(): Operation[] {
    const operations: Operation[] = [];

    const traverse = (node: SceneNode) => {
      if (node.type === "operation") {
        operations.push(node.operation);
        node.children.forEach(traverse);
      }
    };

    if (this.root) {
      traverse(this.root);
    }

    return operations;
  }

  /**
   * Check if the scene structure has changed (used to detect when shader needs recompilation)
   */
  getStructureHash(): string {
    const traverse = (node: SceneNode): string => {
      if (node.type === "primitive") {
        return `P:${node.primitive.getType()}:${node.primitive.id}`;
      } else {
        const childHashes = node.children.map(traverse).join(",");
        return `O:${node.operation.getType()}:(${childHashes})`;
      }
    };

    return this.root ? traverse(this.root) : "";
  }
}
