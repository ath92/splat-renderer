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

/**
 * Wrap a primitive in a scene node (or pass through if already a node)
 */
export function primitive(prim: Primitive | SceneNode): SceneNode {
  if (typeof prim === "object" && "type" in prim) {
    return prim;
  }
  return { type: "primitive", primitive: prim };
}

/**
 * Combine two scene nodes with union (min operation)
 */
export function union(a: Primitive | SceneNode, b: Primitive | SceneNode): SceneNode {
  return {
    type: "operation",
    operation: new Union(),
    children: [primitive(a), primitive(b)],
  };
}

/**
 * Combine two scene nodes with intersection (max operation)
 */
export function intersection(a: Primitive | SceneNode, b: Primitive | SceneNode): SceneNode {
  return {
    type: "operation",
    operation: new Intersection(),
    children: [primitive(a), primitive(b)],
  };
}

/**
 * Subtract the second scene node from the first
 */
export function subtraction(a: Primitive | SceneNode, b: Primitive | SceneNode): SceneNode {
  return {
    type: "operation",
    operation: new Subtraction(),
    children: [primitive(a), primitive(b)],
  };
}

/**
 * Combine two scene nodes with smooth union
 * @param k Smoothness parameter (higher = smoother blend)
 */
export function smoothUnion(k: number, a: Primitive | SceneNode, b: Primitive | SceneNode): SceneNode {
  return {
    type: "operation",
    operation: new SmoothUnion(k),
    children: [primitive(a), primitive(b)],
  };
}

export class SDFScene {
  private root: SceneNode | null = null;
  private primitiveMap: Map<string, Primitive> = new Map();

  /**
   * Set the root of the scene graph
   */
  setRoot(node: SceneNode): void {
    this.root = node;
    this.primitiveMap.clear();
    this.collectPrimitives(node);
  }

  /**
   * Traverse the scene graph and collect all primitives
   */
  private collectPrimitives(node: SceneNode): void {
    if (node.type === "primitive") {
      this.primitiveMap.set(node.primitive.id, node.primitive);
    } else {
      node.children.forEach((child) => this.collectPrimitives(child));
    }
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
