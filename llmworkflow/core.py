from abc import ABC, abstractmethod
from typing import Any, List, Dict, Callable


class Node(ABC):
    def __init__(self, name: str):
        self.name = name
        self.ctx = None

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass

    def set_context(self, ctx: Dict[str, Any]):
        self.ctx = ctx


class CompositeNode(Node):
    def __init__(self, name: str, nodes: List[Node]):
        super().__init__(name)
        self.nodes = nodes

    def set_context(self, ctx: Dict[str, Any]):
        super().set_context(ctx)
        for node in self.nodes:
            node.set_context(ctx)


class Function(Node):
    def __init__(self, name: str, func: Callable, per_item: bool = False):
        super().__init__(name)
        self.func = func
        self.per_item = per_item

    def __call__(self, data: Any) -> Any:
        if isinstance(data, list) and self.per_item:
            return list(map(self.func, data))
        return self.func(data)


class Print(Node):
    def __call__(self, data: Any) -> Any:
        print(data)
        return data

class Chain(CompositeNode):
    def __call__(self, data: Any) -> Any:
        for node in self.nodes:
            data = node(data)
        return data


class Workflow:
    def __init__(self, name: str):
        self.name = name
        self.nodes = []
        self.ctx = {}

    def add(self, node: Node):
        self.nodes.append(node)
        node.set_context(self.ctx)

    def run(self, initial_data: Any) -> Any:
        data = initial_data
        for node in self.nodes:
            print(f"Running node: {node.name}")
            try:
                data = node(data)
            except Exception as e:
                print(f"Error in node {node.name}: {str(e)}")
                raise
        return data
