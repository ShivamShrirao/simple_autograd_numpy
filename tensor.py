import numpy as np
from graphviz import Digraph

# Based on https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

class Tensor:
    def __init__(self, data, prev=(), op=lambda x: None, name=None, *args, **kwargs):
        self.data = np.asarray(data)
        self.prev = prev
        self.grad = 0
        self.op = op
        self.grad_fn = lambda x: None
        self.broadcast_dim = None
        self.name = name

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient

        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t.prev:
                    build_topo(p)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t.grad_fn(t.grad)

    def __repr__(self):
        r = repr(self.data)
        return r[:10].replace('array','tensor') + r[10:]

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.checkbroadcast(other)
        out = Tensor(self.data + other.data, (self, other), op=self.__add__)
        def grad_fn(gradient):
            self.grad += gradient if self.broadcast_dim is None else gradient.sum(axis=self.broadcast_dim, keepdims=True)
            other.grad += gradient if other.broadcast_dim is None else gradient.sum(axis=other.broadcast_dim, keepdims=True)
        out.grad_fn = grad_fn
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), op=self.__mul__)
        def grad_fn(gradient):
            self.grad += gradient * other.data
            other.grad += gradient * self.data
        out.grad_fn = grad_fn
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), op=self.__pow__)
        def grad_fn(gradient):
            self.grad += gradient * (other * (self.data ** (other-1)))
        out.grad_fn = grad_fn
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), op=self.__matmul__)
        def grad_fn(gradient):
            self.grad += gradient @ other.data.T
            other.grad += self.data.T @ gradient
        out.grad_fn = grad_fn
        return out
    
    def relu(self):
        out = Tensor(self.data*(self.data>0), (self,), op=self.relu)
        def grad_fn(gradient):
            self.grad += gradient * (out.data > 0)
        out.grad_fn = grad_fn
        return out

    def sin(self):
        out = Tensor(np.sin(self.data), (self,), op=self.sin)
        def grad_fn(gradient):
            self.grad += gradient * np.cos(self.data)
        out.grad_fn = grad_fn
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), op=self.exp)
        def grad_fn(gradient):
            self.grad += gradient * out.data
        out.grad_fn = grad_fn
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), op=self.log)
        def grad_fn(gradient):
            self.grad += gradient * (1. / self.data)
        out.grad_fn = grad_fn
        return out
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * self**-1

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def checkbroadcast(self, other):
        for n,(i,j) in enumerate(zip(self.shape, other.shape)):
            if i==j:
                continue
            if i<j:
                self.broadcast_dim = n
                break
            else:
                other.broadcast_dim = n
                break

    def generate_graph(self):
        dot = Digraph(comment='DAG')
        visited = set()
        def build_graph(t):
            if t not in visited:
                visited.add(t)
                if t.name:
                    nm = t.name
                    shape = "box"
                    color = ""
                else:
                    nm = t.op.__name__
                    shape = ""
                    color = "lightblue2"
                dot.node(str(hash(t)), nm, shape=shape, color=color, style='filled')
                for p in t.prev:
                    dot.edge(str(hash(p)), str(hash(t)))
                    build_graph(p)
        build_graph(self)
        return dot