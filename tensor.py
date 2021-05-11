import numpy as np

class Tensor:
    def __init__(self, data, prev=(), op=None, *args, **kwargs):
        self.data = data
        self.prev = prev
        self.grad = 0
        self.op = op
        self.grad_fn = lambda x: None
        self.broadcast_dim = None
    
    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient
        self.grad_fn(self.grad)
        self.grad_fn = lambda x: None
        for p in self.prev:
            p.backward(p.grad)

    def __repr__(self):
        r = repr(self.data)
        r = r[:10].replace('array','tensor') + r[10:]
        return r
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.checkbroadcast(other)
        out = Tensor(self.data + other.data, (self, other), op='+')
        def grad_fn(gradient):
            self.grad += gradient if self.broadcast_dim is None else gradient.sum(axis=self.broadcast_dim, keepdims=True)
            other.grad += gradient if other.broadcast_dim is None else gradient.sum(axis=other.broadcast_dim, keepdims=True)
        out.grad_fn = grad_fn
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), op='*')
        def grad_fn(gradient):
            self.grad += gradient * other.data
            other.grad += gradient * self.data
        out.grad_fn = grad_fn
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), op='*')
        def grad_fn(gradient):
            self.grad += gradient * (other * (self.data ** (other-1)))
        out.grad_fn = grad_fn
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), op='@')
        def grad_fn(gradient):
            self.grad += gradient @ other.data.T
            other.grad += self.data.T @ gradient
        out.grad_fn = grad_fn
        return out
    
    def relu(self):
        out = Tensor(self.data*(self.data>0), (self,), op='relu')
        def grad_fn(gradient):
            self.grad += gradient * (out.data > 0)
        out.grad_fn = grad_fn
        return out

    def sin(self):
        out = Tensor(np.sin(self.data), (self,), op='sin')
        def grad_fn(gradient):
            self.grad += gradient * np.cos(self.data)
        out.grad_fn = grad_fn
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), op='exp')
        def grad_fn(gradient):
            self.grad += gradient * out.data
        out.grad_fn = grad_fn
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), op='log')
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