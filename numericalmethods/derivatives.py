import math
import numpy as np
from numericalmethods.limits import Limits

class Derivative(Limits):
  def __init__(self, func, epsilon=1e-6):
    self.dx = epsilon
    self.func = func
    self.order = 0

  def differentiate(self, point):
    def fn(dx):
      return (self.func(point+self.dx) - self.func(point))/(self.dx)
    self.order += 1
    return self.evaluate_limit(fn, 0)[1]

class Element:
  def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0
    self._backward = lambda:None
    self._prev = set(_children)
    self._op = _op

  def __repr__(self):
    return f"Element(data={self.data})" 

  def __rmul__(self, other):
    return self * other

  def __add__(self, other):
    other = other if isinstance(other, Element) else Element(other)
    out = Element(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)
    
  def __mul__(self, other):
    other = other if isinstance(other, Element) else Element(other)
    out = Element(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), 'only ints and floats'
    out = Element(self.data**other, (self, ), f'**{other}')
    def _backward():
      self.grad += other * self.data ** (other -1) * out.grad
    out._backward = _backward
    return out

  def __truediv__(self, other):
    return self * other**-1

  def exp(self):
    x = self.data
    out = Element(math.exp(x), (self, ), 'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def sin(self):
    x = self.data
    out = Element(math.sin(x), (self,), 'sin')
    def _backward():
      self.grad += math.cos(out.data) * out.grad 
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

if (__name__ == '__main__'):
  def f(x):
    return x**2
  
  dx = Derivative(f)
  dx_f = dx.differentiate(4)
  print(dx_f)