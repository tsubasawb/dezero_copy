import numpy as np
from torch import broadcast_to
from dezero.core import Function, as_variable
from dezero import utils

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)

class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
        
class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        return transpose(gy)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)
    
    def backward(self, gy):
        return sum_to(gy, self.x_shape)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)
    
    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)

class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x):
    return Transpose()(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def matmul(x, W):
    return MatMul()(x, W)