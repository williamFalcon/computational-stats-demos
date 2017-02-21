from enum import Enum
import numpy as np


# type_map = {
#     TypeEnum.int32: np.int32,
#     TypeEnum.i64: np.int64,
#     TypeEnum.f32: np.float32,
#     TypeEnum.f64: np.float64,
# }

class RootGate(object):

    def __add__(self, other):
        return AddGate(self, other)

    def __sub__(self, other):
        return SubtractGate(self, other)

    def __mul__(self, other):
        return MultiplyGate(self, other)

    def __pow__(self, power, modulo=None):
        return PowerGate(self, power)

    def __floordiv__(self, other):
        return DivideGate(self, other)

    def __truediv__(self, other):
        return DivideGate(self, other)

class tensor(RootGate):
    def __init__(self, dtype, name=None):
        self.name = dtype if name is None else name
        self.dtype = dtype
        self.value = None
        self.a = None
        self.b = None

    def __str__(self):
        return 'ADTensor(name:{}, dtype: {})'.format(self.name, self.dtype)

    def forward_pass(self):
        return self.value

int32 = 'int32'
float64 = 'float64'



class AddGate(RootGate):

    def __init__(self, a, b):
        super(AddGate, self).__init__()
        self.a = a
        self.b = b
        self.name = 'AddGate'
        self.f = None

    def __str__(self):
        return 'AddGate(a:{}, b:{})'.format(self.a, self.b)

    def __add__(self, other):
        return AddGate(self, other)

    def forward_pass(self):

        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a + forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz, wrt):
        """
        f(x) = a + x
        df/dx = 1
        :param dz:
        :return:
        """
        dx = 1
        return dz * dx

class MultiplyGate(RootGate):

    def __init__(self, a, b):
        super(MultiplyGate, self).__init__()
        self.a = a
        self.b = b
        self.name = 'MultiplyGate'
        self.f = None

    def __str__(self):
        return 'MultiplyGate(a:{}, b:{})'.format(self.a, self.b)

    def __mul__(self, other):
        return MultiplyGate(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a * forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz, wrt):
        """
        f(x) = ax
        df/dx = a
        :param dz:
        :return:
        """
        # df/dx = ax = a

        # backward_a = self.a.backward_pass(dz, wrt) if is_valid_node(self.a) else self.a
        # backward_a = self.a.backward_pass(dz, wrt) if is_valid_node(self.a) else self.a
        dx = self.a
        if wrt == self.a:
            dx = self.b

        # df/dx = x*x = x^2 = 2x
        if self.a == self.b:
            dx = 2 * dx.value

        if is_valid_node(dx):
            dx = dx.value

        return dz * dx

class PowerGate(RootGate):

    def __init__(self, a, b):
        super(PowerGate, self).__init__()
        self.a = a
        self.b = b
        self.name = 'PowerGate'
        self.f = None

    def __str__(self):
        return 'PowerGate(a:{}, b:{})'.format(self.a, self.b)

    def __pow__(self, power, modulo=None):
        return PowerGate(self, power)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a ** forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz):
        """
        f(x) = a + x
        df/dx = 1
        :param dz:
        :return:
        """
        dx = 1
        return dz * dx

class DivideGate(RootGate):

    def __init__(self, a, b):
        super(DivideGate, self).__init__()
        self.a = a
        self.b = b
        self.name = 'DivideGate'
        self.f = None

    def __str__(self):
        return 'DivideGate(a:{}, b:{})'.format(self.a, self.b)

    def __floordiv__(self, other):
        return DivideGate(self, other)

    def __truediv__(self, other):
        return DivideGate(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a / forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz):
        """
        f(x) = a + x
        df/dx = 1
        :param dz:
        :return:
        """
        dx = 1
        return dz * dx

class SubtractGate(RootGate):

    def __init__(self, a, b):
        super(SubtractGate, self).__init__()
        self.a = a
        self.b = b
        self.name = 'SubtractGate'
        self.f = None

    def __str__(self):
        return 'SubtractGate(a:{}, b:{})'.format(self.a, self.b)

    def __sub__(self, other):
        return SubtractGate(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a - forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz):
        """
        f(x) = a + x
        df/dx = 1
        :param dz:
        :return:
        """
        dx = 1
        return dz * dx

def gradients(fn, vars):

    def compute(**kwargs):
        head = fn
        stack = [head]
        while len(stack) > 0:
            top = stack.pop()
            if top.a and is_valid_node(top.a):
                stack.append(top.a)
                if top.a.name in kwargs:
                    top.a.value = kwargs[top.a.name]

            if top.b and is_valid_node(top.b):
                stack.append(top.b)
                if top.b.name in kwargs:
                    top.b.value = kwargs[top.b.name]

        z = head.forward_pass()
        grads = []
        for var in vars:
            grad = head.backward_pass(1.0, var)
            grads.append(grad)

        return grads

    return compute




def compile(output):

    def compute(**kwargs):
        head = output
        stack = [head]
        while len(stack) > 0:
            top = stack.pop()
            if top.a and is_valid_node(top.a):
                stack.append(top.a)
                if top.a.name in kwargs:
                    top.a.value = kwargs[top.a.name]

            if top.b and is_valid_node(top.b):
                stack.append(top.b)
                if top.b.name in kwargs:
                    top.b.value = kwargs[top.b.name]

        answer = head.forward_pass()
        return answer
    return compute

def is_valid_node(a):
    return type(a) != int and type(a) != float







