

"""
This package implements autodifferentiation using
the backprop algorithm.

It's not meant to be performant (aka, no cython or c)
it's just a proof of concept for learning and understanding
how this is done in libraries such as theano and tensorflow
"""


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