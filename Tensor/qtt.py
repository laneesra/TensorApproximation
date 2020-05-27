import numpy as np
import time
from tensor import Tensor


class QTT:
    def __init__(self, f, a, b, d, type=None):
        assert(f is None or type is None)
        if f is not None:
            self.f = f
        else:
            if type == 'sqrt':
                self.f = lambda x: x ** 0.5
            elif type == '1/x':
                self.f = lambda x: 1 / x
        self.a = a
        self.b = b
        self.d = d
        self.V = np.zeros([2 for _ in range(self.d)])

    def get_x(self, i):
        return self.a + (self.b - self.a) * i / 2 ** self.d

    @staticmethod
    def to_bin(num):
        b = []
        while num:
            b.append(num % 2)
            num //= 2
        return b

    @staticmethod
    def to_dec(bin):
        num = 0
        for i in range(len(bin)):
            if bin[i]:
                num += 2 ** i
        return num

    def get_index(self, V, i):
        if len(V.shape) > 1:
            self.get_index(V[0], i + [0])
            self.get_index(V[1], i + [1])
        else:
            V[0] = self.f(self.get_x(self.to_dec(i + [0])))
            V[1] = self.f(self.get_x(self.to_dec(i + [1])))
            return V

    def build(self):
        self.get_index(self.V[0], [0])
        self.get_index(self.V[1], [1])


def rand_shape(low=5, high=15):
    return np.random.randint(low, high)


if __name__ == '__main__':
    tensor = Tensor(np.random.rand(rand_shape(), rand_shape(), rand_shape(), rand_shape(), rand_shape()))
    start = time.time()
    tt = tensor.tt_factorization(0.01, 'svd')
    print('shape: ', tensor.T.shape)
    print('error svd:', tensor.frobenius_norm(np.subtract(tensor.T, tt)))

    start = time.time()
    tt = tensor.tt_factorization(0.01, 'aca')
    print('error aca:', tensor.frobenius_norm(np.subtract(tensor.T, tt)))