from copy import copy

import numpy as np


def _coords(*shape):
    '''Yields all indices of a shape.
    Essentially an n-dimensional `range`.

    Example:
        assert list(_coords(2,3)) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    Args:
        shape:
            The shape to iterate over. You can pass dimension sizes as
            separate arguments or as a single list, e.g. `_coords(1,2)` is
            equivalent to `_coords((1,2))`.

    Yields:
        All indices of a matrix with the given shape.
    '''
    if not isinstance(shape[0], int):
        assert len(shape) == 1
        shape = shape[0]
    if not shape:
        yield ()
    else:
        for i in range(shape[0]):
            for s in _coords(shape[1:]):
                yield (i,) + s


class RddTensor:
    '''A tensor abstraction over RDDs of the form `((i, j, ...), x)`
    where the key is the index of the value within the tensor.

    RddTensor's do not have a "shape" in the numpy sense. Instead, an RddTensor
    only has a number of dimensions (`ndim`), but each of those dimensions are
    considered to be infinite in size where values not explicitly listed in the
    underlying RDD are taken to be 0. Additionally, positions along dimensions
    need not be indexed by natural numbers; they can be indexed by arbitrary
    data, such as strings. You can even mix-and-match index types.

    This data structure is particularly well suited to bag-of-words matrices,
    which we would like to index by document ID and word and which are often
    highly sparse.
    '''

    def __init__(self, rdd, ndim=None, feature=0):
        '''Initialize an RddTensor from the given data.

        If you already know the number of dimensions (i.e. 2 for matrices),
        you should pass it in to avoid triggering a Spark job.

        The input RDD may have multiple features. In that case, the `feature`
        argument specifies the index of the feature to use in this matrix.
        For example, an RDD describing a 2D matrix with 3 features would
        have the form `((i, j), f0, f1, f2)`. If `feature` is 1, then the
        values of this matrix will be taken from `f1`.

        This initializer intentionally avoids validation, because our use case
        is unlikely to generate improper RDDs. Thus let's prefer performance.
        '''
        if not ndim: ndim = len(rdd.take(1)[0][0])
        self.rdd = rdd.map(lambda x: (x[0], x[feature+1]), preservesPartitioning=True)
        self.ndim = ndim

    @staticmethod
    def randint(ctx, low, high=None, shape=None):
        '''Create an RddTensor of random integers.

        The behavior mirrors numpy, e.g. if `high` is not given, `low` is used
        as the maximum value and 0 is used as the minimum.

        Args:
            ctx:
                The SparkContext used to create the underlying RDD.
            low:
                The minimum (inclusive) possible value in the tensor.
            high:
                The maximum (exclusive) possible value in the tensor.
            shape:
                The shape of the tensor.
        '''
        if not shape: shape = (1,)
        arr = np.random.randint(low, high, shape)
        return RddTensor.from_numpy(ctx, arr)

    @staticmethod
    def normal(ctx, loc=0.0, scale=1.0, shape=None):
        '''Draw an RddTensor from a normal (Gaussian) distribution.

        Args:
            ctx:
                The SparkContext used to create the underlying RDD.
            loc:
                The mean of the distribution.
            scale:
                The standard deviation of the distribution.
            shape:
                The shape of the tensor.
        '''
        if not shape: shape = (1,)
        arr = np.random.randint(loc, scale, shape)
        return RddTensor.from_numpy(ctx, arr)

    @staticmethod
    def from_numpy(ctx, arr):
        '''Create an RddTensor from a numpy array.

        Args:
            ctx:
                The SparkContext used to create the underlying RDD.
            arr:
                The numpy array to initialize the RddTensor.
        '''
        shape = arr.shape
        ndim = len(shape)
        keys = _coords(shape)
        r = ctx.parallelize(keys)
        r = r.map(lambda x: (x, arr[x]))
        return RddTensor(r, ndim)

    def to_numpy(self, dtype=None, return_coords=False):
        '''Collect this tensor into a numpy array.

        This is expensive. Only do this in testing.

        It is an error to collect an RddTensor to a numpy array if multiple
        indices exist along the same dimension which cannot be compared.

        Args:
            dtype:
                The dtype of the array to collect into.
            return_coords:
                If True, this method will also return the corresponding labels
                for each axis.
        '''
        rdd = self.rdd.sortByKey()

        coords = []
        keys = rdd.keys()
        for i in range(self.ndim):
            c = keys.map(lambda x: x[i]).distinct().collect()
            coords.append(c)
        shape = tuple(len(c) for c in coords)

        arr = np.zeros(shape, dtype=dtype)
        for x in rdd.collect():
            idx = tuple(coords[i].index(k) for i, k in enumerate(x[0]))
            arr[idx] = x[1]
        if return_coords:
            return arr, coords
        else:
            return arr

    @property
    def t(self):
        '''The transpose of this tensor.
        '''
        return self.transpose()

    def transpose(self, order=None):
        '''Reorder dimensions of the tensor.

        If a new order is not given, the first two dimensions are swapped,
        i.e. perform a standard matrix transpose.

        Args:
            order:
                The new order in terms of the current dimension indices.
        '''
        if not order:
            order = list(range(self.ndim))
            order[0] = 1
            order[1] = 0
        def t(x):
            # The *x[1:] allows this method to generalize to multiple values.
            # (Though not all methods support this generalization).
            y = tuple(x[0][i] for i in order)
            return (y, *x[1:])
        rdd = self.rdd.map(t)
        return RddTensor(rdd, self.ndim)

    def matmul(self, other):
        '''Multiply this matrix by another.
        '''
        # a has form ((A, B), X)
        # b has form ((B, C), Y)
        # c has form ((A, C), Z)  [the result]
        assert self.ndim == 2, 'must be a matrix'
        assert other.ndim == 2, 'must be a matrix'
        a = self.rdd
        b = other.rdd
        a = a.map(lambda x: (x[0][1], (x[0][0], x[1])))  # (B, (A, X))
        b = b.map(lambda x: (x[0][0], (x[0][1], x[1])))  # (B, (C, Y))
        c = a.join(b)  # (B, ((A, X), (C, Y)))
        c = c.mapValues(lambda x: ((x[0][0], x[1][0]), x[0][1] * x[1][1]))  # (B, ((A, C), Z))
        c = c.map(lambda x: x[1])  # ((A, C), Z)
        c = c.reduceByKey(lambda x, y: x + y)
        c = c.filter(lambda x: x[1] != 0)
        return RddTensor(c, self.ndim)

    def scale(self, scale):
        '''Multiply all elements of this tensor by a scalar.
        '''
        rdd = self.rdd.mapValues(lambda x: x * scale)
        rdd = rdd.filter(lambda x: x[1] != 0)
        return RddTensor(rdd, self.ndim)

    def hadamard(self, other):
        '''Perform an element-wise multiplication between two tensors.
        '''
        a = self.data
        b = self.data
        c = a.join(b)
        c.mapValues(np.prod)
        return RddTensor(c, self.ndim)

    def __getitem__(self, key):
        '''Implements element lookup only for complete keys.
        '''
        if len(key) == self.ndim:
            key = tuple(key)
            value = self.rdd.lookup(key)
            return value[0] if value else 0
        else:
            # TODO: We can support this if we have to.
            # Let me know if you need this --cbarrick
            raise Exception('must provide a full key')

    def __setitem__(self, key, value):
        '''Implements element assignment only for complete keys.
        '''
        if len(key) == self.ndim:
            key = tuple(key)
            ctx = self.rdd.ctx
            new = ctx.parallelize([(key, value)])
            rdd = self.rdd.filter(lambda x: x[0] != key)
            rdd = rdd.union(new)
            self.rdd = rdd
        else:
            # TODO: We can support this if we have to.
            # Let me know if you need this --cbarrick
            raise Exception('must provide a full key')

    def __add__(self, other):
        '''Implements the `+` operator as element-wise addition.
        '''
        assert self.ndim == other.ndim
        a = self.rdd
        b = other.rdd
        c = a.join(b)
        c = c.mapValues(lambda x: (x[0] or 0) + (x[1] or 0))
        c = c.filter(lambda x: x[1] != 0)
        return RddTensor(c, self.ndim)

    def __sub__(self, other):
        '''Implements the `-` operator as element-wise subtraction.
        '''
        assert self.ndim == other.ndim
        a = self.rdd
        b = other.rdd
        c = a.join(b)
        c = c.mapValues(lambda x: (x[0] or 0) - (x[1] or 0))
        c = c.filter(lambda x: x[1] != 0)
        return RddTensor(c, self.ndim)

    def __mul__(self, other):
        '''Implements the `*` operator as element-wise product or scale.
        '''
        if isinstance(other, RddTensor):
            return self.hadamard(other)
        else:
            return self.scale(other)

    def __matmul__(self, other):
        '''Implements the `@` operator as matrix product.
        '''
        return self.matmul(other)

    def __neg__(self):
        '''Implements the unary `-` operator as element-wise negation.
        '''
        return self.scale(-1)

    def __abs__(self):
        '''Implements the `abs()` operation as element-wise absolute value.
        '''
        rdd = self.rdd.mapValues(lambda x: abs(x))
        return RddTensor(rdd, self.ndim)
