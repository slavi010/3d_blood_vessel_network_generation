"""
Module for sparse arrays using dictionaries. Inspired in part
by ndsparse (https://launchpad.net/ndsparse) by Pim Schellart

Jan Erik Solem, Feb 9 2010.
solem@maths.lth.se (bug reports and feedback welcome)

Edited by: Sviatoslav Besnard, Feb 8 2023.
"""
import itertools

import numpy


class VirtualProjection(object):
    """ Virtual projection of array. """

    def __init__(self, data, from_to_index):
        self.data = data  # data is not slicable
        self.__from_to_index = from_to_index  # tuple of : (from,to) indices or (index) indice
        self.shape = []
        for i in range(len(from_to_index)):
            if isinstance(from_to_index[i], tuple):
                self.shape.append(from_to_index[i][1] - from_to_index[i][0])
            else:
                self.shape.append(1)
        # fill in missing dimensions
        for i in range(len(self.shape), len(data.shape)):
            self.shape.append(data.shape[i])
            self.__from_to_index.append((0, data.shape[i]))

        self.shape = tuple(self.shape)

    def __getitem__(self, keys):
        return self.get_item(keys, True)

    @staticmethod
    def from_slice(data, slices, creat_virtual: bool):
        """ Create virtual projection from slices. """
        return VirtualProjection(data, []).get_item(slices, creat_virtual)

    def get_data(self):
        return self.get_item([], False)

    def get_item(self, keys, creat_virtual: bool):
        param_slices = []
        if isinstance(keys, tuple) or isinstance(keys, list):
            for k in keys:
                param_slices.append(k)
                if isinstance(k, slice):
                    if k.step is not None:
                        raise ValueError('Step not supported.')
        elif isinstance(keys, slice) or isinstance(keys, int):
            param_slices.append(keys)
        # merging the existing slices with the parameter slices
        new_slices = []
        used_param_slices = 0

        for i in range(len(self.__from_to_index)):
            if isinstance(self.__from_to_index[i], tuple):
                if used_param_slices < len(param_slices):
                    if isinstance(param_slices[i], slice):
                        # relative to the from index
                        new_slices.append((self.__from_to_index[i][0] + param_slices[i].start,
                                           self.__from_to_index[i][0] + param_slices[i].stop))
                    else:
                        new_slices.append(self.__from_to_index[i][0] + param_slices[i])
                    used_param_slices += 1

                else: # no more parameter slices
                    new_slices.append(self.__from_to_index[i])
            else:
                new_slices.append(self.__from_to_index[i])
        # fill in missing dimensions
        for i in range(len(new_slices), len(self.data.shape)):
            if used_param_slices < len(param_slices):
                if isinstance(param_slices[i], slice):
                    new_slices.append((param_slices[i].start, param_slices[i].stop))
                else:
                    new_slices.append(param_slices[i])
                used_param_slices += 1
            else:
                new_slices.append((0, self.data.shape[i]))

        if used_param_slices < len(param_slices):
            raise ValueError('Too many indices for array.')
        elif len(new_slices) != len(self.data.shape):
            raise ValueError('Too few indices for array.')

        if creat_virtual:
            return self.__class__(self.data, new_slices)
        else:
            iterators = [(range(i[0], i[1]) if isinstance(i, tuple) else range(i, i+1)) for i in new_slices]
            cumulative_iterators = itertools.product(*iterators)

            return [self.data.data.get(k, self.data.default) for k in cumulative_iterators]


class sparray(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """

    def __init__(self, shape, origin=0, default=0, dtype=complex):

        self.default = default  # default value of non-assigned elements
        self.shape = tuple(shape)
        if isinstance(origin, int):
            self.origin = tuple([origin] * len(shape))
        else:
            self.origin = origin
        self.ndim = len(shape)
        self.dtype = dtype
        self.data = {}

    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self.data[index] = value

    def __getitem__(self, key):
        """
        Get value at position given in index, if slice is given
        return a flat iterator array.
        """
        if isinstance(key, tuple) and any([isinstance(k, slice) for k in key]) or isinstance(key, slice):
            return VirtualProjection.from_slice(self, key, False)
        else:
            return self.data.get(key, self.default)

    def virtual_projection(self):
        """ Return a virtual projection of the array (read-only). """
        return VirtualProjection.from_slice(self, [], True)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if index in self.data:
            del (self.data[index])

    def __abs__(self):
        """ Absolute value (element wise). """
        if self.dtype == complex:
            dtype = float
        else:
            dtype = self.dtype
        out = self.__class__(self.shape, origin=self.origin, dtype=dtype)
        for k in self.data.keys():
            out.data[k] = numpy.abs(self.data[k])
        return out

    def __add__(self, other):
        """ Add two arrays or add a scalar to all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] + other
            # out.__default = self.__default + other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] + other.default
                out.default = self.default + other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val + other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ Subtract two arrays or substract a scalar to all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] - other
            # out.__default = self.__default - other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] - other.default
                out.default = self.default - other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val - other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __mul__(self, other):
        """ Multiply two arrays (element wise) or multiply a scalar to all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] * other
            # out.__default = self.__default * other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] * other.default
                out.default = self.default * other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val * other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """ Divide two arrays (element wise).
            Type of division is determined by dtype.
            Or divide by a scalar all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] / other
            # out.__default = self.__default / other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] / other.default
                out.default = self.default / other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val / other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __truediv__(self, other):
        """ Divide two arrays (element wise).
            Type of division is determined by dtype.
            Or divide by a scalar all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] / other
            # out.__default = self.__default / other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] / other.default
                out.default = self.default / other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val / other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __floordiv__(self, other):
        """ Floor divide ( // ) two arrays (element wise)
        or floor divide by a scalar all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] // other
            # out.__default = self.__default // other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] // other.default
                out.default = self.default // other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val // other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __mod__(self, other):
        """ mod of two arrays (element wise)
        or mod of all elements of an array and a scalar. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] % other
            # out.__default = self.__default % other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] % other.default
                out.default = self.default % other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val % other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __pow__(self, other):
        """ power (**) of two arrays (element wise)
        or power of all elements of an array with a scalar. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.data = self.data.copy()
            for k in self.data.keys():
                out.data[k] = self.data[k] ** other
            # out.__default = self.__default ** other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.data = self.data.copy()
                for k in set.difference(set(out.data.keys()), set(other.data.keys())):
                    out.data[k] = out.data[k] ** other.default
                out.default = self.default ** other.default
                for k in other.data.keys():
                    old_val = out.data.setdefault(k, self.default)
                    out.data[k] = old_val ** other.data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __iadd__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] + other
            # self.__default = self.__default + other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] + other.default
                self.default = self.default + other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val + other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __isub__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] - other
            # self.__default = self.__default - other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] - other.default
                self.default = self.default - other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val - other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __imul__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] * other
            # self.__default = self.__default * other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] * other.default
                self.default = self.default * other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val * other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __idiv__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] / other
            # self.__default = self.__default / other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] / other.default
                self.default = self.default / other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val / other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __itruediv__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] / other
            # self.__default = self.__default / other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] / other.default
                self.default = self.default / other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val / other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __ifloordiv__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] // other
            # self.__default = self.__default // other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] // other.default
                self.default = self.default // other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val // other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __imod__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] % other
            # self.__default = self.__default % other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] % other.default
                self.default = self.default % other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val % other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __ipow__(self, other):
        if numpy.isscalar(other):
            for k in self.data.keys():
                self.data[k] = self.data[k] ** other
            # self.__default = self.__default ** other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.data.keys()), set(other.data.keys())):
                    self.data[k] = self.data[k] ** other.default
                self.default = self.default ** other.default
                for k in other.data.keys():
                    old_val = self.data.setdefault(k, self.default)
                    self.data[k] = old_val ** other.data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. ' + str(self.shape) + ' versus ' + str(other.shape))

    def __str__(self):
        return str(self.dense())

    def dense(self):
        """ Convert to dense NumPy array. """
        out = self.default * numpy.ones(self.shape, dtype=self.dtype)
        for ind in self.data:
            shift = tuple(numpy.asarray(ind) - numpy.asarray(self.origin))
            out[shift] = self.data[ind]
        return out

    def sum(self):
        """ Sum of elements."""
        s = self.default * numpy.array(self.shape).prod()
        for ind in self.data:
            s += (self.data[ind] - self.default)
        return s

    def get_items(self):
        """ Get multi_indices list with their values. Default values of
        non-assigned elements are not included"""
        return list(self.data.items())

    def get_values(self):
        """ Get all values. Default values of
        non-assigned elements are not included"""
        return list(self.data.values())

    def get_multi_indices(self):
        """ Get all multi_indices. Default values of
        non-assigned elements are not included"""
        return list(self.data.keys())

    def sort(self):
        """ Sort multi_index and values. """
        items = self.get_items()
        items.sort()
        self.data = {}
        for item in items:
            self.data[item[0]] = item[1]
        return self

    def conj(self):
        """ Conjugate value (element wise). """
        out = self.__class__(self.shape, origin=self.origin, \
                             default=self.default, dtype=self.dtype)
        for k in self.data.keys():
            out.data[k] = numpy.conj(self.data[k])
        return out

    def hierarchy_augmentation(self, default=1, copy=True):
        """ Hierarchy augmentation includes the boundary of
        the multi index set. """
        multi_index = self.get_multi_indices()
        new_multi_index = []
        m = numpy.asarray([3] * self.ndim)
        surrounding = []
        it = numpy.nditer(numpy.ones(tuple(m)), flags=['multi_index'])
        while not it.finished:
            surrounding.append(numpy.asarray(it.multi_index) \
                               - numpy.ones(self.ndim, dtype=int))
            it.iternext()
        for index in multi_index:
            for surr_j in surrounding:
                new_multi_index.append(tuple(numpy.asarray(index) + surr_j))
        new_multi_index = list(dict.fromkeys(new_multi_index))
        augm_factor = numpy.asarray([2] * self.ndim).astype(int)
        augm_shape = tuple(numpy.asarray(self.shape).astype(int) + augm_factor)
        augm_origin = tuple(numpy.asarray(self.origin).astype(int) - numpy.ones(self.ndim).astype(int))
        augm = self.__class__(augm_shape, origin=augm_origin, \
                              default=0, dtype=self.dtype)
        for index in new_multi_index:
            augm[index] = default
        if copy:
            for index in multi_index:
                augm[index] = self[index]
        augm.sort()
        return augm



if __name__ == '__main__':
    s = sparray((3, 3), default=0, dtype=int)
    s[0, 0] = 1
    s[1, 1] = 2
    s[2, 2] = 3
    print(s)
    print(s[0:2, 0:2])