# -*- coding: utf-8 -*-
import sys
import numpy as np


class NamedArrayBase(np.ndarray):
    _names = []

    def __new__(cls, values=None, dtype=float, **kwargs):

        num_elements = len(cls._names)

        if values is None and kwargs:
            values = [kwargs.get(name, 0.0) for name in cls._names]
        elif values is not None:
            if len(values) != num_elements:
                raise ValueError(
                    f'Expected {num_elements} elements, got {len(values)}.')
        else:
            values = np.zeros(num_elements, dtype=dtype)

        try:
            obj = np.asarray(values, dtype=dtype).view(cls)
        except ValueError:
            obj = values
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __getattr__(self, attr):
        if attr in self._names:
            idx = self._names.index(attr)
            return self[idx]
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr in self._names:
            idx = self._names.index(attr)
            self[idx] = value
        else:
            super().__setattr__(attr, value)

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def __repr__(self):
        components = ', '.join(f'{name}={getattr(self, name)}'
                               for name in self._names)
        return f'{type(self).__name__}({components})'

    def __str__(self):
        return '[' + ', '.join(str(val) for val in self) + ']'


def NamedArrayFactory(names, default_dtype=float):
    class_name = 'NamedArray_' + '_'.join(names)
    cls = type(class_name, (NamedArrayBase,), {
        '_names': names,
        '__module__': __name__  # Make sure it points to the real module
    })

    # Register the class in the module’s namespace so pickle can find it
    sys.modules[__name__].__dict__[class_name] = cls
    return cls


Center = NamedArrayFactory(['x', 'y', 'z'])
Limits = NamedArrayFactory(['lmin', 'lmax'])
Opening = NamedArrayFactory(['left', 'right', 'bottom', 'top'])
Image2D = NamedArrayFactory(['width', 'height'], default_dtype=int)
