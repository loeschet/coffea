from ..util import awkward
from ..util import numpy as np
from ..util import USE_CUPY
import time

if USE_CUPY:
    import cupy as np

class lookup_base(object):
    # base class for all objects that do some sort of value or function lookup
    def __init__(self):
        pass

    def __call__(self, *args):
        t0 = time.time()
        inputs = list(args)
        offsets = None
        # TODO: check can use offsets (this should always be true for striped)
        # Alternatively we can just use starts and stops
        for i in range(len(inputs)):
            if isinstance(inputs[i], awkward.JaggedArray):
                if offsets is not None and offsets.base is not inputs[i].offsets.base:
                    if type(offsets) is int:
                        raise Exception('Do not mix JaggedArrays and numpy arrays when calling derived class of lookup_base')
                    elif type(offsets) is np.ndarray and offsets.base is not inputs[i].offsets.base:
                        raise Exception('All input jagged arrays must have a common structure (offsets)!')
                offsets = inputs[i].offsets
                inputs[i] = inputs[i].content
            elif isinstance(inputs[i], np.ndarray):
                if offsets is not None:
                    if type(offsets) is np.ndarray:
                        raise Exception('do not mix JaggedArrays and numpy arrays when calling a derived class of lookup_base')
                offsets = -1
        retval = self._evaluate(*tuple(inputs))
        if offsets is not None and type(offsets) is not int:
            retval = awkward.JaggedArray.fromoffsets(offsets, retval)
        t1 = time.time()
        print("prof", self.__class__, self.signature, self._formula_str, t1 - t0)
        return retval

    def _evaluate(self, *args):
        raise NotImplementedError
