from .lookup_base import lookup_base

from ..util import awkward
from ..util import numpy as np
from copy import deepcopy

import scipy
import scipy.interpolate
from scipy.interpolate import interp1d

from .jme_standard_function import searchsorted_wrapped

from ..util import USE_CUPY

if USE_CUPY:
    import cupy as np
    import cupy

def masked_bin_eval(dim1_indices, dimN_bins, dimN_vals):
    dimN_indices = np.empty_like(dim1_indices)
    for i in np.unique(dim1_indices):
        idx = np.where(dim1_indices == i)
        dimN_indices[idx] = np.clip(searchsorted_wrapped(dimN_bins[i],
                                                    dimN_vals[idx],
                                                    side='right') - 1,
                                    0, len(dimN_bins[i]) - 2)
    return dimN_indices

if USE_CUPY:
    class interp1d:
        def __init__(self, x, y):
            self.x = cupy.array(x)
            self.y = cupy.array(y)
            self.ydiffs = cupy.diff(self.y)
            self.xdiffs = cupy.diff(self.x)
            self.delta = self.ydiffs / self.xdiffs
    
            #self.scipy_interp1d = scipy.interpolate.interp1d(x, y)
    
        def __call__(self, xvals):
            idx = searchsorted_wrapped(self.x, xvals, asnumpy=False)
           
            xshifts =  self.x[idx] - xvals 
            yvals = self.y[idx] 
            yvals_new = yvals - self.delta[idx-1]*xshifts
            
            #yvals_scipy = self.scipy_interp1d(xvals)
            #assert(np.sum(np.power(yvals_new - yvals_scipy,2)) < 1e-1)
            return yvals_new
 
class jec_uncertainty_lookup(lookup_base):
    """
    This class defines a lookup table for jet energy scale uncertainties.

    The uncertainty values can be looked up with a call as follows:
        junc_lut = jec_uncertainty_lookup()
        uncertainties = junc_lut(JetProperty1=jet.property1,...)

    "uncertainties" will be of the same shape as the input jet properties.

    The list of required jet properties are given in junc_lut.signature
    """
    def __init__(self, formula, bins_and_orders, knots_and_vars):
        """
        The constructor takes the output of the "convert_junc_txt_file"
        text file converter, which returns a formula, bins, and an interpolation table.
        """
        super(jec_uncertainty_lookup, self).__init__()
        self._dim_order = bins_and_orders[1]
        self._bins = bins_and_orders[0]
        self._eval_vars = knots_and_vars[1]
        self._eval_knots = knots_and_vars[0]['knots']
        self._eval_downs = []
        self._eval_ups = []
        self._formula_str = formula.strip('"')
        self._formula = None
        if self._formula_str != 'None' and self._formula_str != '':
            raise Exception('jet energy uncertainties have no formula!')

        for binname in self._dim_order[1:]:
            binsaslists = self._bins[binname].tolist()
            self._bins[binname] = [np.array(bins) for bins in binsaslists]

        # convert downs and ups into interp1ds
        # (yes this only works for one binning dimension right now, fight me)
        for bin in range(self._bins[self._dim_order[0]].size - 1):
            self._eval_downs.append(interp1d(self._eval_knots,
                                             knots_and_vars[0]['downs'][bin]))
            self._eval_ups.append(interp1d(self._eval_knots,
                                           knots_and_vars[0]['ups'][bin]))

        # get the jit to compile if we've got more than one bin dim
        if len(self._dim_order) > 1:
            masked_bin_eval(np.array([0]), self._bins[self._dim_order[1]], np.array([0.0]))

        self._signature = deepcopy(self._dim_order)
        for eval in self._eval_vars:
            if eval not in self._signature:
                self._signature.append(eval)
        self._dim_args = {self._dim_order[i]: i for i in range(len(self._dim_order))}
        self._eval_args = {}
        for i, argname in enumerate(self._eval_vars):
            self._eval_args[argname] = i + len(self._dim_order)
            if argname in self._dim_args.keys():
                self._eval_args[argname] = self._dim_args[argname]

    def _evaluate(self, *args):
        """ uncertainties = f(args) """
        bin_vals = {argname: args[self._dim_args[argname]] for argname in self._dim_order}
        eval_vals = {argname: args[self._eval_args[argname]] for argname in self._eval_vars}
        
        #if USE_CUPY:
        #    bin_vals = {k: cupy.array(v) for k, v in bin_vals.items()}
        #    eval_vals = {k: cupy.array(v) for k, v in eval_vals.items()}

        # lookup the bins that we care about
        dim1_name = self._dim_order[0]
        dim1_indices = np.clip(searchsorted_wrapped(self._bins[dim1_name],
                                               bin_vals[dim1_name],
                                               side='right', asnumpy=not USE_CUPY) - 1,
                               0, self._bins[dim1_name].size - 2)
        #if USE_CUPY:
        #    dim1_indices = cupy.array(dim1_indices)

        # get clamp values and clip the inputs
        outs = np.ones(shape=(args[0].size, 2), dtype=np.float)
        for i in np.unique(dim1_indices):
            i = int(i)
            mask = np.where(dim1_indices == i)
            vals = np.clip(eval_vals[self._eval_vars[0]][mask],
                           self._eval_knots[0], self._eval_knots[-1])
            outs[:, 0][mask] += self._eval_ups[i](vals)
            outs[:, 1][mask] -= self._eval_downs[i](vals)

        if USE_CUPY:
            outs = cupy.asnumpy(outs)

        return outs

    @property
    def signature(self):
        """ the list of all needed jet properties to be passed as kwargs to this lookup """
        return self._signature

    def __repr__(self):
        out = 'binned dims   : %s\n' % (self._dim_order)
        out += 'eval vars     : %s\n' % (self._eval_vars)
        out += 'signature     : (%s)\n' % (','.join(self._signature))
        return out
