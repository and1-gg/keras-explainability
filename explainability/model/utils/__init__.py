from .fuse_batchnorm import fuse_batchnorm
from .remove_activation import remove_activation, remove_sigmoid, remove_softmax
from .reference import ReferenceActivations, ReferenceShift, \
                       find_linear_dense_head, output_coefficients, \
                       restructure_for_reference
