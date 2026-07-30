from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, \
                                    Dense, Dropout, GlobalAveragePooling3D, \
                                    Input, MaxPooling3D, Reshape
from tensorflow.keras.regularizers import l2
from typing import List, Tuple

from .model import Model
from .model_type import ModelType
from .utils import restrict_range


class SoftClassificationSFCN(Model):
    @property
    def type(self) -> ModelType:
        return ModelType.CLASSIFICATION

    def __init__(self, *, input_shape: Tuple[int, int, int] = (167, 212, 160), 
                 dropout: float = .0, weight_decay: float = .0, 
                 activation: str = 'relu', include_top: bool = True,
                 depths: List[int] = [32, 64, 128, 256, 256, 64],
                 prediction_range: Tuple[float, float] = (3, 95),
                 name: str = 'SoftClassification3DSFCN', 
                 weights: str = None):

        regularizer = l2(weight_decay) if weight_decay is not None else None

        inputs = Input(input_shape, name=f'{name}_inputs')

        x = inputs
        x = Reshape(input_shape + (1,), name=f'{name}_expand_dims')(x)

        for i in range(5):
            x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
                       activation=None, kernel_regularizer=regularizer,
                       bias_regularizer=regularizer,
                       name=f'{name}_block{i+1}_conv')(x)
            x = BatchNormalization(name=f'{name}_block{i+1}_norm')(x)
            x = Activation(activation,
                           name=f'{name}_block{i+1}_{activation}')(x)
            x = MaxPooling3D((2, 2, 2), name=f'{name}_block{i+1}_pool')(x)

        x = Conv3D(depths[-1], (1, 1, 1), padding='SAME', activation=None,
                   name=f'{name}_top_conv')(x)
        x = BatchNormalization(name=f'{name}_top_norm')(x)
        x = Activation(activation, name=f'{name}_top_{activation}')(x)
        x = GlobalAveragePooling3D(name=f'{name}_top_pool')(x)
        bottleneck = x

        x = Dropout(dropout, name=f'{name}_top_dropout')(x)
        x = Dense((prediction_range[1] - prediction_range[0]) + 1, 
                  activation='softmax', name=f'{name}_predictions')(x)

        if prediction_range is not None:
            x = restrict_range(x, *prediction_range, name=f'{name}_restrict')

        if not include_top:
            x = bottleneck

        super().__init__(inputs, x, weights=weights, include_top=include_top,
                         name=name)
