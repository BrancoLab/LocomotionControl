from tensorflow.keras.layers import Dense, SimpleRNN, Masking
from functools import wraps


def make_masking_layer(input_shape, batch_input_shape=None):
    return Masking(
        mask_value=0.0,
        input_shape=input_shape,
        batch_input_shape=batch_input_shape,
        name="mask",
    )


def make_dense_layer(layer_params):
    return Dense(
        units=layer_params["units"],
        activation=layer_params["activation"],
        name="Dense",
        trainable=layer_params["trainable"],
        kernel_initializer=layer_params["kernel_initializer"],
    )


def make_rnn_layer(
    layer_params, batch_input_shape, input_shape=None, for_prediction=False
):
    if not for_prediction:
        return SimpleRNN(
            units=layer_params["units"],
            activation=layer_params["activation"],
            input_shape=input_shape,
            batch_input_shape=batch_input_shape,
            return_sequences=True,
            name="Recurrent",
            trainable=layer_params["trainable"],
            kernel_initializer=layer_params["kernel_initializer"],
            stateful=layer_params["stateful"],
        )
    else:
        return SimpleRNN(
            units=layer_params["units"],
            activation=layer_params["activation"],
            batch_input_shape=(1, 1, batch_input_shape[2]),
            return_sequences=True,
            name="Recurrent",
            trainable=False,
            kernel_initializer=layer_params["kernel_initializer"],
            stateful=True,
        )


def changes_batch_size(n_batch):
    def inner(method):
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            _n_batch = instance.task.N_batch
            instance.task.N_batch = n_batch
            method(instance, *args, **kwargs)
            instance.task.N_batch = _n_batch

        return wrapper

    return inner
