from tensorflow.keras.layers import (
    Dense,
    Masking,
    Layer,
    RNN,
    # SimpleRNN,
    # SimpleRNNCell,
)
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from functools import wraps
from tensorflow.python.keras import (
    initializers,
    regularizers,
    constraints,
    activations,
)
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.python.keras import backend as K


class CTRNNCell(DropoutRNNCellMixin, Layer):
    def __init__(
        self,
        units,
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        dt=0.005,
        tau=0.1,
        **kwargs,
    ):

        super(CTRNNCell, self).__init__(**kwargs)

        # Set time constant
        self.alpha = dt / tau

        # Set parameters
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=None,
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=None,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=None,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training
        )

        # Get internal state
        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask

        # Get output
        # output = (1 - self.alpha * h) + self.alpha * K.dot(prev_output, self.recurrent_kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)

        # Pass through activation function
        if self.activation is not None:
            output = self.activation(output)

        new_state = [output] if nest.is_sequence(states) else output
        return output, new_state

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }

        base_config = super(CTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CTRNN(RNN):
    def __init__(
        self,
        units,
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        dt=0.005,
        tau=0.1,
        **kwargs,
    ):

        cell = CTRNNCell(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
            dt=dt,
            tau=tau,
        )

        super(CTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }
        base_config = super(CTRNN, self).get_config()
        del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if "implementation" in config:
            config.pop("implementation")
        print(config.keys())
        return cls(**config)


# ---------------------------------------------------------------------------- #
#                                 Layers Makers                                #
# ---------------------------------------------------------------------------- #


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
        return CTRNN(
            units=layer_params["units"],
            activation=layer_params["activation"],
            input_shape=input_shape,
            batch_input_shape=batch_input_shape,
            return_sequences=True,
            name="Recurrent",
            trainable=layer_params["trainable"],
            kernel_initializer=layer_params["kernel_initializer"],
            stateful=layer_params["stateful"],
            dt=layer_params["dt"],
            tau=layer_params["tau"],
        )
    else:
        return CTRNN(
            units=layer_params["units"],
            activation=layer_params["activation"],
            batch_input_shape=(1, 1, batch_input_shape[2]),
            return_sequences=True,
            name="Recurrent",
            trainable=False,
            kernel_initializer=layer_params["kernel_initializer"],
            stateful=True,
            dt=layer_params["dt"],
            tau=layer_params["tau"],
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
