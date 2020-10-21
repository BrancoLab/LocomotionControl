from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras import models

from pyinspect._colors import orange, lightorange

from proj.rnn._rnn import (
    make_dense_layer,
    make_rnn_layer,
    make_masking_layer,
    changes_batch_size,
)
from proj.rnn._utils import RNNLog
from proj.rnn.task import ControlTask
from proj.utils.progress_bars import train_progress, CustomCallback
from proj.rnn._rnn import CTRNN


class ControlRNN(RNNLog):
    """
        This class takes care of building RNN models
        for training and prediction as well as saving/loading them.
    """

    def __init__(self, *args, **kwargs):
        RNNLog.__init__(self, *args, **kwargs)

        # Get task representations
        self.task = ControlTask(
            *args,
            dt=self.config["dt"],
            tau=self.config["tau"],
            T=self.config["T"],
            N_batch=self.config["BATCH"],
            **kwargs,
        )

        # Get input/output shapes
        x, y, mask, trial_params = self.task.get_trial_batch()

        self.STEP = x.shape[1]
        self.N_inputs = x.shape[2]
        self.N_outputs = y.shape[2]
        self.batch_input_shape = x.shape

    def load_model(self, custom_objects=None):
        return models.load_model(
            self.rnn_weights_save_path, custom_objects=custom_objects
        )

    @changes_batch_size(1)
    def make_model_for_prediction(self):
        """
            Creates a new instance of the RNN which can predict single steps
        """
        # Get input shapes
        x, y, mask, trial_params = self.task.get_trial_batch()

        # Load trained RNN
        trained_rnn = self.load_model(custom_objects={"CTRNN": CTRNN})

        # Make new stateful model
        model = keras.Sequential()

        # Add masking layer
        model.add(
            make_masking_layer(
                (1, x.shape[2]), batch_input_shape=(1, 1, x.shape[2])
            )
        )

        # Add RNN layer
        layer_params = self.config["layers"][0]
        model.add(make_rnn_layer(layer_params, x.shape, for_prediction=True))

        # Add Dense layer
        layer_params = self.config["layers"][1]
        model.add(make_dense_layer(layer_params))

        # Set weights and return
        model.build()
        model.set_weights(trained_rnn.get_weights())
        return model

    def _get_scheduler(self):
        schedule = PiecewiseConstantDecay(
            self.config["lr_schedule"]["boundaries"],
            self.config["lr_schedule"]["values"],
        )
        self.log.add(
            f'Learning rate scheduler: {self.config["lr_schedule"]["name"]}\n bounds: {self.config["lr_schedule"]["boundaries"]} - vals: {self.config["lr_schedule"]["values"]} '
        )

        return schedule

    def _get_optimizer(self, schedule):
        if self.config["optimizer"] != "Adam":
            raise NotImplementedError(
                f"Needs to be setup to work with optimizer {self.config['optimizer']}"
            )
        optimizer = Adam(
            learning_rate=schedule,
            name="Adam",
            clipvalue=self.config["clipvalue"],
            amsgrad=self.config["amsgrad"],
            clipnorm=self.config["clipnorm "],
        )

        self.log.add(
            f'Optimizer: {self.config["optimizer"]} with clipvalue: {self.config["clipvalue"]}'
        )
        self.log.add(f'Loss function: {self.config["loss"]}')

        return optimizer

    def make_model(self):
        """ 
            Creates a Keras Sequential RNN model + optimizer and scheduler
            for training an RNN.
        """
        self.log.add(f"[b {orange}]Creating model")

        # scheduler
        schedule = self._get_scheduler()

        # optimizer
        optimizer = self._get_optimizer(schedule)

        # ----------------------------------- model ---------------------------------- #

        self.log.spacer(2)
        self.log.add(f"[{orange}]Layers")

        model = keras.Sequential()

        # Add masking layer
        model.add(
            make_masking_layer(
                (self.STEP, self.N_inputs), self.batch_input_shape
            )
        )

        # Add RNN and Dense layers
        for n, layer_params in enumerate(self.config["layers"]):
            if layer_params["name"] == "dense":
                l = make_dense_layer(layer_params)
            else:
                l = make_rnn_layer(
                    layer_params,
                    self.batch_input_shape,
                    input_shape=(self.STEP, self.N_inputs),
                )

            model.add(l)
            self.log.add(
                f'[green]Layer {n}[/green]  --  [b {lightorange}]{layer_params["name"]}[/b {lightorange}] - [blue]{layer_params["units"]}[/blue] units - [green]{layer_params["activation"]}[/green] activation'
            )

        # ---------------------------------- compile --------------------------------- #

        model.compile(loss=self.config["loss"], optimizer=optimizer)

        self.log.spacer(1)
        model.summary(print_fn=self.log.add)

        # Set callback function
        self.callback = CustomCallback(
            self.config["EPOCHS"],
            train_progress,
            self.config["steps_per_epoch"],
            schedule,
            self.log,
        )

        return model
