from pathlib import Path
from pyinspect.utils import timestamp
from pyinspect._colors import orange, lightorange
from pyinspect import Report
from pyinspect import install_traceback
from pyinspect.utils import stringify

from fcutils.file_io.io import load_yaml, save_yaml

from proj import paths

install_traceback()


class RNNLog:
    base_config_path = "proj/rnn/rnn_config.yml"

    _history = {"lr": [], "loss": []}

    def __init__(self):
        self.main_fld = Path(paths.rnn)

        self.load_config()

        # make a folder
        self.folder = self.main_fld / f'{self.config["name"]}_{timestamp()}'
        self.folder.mkdir(exist_ok=True)

        # make useful paths
        self.trials_folder = self.main_fld / "training_data"
        self.dataset_folder = (
            self.main_fld / f'dataset_{self.config["dataset_normalizer"]}'
        )
        self.dataset_folder.mkdir(exist_ok=True)

        self.dataset_path = self.dataset_folder / "training_data.h5"
        self.input_scaler_path = self.dataset_folder / "input_scaler.gz"
        self.output_scaler_path = self.dataset_folder / "output_scaler.gz"

        self.rnn_weights_save_path = self.folder / "trained_model.h5"

        # create report
        self.log = Report(
            title=self.config["name"],
            accent=orange,
            color=lightorange,
            dim=orange,
        )

    def load_config(self):
        self.config = load_yaml(self.base_config_path)

    def save_config(self):
        save_path = self.folder / "config.yml"

        config = dict(
            name=self.config["name"],
            interactive=self.config["interactive"],
            training_data_description=self.config["training_data_description"],
            dataset_normalizer=self.config["dataset_normalizer"],
            n_trials_training=self.config["n_trials_training"],
            n_trials_test=self.config["n_trials_test"],
            single_trial_mode=self.config["single_trial_mode"],
            BATCH=self.config["BATCH"],
            T=self.config["T"],
            dt=self.config["dt"],
            tau=self.config["tau"],
            EPOCHS=self.config["EPOCHS"],
            steps_per_epoch=self.config["steps_per_epoch"],
            lr_schedule=dict(
                boundaries=self.config["lr_schedule"]["boundaries"],
                values=self.config["lr_schedule"]["values"],
                name=self.config["lr_schedule"]["name"],
            ),
            optimizer=self.config["optimizer"],
            clipvalue=self.config["clipvalue"],
            amsgrad=self.config["amsgrad"],
            layers=self.config["layers"],
            loss=self.config["loss"],
        )

        save_yaml(str(save_path), config)

    def save_log(self, log):
        log = stringify(log, maxlen=-1)
        savepath = self.folder / "log.txt"

        with open(str(savepath), "w") as f:
            f.write(log)
