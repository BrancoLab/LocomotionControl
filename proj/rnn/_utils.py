from pathlib import Path
from pyinspect.utils import timestamp
from pyinspect._colors import orange, lightorange
from pyinspect import Report
from pyinspect import install_traceback
from pyinspect.utils import stringify
import joblib
import shutil
import numpy as np

from proj import paths

install_traceback()


class RNNPaths:
    """
        Helper class that takes care of setting up paths
        used for RNN training and datasets, helps loading and saving
        RNN models and data normalizers etc...
    """

    _name = "RNNPaths"
    _history = {"lr": [], "loss": []}

    def __init__(self, mk_dir=True, folder=None, winstor=False):
        self.main_fld = (
            Path(paths.rnn) if not winstor else Path(paths.winstor_rnn)
        )

        name = self.name or self._name

        # make a folder
        if folder is None:
            self.folder = self.main_fld / f"{name}_{timestamp()}"
            if mk_dir:
                self.folder.mkdir(exist_ok=True)
        else:
            self.folder = Path(folder)

        # make useful paths
        self.trials_folder = self.main_fld / "training_data"
        self.dataset_folder = self.main_fld / f"dataset_scaled"
        self.dataset_folder.mkdir(exist_ok=True)

        self.dataset_train_path = self.dataset_folder / "training_data.h5"
        self.dataset_test_path = self.dataset_folder / "test_data.h5"

        self.input_scaler_path = self.dataset_folder / "input_scaler.joblib"
        self.output_scaler_path = self.dataset_folder / "output_scaler.joblib"
        self.input_scaler_data_path = (
            self.dataset_folder / "input_scaler_data.npy"
        )
        self.output_scaler_data_path = (
            self.dataset_folder / "output_scaler_data.npy"
        )

        self.input_scaler_folder_path = self.folder / "input_scaler.joblib"
        self.output_scaler_folder_path = self.folder / "output_scaler.joblib"
        self.input_scaler_data_folder_path = (
            self.folder / "input_scaler_data.npy"
        )
        self.output_scaler_data_folder_path = (
            self.folder / "output_scaler_data.npy"
        )

        self.rnn_weights_save_path = self.folder / "trained_model.h5"

        # create report
        self.log = Report(
            title=name, accent=orange, color=lightorange, dim=orange,
        )

    def save_log(self, log):
        log = stringify(log, maxlen=-1)
        savepath = self.folder / "log.txt"

        with open(str(savepath), "w", encoding="utf-8") as f:
            f.write(log)

    def save_data_to_training_folder(self):
        # Save regularizers to RNN folder
        _in, _out = self.load_normalizers()
        joblib.dump(_in, str(self.input_scaler_folder_path))
        joblib.dump(_out, str(self.output_scaler_folder_path))

        # Save data to RNN folder
        training_new_path = str(self.folder / "training_data.h5")
        test_new_path = str(self.folder / "test_data.h5")

        # Copy datasets and normalizers
        shutil.copy(self.dataset_train_path, training_new_path)
        shutil.copy(self.dataset_test_path, test_new_path)

        shutil.copy(self.input_scaler_path, self.input_scaler_folder_path)
        shutil.copy(self.output_scaler_path, self.output_scaler_folder_path)

        shutil.copy(
            self.input_scaler_data_path, self.input_scaler_data_folder_path
        )
        shutil.copy(
            self.output_scaler_data_path, self.output_scaler_data_folder_path
        )

    def load_normalizers(self, from_model_folder=False):
        if from_model_folder:
            _inp = joblib.load(self.input_scaler_folder_path)
            _out = joblib.load(self.output_scaler_folder_path)
            _in_data = np.load(self.input_scaler_data_folder_path)
            _out_data = np.load(self.output_scaler_data_folder_path)
        else:
            _inp = joblib.load(self.input_scaler_path)
            _out = joblib.load(self.output_scaler_path)
            _in_data = np.load(self.input_scaler_data_path)
            _out_data = np.load(self.output_scaler_data_path)

        _inp = _inp.fit(_in_data)
        _out = _out.fit(_out_data)

        return _inp, _out

    def save_normalizers(self, _in, _out, _in_data, _out_data):
        joblib.dump(_in, str(self.input_scaler_path))
        joblib.dump(_out, str(self.output_scaler_path))

        np.save(str(self.input_scaler_data_path), _in_data)
        np.save(str(self.output_scaler_data_path), _out_data)
