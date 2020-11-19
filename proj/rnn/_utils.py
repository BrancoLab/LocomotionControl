from pathlib import Path
from pyinspect import install_traceback
import joblib

from proj import paths

install_traceback()


class RNNPaths:
    """
        Helper class that takes care of setting up paths
        used for RNN training and datasets, helps loading and saving
        RNN models and data normalizers etc...
    """

    def __init__(self, mk_dir=True, folder=None, winstor=False):
        self.main_fld = (
            Path(paths.rnn) if not winstor else Path(paths.winstor_rnn)
        )

        # path to unprocssed data
        self.trials_folder = self.main_fld / "training_data"

        # Path to pre-processed data
        self.dataset_folder = self.main_fld / f"dataset_scaled"
        self.dataset_folder.mkdir(exist_ok=True)

        # Paths to datasets and normalizers
        self.dataset_train_path = self.dataset_folder / "training_data.h5"
        self.dataset_test_path = self.dataset_folder / "test_data.h5"

        self.input_scaler_path = self.dataset_folder / "input_scaler.joblib"
        self.output_scaler_path = self.dataset_folder / "output_scaler.joblib"

    def load_normalizers(self, from_model_folder=False):
        inp = joblib.load(self.input_scaler_path)
        out = joblib.load(self.output_scaler_path)
        return inp, out

    def save_normalizers(self, _in, _out):
        joblib.dump(_in, str(self.input_scaler_path))
        joblib.dump(_out, str(self.output_scaler_path))
