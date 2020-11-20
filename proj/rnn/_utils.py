from pathlib import Path
from pyinspect import install_traceback
import joblib
from rich import print
import pandas as pd

from proj import paths

install_traceback()


class RNNPaths:
    """
        Helper class that takes care of setting up paths
        used for RNN training and datasets, helps loading and saving
        RNN models and data normalizers etc...
    """

    def __init__(self, dataset_name="", winstor=False):
        self.main_fld = (
            Path(paths.rnn) if not winstor else Path(paths.winstor_rnn)
        )

        # path to unprocssed data
        self.trials_folder = self.main_fld / "training_data"

        # Path to pre-processed data
        self.dataset_folder = self.main_fld / dataset_name
        self.dataset_folder.mkdir(exist_ok=True)

        # Paths to datasets and normalizers
        self.dataset_train_path = self.dataset_folder / "training_data.h5"
        self.dataset_test_path = self.dataset_folder / "test_data.h5"

        self.train_scaler_path = self.dataset_folder / "train_scaler.joblib"
        self.test_scaler_path = self.dataset_folder / "test_scler.joblib"

    def load_normalizers(self, from_model_folder=False):
        train = joblib.load(self.train_scaler_path)
        test = joblib.load(self.test_scaler_path)
        return train, test

    def save_normalizers(self, train, test):
        joblib.dump(train, str(self.train_scaler_path))
        joblib.dump(test, str(self.test_scaler_path))

    def load_dataset(self):
        train = pd.read_hdf(self.dataset_train_path, key="hdf")
        test = pd.read_hdf(self.dataset_test_path, key="hdf")
        return train, test

    def save_dataset(self, train, test):
        train.to_hdf(self.dataset_train_path, key="hdf")
        test.to_hdf(self.dataset_test_path, key="hdf")

        print(f"Saved at {self.dataset_train_path}, {len(train)} trials")
