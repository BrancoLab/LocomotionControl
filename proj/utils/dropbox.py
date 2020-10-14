import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError

from pathlib import Path

try:
    from proj.secrets import DB_TOKEN
except ModuleNotFoundError:
    DB_TOKEN = None
from proj.paths import db_app


def upload_folder(dbx, fld, base):
    fld = Path(fld)
    base = Path(base)

    # loop subfolders
    for subf in fld.glob("*"):
        if not subf.is_dir():
            continue
        upload_folder(dbx, subf, base / subf.name)

    for f in fld.glob("*.*"):
        if not f.is_file():
            continue

        dest = base / f.name
        dbx.upload_file(f, dest)


class DropBoxUtils:
    def __init__(self):
        # log in
        try:
            self.db = dropbox.Dropbox(DB_TOKEN)
        except AuthError:
            raise ValueError("Failed to access dropbox app")

        # keep ref to path where db app is stored locally
        self.local = Path(db_app)

    def _fix_path(self, path):
        return str(self.local / path)

    def _check_path(self, path):
        path = str(path)
        path = path.replace("\\", "/")
        if path[0] != "/":
            return "/" + path
        else:
            return path

    def download_file(self, filepath, dest_path=None):
        """
            Downloads a file from self.local/filepath.
            if dest_path is not None the file is saved there
        """
        if dest_path is None:
            try:
                self.db.files_download(self._fix_path(filepath))
            except ApiError as e:
                raise ValueError(f"Failed to download {filepath}: {e}")
        else:
            raise NotImplementedError

    def upload_file(self, source, dest):
        try:
            with open(source, "rb") as f:
                self.db.files_upload(
                    f.read(),
                    self._check_path(dest),
                    mode=WriteMode("overwrite"),
                    autorename=True,
                )
        except ApiError as e:
            raise ValueError(f"Failed to upload file {source}\n\n {e}")

    # def upload_folder(self, source, dest):
    #     source = Path(source)
    #     dest = Path(dest)

    #     # loop subfolders
    #     for subf in source.glob('*'):
    #         if not subf.is_dir(): continue
    #         print(subf)
    #         self.upload_folder(subf, dest / subf.name)

    # # loop files
    # for f in source.glob('*.*'):
    #     if not f.is_file(): continue

    #     dest = dest / f.name
    #     self.upload_file(f, dest)
