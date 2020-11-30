import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
from pathlib import Path
from slack import WebClient
from slack.errors import SlackApiError

from control.paths import db_app

try:
    from control.secrets import SLACK_TOKEN, SLACK_USER_ID, DB_TOKEN
except ModuleNotFoundError:
    SLACK_TOKEN = None
    SLACK_USER_ID = None
    DB_TOKEN = None


def send_slack_message(message):
    client = WebClient(token=SLACK_TOKEN)

    try:
        client.chat_postMessage(channel=SLACK_USER_ID, text=message)
    except SlackApiError as e:
        print(f"Got an error: {e.response['error']}")


def upload_folder(dbx, fld, base):
    fld = Path(fld)
    base = Path(base)

    # loop subfolders
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
