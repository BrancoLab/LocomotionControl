from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback

install_traceback(keep_frames=0, hide_locals=True, relevant_only=True)

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)


from tpd import recorder

try:
    recorder.start(base_folder=".", folder_name="logs", timestamp=False)
except PermissionError:
    recorder.start(base_folder=".", folder_name="logs", timestamp=True)
