from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback

install_traceback()

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)


from tpd import recorder

try:
    recorder.start(base_folder=".", name="logs", timestamp=False)
except  PermissionError:
    recorder.start(base_folder=".", name="logs", timestamp=True)
