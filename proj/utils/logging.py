import logging
from rich.logging import RichHandler
from rich.console import Console

# supress warnings
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Set up RICH logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")
logging_console = Console(record=True)


def rich_to_txt(obj):
    logging_console.print(obj)
    return logging_console.export_text()


# Disable logging some packages
for module in [
    "matplotlib",
    "pandas",
    "numpy",
    "tensorflow",
    "pandas",
    "tables",
]:
    logger = logging.getLogger(module)
    logger.setLevel(logging.ERROR)
