from loguru import logger
from rich.logging import RichHandler

from rnn.analysis.pipeline import Pipeline

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)
