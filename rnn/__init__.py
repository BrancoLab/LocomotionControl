from rnn import dataset
from loguru import logger
from rich.logging import RichHandler

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)
