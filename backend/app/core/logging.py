import sys
from loguru import logger
from app.core.config import settings


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="WARNING",
        format="{time} | {level} | {name}:{line} — {message}",
    )
    return logger


app_logger = setup_logging()
