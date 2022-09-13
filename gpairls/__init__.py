# setup GPAIRLS logger
import sys
from loguru import logger


logger.remove()
log_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}"
logger.add(sys.stderr, format=log_fmt)