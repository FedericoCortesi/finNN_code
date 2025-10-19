# utils/custom_formatter.py
import logging

class CustomFormatter(logging.Formatter):
    blue = "\x1b[34;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + base_format + reset,
        logging.INFO: grey + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.base_format)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str, level="DEBUG") -> logging.Logger:
    level = level.upper()
    levels = ["DEBUG", "INFO", "WARNING"]

    assert level.upper() in levels, f"level variable must be in {levels}"

    logger = logging.getLogger(name)
    
    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "WARNING":
        logger.setLevel(logging.WARNING)


    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    
    if level == "DEBUG":
        ch.setLevel(logging.DEBUG)
    elif level == "INFO":
        ch.setLevel(logging.INFO)
    elif level == "WARNING":
        ch.setLevel(logging.WARNING)
    
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    return logger
