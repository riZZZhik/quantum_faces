import logging


# Logging
def init_logger(log_file, log_level, log_name, date_format=None):
    if date_format is None:
        date_format = {
            "format": '%(asctime)s - %(levelname)s: %(message)s',
            "datefmt": '%d-%b-%y %H:%M:%S'
        }
    else:
        assert "format" in date_format and "datefmt" in date_format, 'date_format should have "format" and "datefmt"'

    logging.basicConfig(level=log_level, **date_format)
    logger = logging.getLogger(log_name)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(*date_format.values())
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
