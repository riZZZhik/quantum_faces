import logging

import numpy as np
from PIL import Image
from facenet_pytorch import fixed_image_standardization

# Logging
from torchvision import transforms


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


trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])


# Images
def norm_image(img):
    img *= 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    img = trans(img)

    return img
