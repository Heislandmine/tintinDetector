import numpy as np
from PIL import Image


def read_image(file_path: str) -> np.ndarray:
    image = Image.open(file_path).convert("RGB")

    return image
