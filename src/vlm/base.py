from abc import ABC, abstractmethod

from torch_snippets import read, resize, Info, in_debug_mode, show
from torch_snippets.adapters import np_2_b64

class VLM(ABC):
    def __init__(self, token):
        self.token = token

    @abstractmethod
    def __call__(self, image, prompt, **kwargs):
        pass

    def path_2_b64(self, path, image_size=None):
        image = read(path)
        if image_size:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = resize(image, ('at-most', image_size))
        if in_debug_mode():
            Info(f'{image.shape=}')
            show(image)
        return np_2_b64(image)
