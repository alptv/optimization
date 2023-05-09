from .linesearch import DirectionProvider, Statistics
from .grad import grad


class GradientDirection(DirectionProvider):

    def __init__(self, eps=0.001) -> None:
        super().__init__()
        self.eps = eps

    def direction(self, f, x) -> tuple:
        return -grad(f, x), Statistics({Statistics.GRADIENT: 1})