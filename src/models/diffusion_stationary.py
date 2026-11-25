from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class StationaryDiffusionProblem:
    """
    Problema de difusión estacionaria unidimensional

    Ecuación:
        -(k(x) * u'(x))' = q(x)

    Condiciones de frontera:
        u(0) = u_left
        -k(L) * u'(L) = h_robin * (u(L) - u_infinity)
    """

    L: float
    conductivity: Callable
    source: Callable
    u_left: float
    h_robin: float
    u_infinity: float

    def __post_init__(self):
        """Validar los parámetros del problema."""

        if not np.isfinite(self.L) or self.L <= 0:
            raise ValueError("L debe ser positivo y finito.")

        if not callable(self.conductivity):
            raise TypeError("conductivity debe ser una función llamable.")

        if not callable(self.source):
            raise TypeError("source debe ser una función llamable.")

        if not np.isfinite(self.u_left):
            raise ValueError("u_left debe ser finito.")

        if not np.isfinite(self.h_robin) or self.h_robin < 0:
            raise ValueError("h_robin debe ser no negativo y finito.")

        if not np.isfinite(self.u_infinity):
            raise ValueError("u_infinity debe ser finito.")

    def evaluate_conductivity(self, x):
        """Evaluar conductividad"""

        values = np.asarray(self.conductivity(x), dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError("Conductivity debe ser finita.")

        if np.any(values <= 0):
            raise ValueError("Conductivity debe ser estrictamente positiva.")

        return values

    def evaluate_source(self, x):
        """Evaluar el término de fuente."""

        values = np.asarray(self.source(x), dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError("Source debe ser finito.")

        return values