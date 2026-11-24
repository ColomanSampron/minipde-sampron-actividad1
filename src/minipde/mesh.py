import numpy as np


def create_uniform_mesh(L: float, N: int):
    """
    Crea una malla unidimensional uniforme.

    Parametros
    ----------
    L : float
        Longitud del dominio.
    N : int
        Número de nodos.

    Devuelve
    -------
    x : numpy.ndarray
        Coordenadas de los nodos.
    x_interfaces : numpy.ndarray
        Coordenadas de las interfaces.
    dx : float
        Espaciado de la malla.
    """

    # Validar entradas
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L debe ser un número finito positivo.")

    if isinstance(N, (bool, np.bool_)) or not isinstance(
        N, (int, np.integer)
    ) or N < 2:
        raise ValueError("N debe ser un entero mayor o igual a 2.")

    # Calcular el espaciado de la malla
    dx = L / (N - 1)

    # Generar nodos
    x = np.linspace(0.0, L, N)

    # Generar coordenadas de las interfaces
    x_interfaces = (x[:-1] + x[1:]) / 2.0

    return x, x_interfaces, dx