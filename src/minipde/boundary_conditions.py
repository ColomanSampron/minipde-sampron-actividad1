import numpy as np


def apply_dirichlet(A, b, u_left):
    """
    Aplicar la condición de contorno de Dirichlet en x = 0.

    u(0) = u_left
    """

    A[0, :] = 0.0
    A[0, 0] = 1.0

    b[0] = u_left


def apply_robin(A, b, problem, dx, k_last):
    """
    Aplicar la condición de contorno de Robin en x = L.

    F(L) = h_robin * (u(L) - u_infinity)

    Utiliza un balance de media celda conservativo.
    """

    N = A.shape[0]

    h_robin = problem.h_robin
    u_infinity = problem.u_infinity

    # Fuente evaluada en el punto medio de la última media celda
    x_source = problem.L - dx / 4.0

    q_boundary = float(
        problem.evaluate_source(x_source)
    )

    # Limpiar la última fila
    A[-1, :] = 0.0

    # Coeficientes de la ecuación de Robin
    A[-1, N - 2] = -k_last

    A[-1, N - 1] = k_last + dx * h_robin

    # Lado derecho
    b[-1] = (
        dx * h_robin * u_infinity
        + (dx**2 / 2.0) * q_boundary
    )