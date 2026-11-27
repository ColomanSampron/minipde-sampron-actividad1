import numpy as np
from minipde.boundary_conditions import apply_dirichlet, apply_robin

def assemble_interior(problem, x, interfaces, dx):
    """
    Construye las ecuaciones interiores del 
    problema de difusión estacionaria
    utilizando diferencias finitas conservativas.


    Parametros
    ----------
    problema : StationaryDiffusionProblem
        Definición del problema de difusión estacionaria.

    x : numpy.ndarray
        Coordenadas de los nodos de la malla.

    interfaces : numpy.ndarray
        Coordenadas de las interfaces.

    dx : float
        Espaciado uniforme de la malla.

    Devuelve
    -------
    A : numpy.ndarray
        Matriz de coeficientes de las ecuaciones interiores de forma (N-2, N).

    b : numpy.ndarray
        Vector del lado derecho de forma (N-2,).
    """

    x = np.asarray(x, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)

    N = len(x)

    if N < 3:
        raise ValueError("Se necesitan al menos 3 nodos para formar ecuaciones interiores.")

    if x.ndim != 1 or interfaces.ndim != 1:
        raise ValueError("Las coordenadas de la malla deben ser unidimensionales.")

    if len(interfaces) != N - 1:
        raise ValueError("Numero incorrecto de interfaces.")

    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx debe ser positivo y finito.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Los nodos deben ser finitos.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Los nodos deben estar estrictamente en aumento.")

    if not np.allclose(np.diff(x), dx):
        raise ValueError("La malla debe ser uniforme.")

    if not np.allclose(
        interfaces, (x[:-1] + x[1:]) / 2.0
    ):
        raise ValueError("Posiciones incorrectas de las interfaces.")

    # Condictividad en las interfaces
    k_interfaces = np.asarray(
        problem.evaluate_conductivity(interfaces),
        dtype=float
    )

    if k_interfaces.shape != (N - 1,):
        raise ValueError("Dimensiones incorrectas de la conductividad en las interfaces.")

    # Fuente en los nodos interiores
    q_interior = np.asarray(
        problem.evaluate_source(x[1:-1]),
        dtype=float
    )

    if q_interior.shape != (N - 2,):
        raise ValueError("Dimensiones incorrectas de la fuente en los nodos interiores.")

    # Inicializar la matriz A y el vector b
    A = np.zeros((N - 2, N))
    b = np.zeros(N - 2)

    # Montar las ecuaciones interiores usando diferencias finitas conservativas
    for i in range(1, N - 1):

        k_left = k_interfaces[i - 1]
        k_right = k_interfaces[i]

        row = i - 1

        A[row, i - 1] = -k_left
        A[row, i] = k_left + k_right
        A[row, i + 1] = -k_right

        b[row] = dx**2 * q_interior[row]

    return A, b

def assemble_system(problem, x, interfaces, dx):
    """
    Monta el sistema completo de difusión estacionaria, 
    incluyendo las ecuaciones interiores y las condiciones de contorno.

    Devuelve
    -------
    A : numpy.ndarray
        Matriz de coeficientes del sistema completo.

    b : numpy.ndarray
        Vector del lado derecho del sistema completo.
    """

    N = len(x)

    if N < 3:
        raise ValueError("Se requieren al menos 3 nodos.")

    # Ecuaciones interiores
    A_interior, b_interior = assemble_interior(
        problem, x, interfaces, dx
    )

    # Inicializar la matriz completa y el vector del lado derecho
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Insertar las ecuaciones interiores en la matriz completa y el vector del lado derecho
    A[1:-1, :] = A_interior
    b[1:-1] = b_interior

    # Aplicar la condición de contorno de Dirichlet
    apply_dirichlet(
        A,
        b,
        problem.u_left
    )

    # Conductividad en la última interfaz
    k_last = float(
        problem.evaluate_conductivity(interfaces[-1])
    )

    # Aplicar la condición robin
    apply_robin(
        A,
        b,
        problem,
        dx,
        k_last
    )

    return A, b