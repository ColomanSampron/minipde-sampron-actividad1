import numpy as np

from minipde.mesh import create_uniform_mesh
from models.diffusion_stationary import StationaryDiffusionProblem


def conductivity(x):
    return 1.0 + x


def source(x):
    return (
        -10.0
        - 5.0 * np.pi * np.cos(np.pi * x)
        + 5.0 * np.pi**2 * (1.0 + x) * np.sin(np.pi * x)
    )


def exact_solution(x):
    return 20.0 + 10.0 * x + 5.0 * np.sin(np.pi * x)


def main():

    problem = StationaryDiffusionProblem(
        L=1.0,
        conductivity=conductivity,
        source=source,
        u_left=20.0,
        h_robin=2.0,
        u_infinity=40.0 - 5.0 * np.pi,
    )

    x, interfaces, dx = create_uniform_mesh(problem.L, 5)

    k_values = problem.evaluate_conductivity(x)
    q_values = problem.evaluate_source(x)
    u_exact = exact_solution(x)

    print("Nodes:", x)
    print("Conductivity:", k_values)
    print("Source:", q_values)
    print("Exact solution:", u_exact)


if __name__ == "__main__":
    main()