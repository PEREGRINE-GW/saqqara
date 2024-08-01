import sgwbfish as sf
import jax
import numpy as np
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

signal = sf.PowerLaw()
lisa = sf.LISA()
freq_grid = jnp.geomspace(5e-5, 3e-1, 1000)

Tc, Nc = 1e6, 3
fish = sf.SGWBFish(Sn=lisa, Sh=signal, freq_grid=freq_grid, Tc=Tc, Nc=Nc)
small_fish = sf.SGWBFish(
    Sn=lisa, Sh=signal, freq_grid=freq_grid, Tc=Tc * Nc, Nc=1
)


def get_full_theta_fixed():
    return jnp.hstack(
        [
            jnp.array([-12.0, 0.0]),
            jnp.hstack([jnp.array([3.0, 15.0]) for i in range(Nc)]),
        ]
    )


def get_small_theta_fixed():
    return jnp.hstack(
        [
            jnp.array([-12.0, 0.0]),
            jnp.hstack([jnp.array([3.0, 15.0]) for i in range(1)]),
        ]
    )


theta_full = get_full_theta_fixed()
theta_small = get_small_theta_fixed()

import time

start_time = time.time()
covariance_matrix_full = fish.get_covariance_matrix(theta_full)
end_time = time.time()
execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")

start_time = time.time()
fisher_matrix_full = fish.get_fisher_matrix(theta_full)
eigenvalues_full, eigenvectors_full = jnp.linalg.eigh(fisher_matrix_full)
print(eigenvalues_full)
print(eigenvectors_full.T)

vec = jnp.sqrt(jnp.diag(fisher_matrix_full))
rescale = jnp.einsum("i,j->ij", vec, vec)
inv_mat = jnp.linalg.inv(fisher_matrix_full / rescale) / rescale
covariance_matrix_full = fish.get_covariance_matrix(theta_full)
print((inv_mat - covariance_matrix_full) / covariance_matrix_full)
# print(jnp.matmul(fisher_matrix_full, covariance_matrix_full))
end_time = time.time()
execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")

start_time = time.time()
fisher_matrix_small = small_fish.get_fisher_matrix(theta_small)
eigenvalues_small, eigenvectors_small = jnp.linalg.eigh(fisher_matrix_small)
print(eigenvalues_small)
print(eigenvectors_small.T)
covariance_matrix_small = small_fish.get_covariance_matrix(theta_small)
# print(jnp.matmul(fisher_matrix_small, covariance_matrix_small))
end_time = time.time()
execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")

start_time = time.time()
theta_stack = jnp.vstack([get_small_theta_fixed() for _ in range(10000)])
covariance_matrix_small = small_fish.get_covariance_matrix_vmap(theta_stack)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

start_time = time.time()
theta_stack = jnp.vstack([get_small_theta_fixed() for _ in range(10000)])
covariance_matrix_small = small_fish.get_covariance_matrix_vmap(theta_stack)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

start_time = time.time()
theta_stack = jnp.vstack([get_small_theta_fixed() for _ in range(10000)])
covariance_matrix_small = small_fish.get_covariance_matrix_vmap(theta_stack)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")


# print(fisher_matrix_full[:4, :4])
# print(covariance_matrix_full[:4, :4])

# print(fisher_matrix_small)
# print(covariance_matrix_small)
