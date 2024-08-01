import sgwbfish as sf
import jax
import time
import numpy as np
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt


signal = sf.PowerLaw()
lisa = sf.LISA()
freq_grid = jnp.geomspace(5e-5, 3e-1, 1000)

Tc, Nc = 1e6, 3
fish = sf.SGWBFish(Sn=lisa, Sh=signal, freq_grid=freq_grid, Tc=Tc, Nc=Nc)


def get_full_theta_fixed():
    return jnp.hstack(
        [
            jnp.array([-12.0, 0.0]),
            jnp.hstack([jnp.array([3.0, 15.0]) for i in range(Nc)]),
        ]
    )


theta_full = get_full_theta_fixed()

start_time = time.time()
fisher_matrix_full = fish.get_fisher_matrix_unbatched(theta_full)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

start_time = time.time()
fisher_matrix_full = fish.get_fisher_matrix_unbatched(theta_full)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

start_time = time.time()
fisher_matrix_full_batched = fish.get_fisher_matrix(theta_full)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

start_time = time.time()
fisher_matrix_full_batched = fish.get_fisher_matrix(theta_full)
covariance_matrix_full_batched = fish.get_covariance_matrix(theta_full)
print(covariance_matrix_full_batched)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
