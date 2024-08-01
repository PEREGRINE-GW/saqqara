import sgwbfish as sf
import jax
import numpy as np
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

pl = sf.PowerLaw()
freq_grid = jnp.geomspace(1e-5, 5e-1, 1000)
Nc = 95
Tc = 1e6
theta = jnp.hstack(
    [
        jnp.array([-11.0, 0.0]),
        jnp.hstack([jnp.array([-10.0, 1.0]) for i in range(Nc)]),
    ]
)
fish = sf.SGWBFish(pl, pl, freq_grid, Tc, Nc)


import time

start_time = time.time()

fisher_matrix = fish.get_fisher_matrix(theta)
covariance_matrix = fish.get_covariance_matrix(theta)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

theta = jnp.hstack(
    [
        jnp.array([-12.0, 0.0]),
        jnp.hstack([jnp.array([-11.0, 0.1]) for i in range(Nc)]),
    ]
)

start_time = time.time()

theta = jnp.vstack(
    [
        jnp.hstack(
            [
                jnp.array([np.random.uniform(-12.0, -10.0), 0.0]),
                jnp.hstack(
                    [
                        jnp.array([np.random.uniform(-12.0, -10.0), 0.1])
                        for j in range(Nc)
                    ]
                ),
            ]
        )
        for k in range(10)
    ]
)
print(theta.shape)
# fisher_matrix = fish.get_fisher_matrix(theta)
# covariance_matrix = fish.get_covariance_matrix(theta)

vmapped_version = jax.vmap(fish.get_fisher_matrix)(theta)
print(vmapped_version.shape)

theta = jnp.vstack(
    [
        jnp.hstack(
            [
                jnp.array([np.random.uniform(-12.0, -10.0), 0.0]),
                jnp.hstack(
                    [
                        jnp.array([np.random.uniform(-12.0, -10.0), 0.1])
                        for j in range(Nc)
                    ]
                ),
            ]
        )
        for k in range(1000)
    ]
)

start_time = time.time()
vmapped_version = jax.vmap(fish.get_fisher_matrix)(theta)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

print(covariance_matrix)
