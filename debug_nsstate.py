#!/usr/bin/env python3
"""
Debug NSState attributes
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.ns.utils as ns_utils

# Quick setup
jax.config.update("jax_enable_x64", True)
np.random.seed(42)

# Generate minimal data
n_data = 10
x_data = jnp.linspace(0, 5, n_data)
y_data = 2.0 * x_data + 1.0 + 0.3 * np.random.normal(size=n_data)

@jax.jit
def simple_loglik(params_dict):
    m, c, sigma = params_dict["m"], params_dict["c"], params_dict["sigma"]
    y_pred = m * x_data + c
    return -0.5 * jnp.sum(((y_data - y_pred) / sigma)**2)

prior_bounds = {"m": (0.0, 5.0), "c": (-2.0, 4.0), "sigma": (0.1, 2.0)}

rng_key = jax.random.key(42)
particles, logprior_fn = ns_utils.uniform_prior(rng_key, 50, prior_bounds)

sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=simple_loglik,
    num_delete=10,
    num_inner_steps=10
)

live_state = sampler.init(particles)

print("NSState attributes:")
print(dir(live_state))
print(f"\nlogZ: {live_state.logZ}")
print(f"logZ_live: {live_state.logZ_live}")

# Check if there are other information-related attributes
for attr in dir(live_state):
    if not attr.startswith('_'):
        try:
            value = getattr(live_state, attr)
            print(f"{attr}: {value}")
        except:
            print(f"{attr}: <could not access>")

# Run a few steps to see if H appears
dead_points = []
for i in range(5):
    rng_key, subkey = jax.random.split(rng_key)
    live_state, dead_info = sampler.step(subkey, live_state)
    dead_points.append(dead_info)
    
print(f"\nAfter 5 steps:")
print(f"logZ: {live_state.logZ}")

# Check if finalise creates something with H
dead = ns_utils.finalise(live_state, dead_points)
print(f"\nAfter finalise:")
print("Dead attributes:")
for attr in dir(dead):
    if not attr.startswith('_'):
        try:
            value = getattr(dead, attr)
            print(f"{attr}: {value}")
        except:
            print(f"{attr}: <could not access>")