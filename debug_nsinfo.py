#!/usr/bin/env python3
"""
Debug NSInfo attributes
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.ns.utils as ns_utils

# Quick setup
jax.config.update("jax_enable_x64", True)
np.random.seed(42)

n_data = 10
x_data = jnp.linspace(0, 5, n_data)
y_data = 2.0 * x_data + 1.0 + 0.3 * np.random.normal(size=n_data)

@jax.jit
def simple_loglik(params_dict):
    m, c, sigma = params_dict["m"], params_dict["c"], params_dict["sigma"]
    y_pred = m * x_data + c
    residuals = (y_data - y_pred) / sigma
    return -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)

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

# Run a few steps
dead_points = []
for i in range(10):
    rng_key, subkey = jax.random.split(rng_key)
    live_state, dead_info = sampler.step(subkey, live_state)
    dead_points.append(dead_info)

# Check what finalise returns
dead = ns_utils.finalise(live_state, dead_points)

print("NSInfo attributes:")
for attr in dir(dead):
    if not attr.startswith('_'):
        try:
            value = getattr(dead, attr)
            if hasattr(value, 'shape'):
                print(f"{attr}: shape {value.shape}, type {type(value)}")
            else:
                print(f"{attr}: {type(value)}")
        except:
            print(f"{attr}: <could not access>")

# Check specifically for likelihood-related attributes
print("\nLikelihood-related attributes:")
for attr in ['loglikelihood', 'logL', 'logL_birth', 'loglikelihood_birth']:
    if hasattr(dead, attr):
        value = getattr(dead, attr)
        print(f"✓ {attr}: shape {value.shape if hasattr(value, 'shape') else type(value)}")
    else:
        print(f"✗ {attr}: not found")

# Check if we can construct NestedSamples without logL_birth
try:
    from anesthetic import NestedSamples
    
    # Try with just required parameters
    samples_dict = dead.particles
    samples_array = jnp.stack([samples_dict[key] for key in ['m', 'c', 'sigma']], axis=1)
    
    # Try minimal NestedSamples
    ns_minimal = NestedSamples(
        data=samples_array,
        logL=dead.loglikelihood,
        columns=['m', 'c', 'σ']
    )
    print("\n✓ NestedSamples created successfully without logL_birth")
    
except Exception as e:
    print(f"\n✗ NestedSamples creation failed: {e}")

# Check what anesthetic NestedSamples expects
print("\nChecking NestedSamples signature...")
import inspect
sig = inspect.signature(NestedSamples.__init__)
print("NestedSamples.__init__ parameters:")
for name, param in sig.parameters.items():
    if name != 'self':
        print(f"  {name}: {param}")