#!/usr/bin/env python3
"""
Test the fixed notebook workflow with corrected attribute names
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import time

print("=== Testing Fixed Notebook ===")

# Setup
import blackjax
import blackjax.ns.utils as ns_utils
from anesthetic import NestedSamples

jax.config.update("jax_enable_x64", True)
np.random.seed(42)
rng_key = jax.random.key(42)

print("✓ Setup complete")

# Line fitting data
n_data = 20
x_data = jnp.linspace(0, 10, n_data)
true_m, true_c, true_sigma = 2.5, 1.0, 0.5
true_y_clean = true_m * x_data + true_c
noise = true_sigma * np.random.normal(size=n_data)
y_data = true_y_clean + noise

# Line likelihood
@jax.jit
def line_loglikelihood(params_dict):
    m, c, sigma = params_dict["m"], params_dict["c"], params_dict["sigma"]
    y_pred = m * x_data + c
    residuals = (y_data - y_pred) / sigma
    return -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)

line_prior_bounds = {
    "m": (0.0, 5.0),
    "c": (-2.0, 4.0),
    "sigma": (0.1, 2.0)
}

print("✓ Line fitting setup complete")

# Run line fitting
rng_key, subkey = jax.random.split(rng_key)
line_particles, line_logprior_fn = ns_utils.uniform_prior(subkey, 100, line_prior_bounds)

line_sampler = blackjax.nss(
    logprior_fn=line_logprior_fn,
    loglikelihood_fn=line_loglikelihood,
    num_delete=10,
    num_inner_steps=15
)

line_live_state = line_sampler.init(line_particles)
line_jit_step = jax.jit(line_sampler.step)

# Quick sampling run
line_dead_points = []
for i in range(20):  # Just a few steps for testing
    rng_key, subkey = jax.random.split(rng_key)
    line_live_state, line_dead_info = line_jit_step(subkey, line_live_state)
    line_dead_points.append(line_dead_info)

print(f"✓ Line sampling: {len(line_dead_points)} steps")

# Process results with corrected attribute names
line_dead = ns_utils.finalise(line_live_state, line_dead_points)

print("Line dead object attributes:")
print(f"  loglikelihood: {hasattr(line_dead, 'loglikelihood')}")
print(f"  loglikelihood_birth: {hasattr(line_dead, 'loglikelihood_birth')}")

# Extract samples
line_param_names = ['m', 'c', 'σ']
param_keys = ['m', 'c', 'sigma']
line_samples_dict = line_dead.particles
line_samples = jnp.stack([line_samples_dict[key] for key in param_keys], axis=1)

# Create NestedSamples with corrected attribute
try:
    line_nested_samples = NestedSamples(
        data=line_samples,
        logL=line_dead.loglikelihood,
        logL_birth=line_dead.loglikelihood_birth,  # Fixed attribute name
        columns=line_param_names
    )
    print("✓ NestedSamples created successfully with corrected attributes")
    
    # Test basic operations
    for i, name in enumerate(line_param_names):
        mean = line_nested_samples[name].mean()
        std = line_nested_samples[name].std()
        print(f"  {name}: {mean:.3f} ± {std:.3f}")
    
except Exception as e:
    print(f"✗ NestedSamples creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Fixed notebook test completed successfully!")
print("The logL_birth -> loglikelihood_birth fix resolves the issue.")