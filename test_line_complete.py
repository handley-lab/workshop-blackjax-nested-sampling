#!/usr/bin/env python3
"""
Test complete line fitting example
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.ns.utils as ns_utils
from anesthetic import NestedSamples
import time

# JAX configuration
jax.config.update("jax_enable_x64", True)

# Generate line data
np.random.seed(42)
n_data = 20
x_data = jnp.linspace(0, 10, n_data)

true_m = 2.5    # slope
true_c = 1.0    # intercept  
true_sigma = 0.5 # noise level

true_y_clean = true_m * x_data + true_c
noise = true_sigma * np.random.normal(size=n_data)
y_data = true_y_clean + noise

print(f"Generated data: {n_data} points")
print(f"True parameters: m={true_m}, c={true_c}, σ={true_sigma}")

# Define likelihood
@jax.jit
def line_loglikelihood(params_dict):
    m = params_dict["m"]
    c = params_dict["c"] 
    sigma = params_dict["sigma"]
    
    y_pred = m * x_data + c
    residuals = (y_data - y_pred) / sigma
    loglik = -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)
    
    return loglik

# Prior bounds
line_prior_bounds = {
    "m": (0.0, 5.0),
    "c": (-2.0, 4.0),
    "sigma": (0.1, 2.0)
}

# Test likelihood
test_params = {"m": true_m, "c": true_c, "sigma": true_sigma}
test_loglik = line_loglikelihood(test_params)
print(f"Test likelihood: {test_loglik:.3f}")

# Setup nested sampling
print("\nSetting up nested sampling...")
rng_key = jax.random.key(42)
rng_key, subkey = jax.random.split(rng_key)

line_particles, line_logprior_fn = ns_utils.uniform_prior(
    subkey, 200, line_prior_bounds  # Smaller for quick test
)

line_sampler = blackjax.nss(
    logprior_fn=line_logprior_fn,
    loglikelihood_fn=line_loglikelihood,
    num_delete=20,
    num_inner_steps=15
)

line_live_state = line_sampler.init(line_particles)
line_jit_step = jax.jit(line_sampler.step)

print(f"Initialized. Starting sampling...")

# Run nested sampling
line_dead_points = []
iteration = 0
start_time = time.time()

max_iterations = 100  # Safety limit for testing
while (line_live_state.logZ_live - line_live_state.logZ) > -3.0 and iteration < max_iterations:
    rng_key, subkey = jax.random.split(rng_key)
    
    line_live_state, line_dead_info = line_jit_step(subkey, line_live_state)
    line_dead_points.append(line_dead_info)
    
    iteration += 1
    
    if iteration % 10 == 0:
        remaining = line_live_state.logZ_live - line_live_state.logZ
        print(f"  Iteration {iteration:3d}: logZ = {line_live_state.logZ:.3f}, remaining = {remaining:.3f}")

sampling_time = time.time() - start_time

print(f"\nSampling completed!")
print(f"Total iterations: {iteration}")
print(f"Sampling time: {sampling_time:.2f} seconds")
print(f"Evidence: logZ = {line_live_state.logZ:.3f} ± {jnp.sqrt(line_live_state.H):.3f}")

# Process results
print("\nProcessing results...")
line_dead = ns_utils.finalise(line_live_state, line_dead_points)

line_param_names = ['m', 'c', 'σ']
param_keys = ['m', 'c', 'sigma']

line_samples_dict = line_dead.particles
line_samples = jnp.stack([line_samples_dict[key] for key in param_keys], axis=1)

print(f"Sample shape: {line_samples.shape}")

# Create NestedSamples
line_nested_samples = NestedSamples(
    data=line_samples,
    logL=line_dead.loglikelihood,
    logL_birth=line_dead.logL_birth,
    columns=line_param_names
)

print("Posterior summary:")
line_true_params = jnp.array([true_m, true_c, true_sigma])
for i, name in enumerate(line_param_names):
    mean = line_nested_samples[name].mean()
    std = line_nested_samples[name].std()
    true_val = line_true_params[i]
    print(f"  {name}: {mean:.3f} ± {std:.3f} (true: {true_val:.3f})")

print("\n🎉 Line fitting test completed successfully!")