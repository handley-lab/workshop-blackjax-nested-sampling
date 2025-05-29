#!/usr/bin/env python3
"""
Final test to understand BlackJAX nested sampling API
"""

import jax
import jax.numpy as jnp
import blackjax
import blackjax.ns.utils as ns_utils

# Generate some test data
np_random = jax.random.normal
n_data = 20
x_data = jnp.linspace(0, 10, n_data)
true_m, true_c, true_sigma = 2.5, 1.0, 0.5

rng_key = jax.random.key(42)
rng_key, subkey = jax.random.split(rng_key)
y_data = true_m * x_data + true_c + true_sigma * jax.random.normal(subkey, (n_data,))

# Define likelihood that works with dictionary input
@jax.jit
def dict_loglikelihood(params_dict):
    """Likelihood that expects dictionary input."""
    m = params_dict["m"]
    c = params_dict["c"]
    sigma = params_dict["sigma"]
    
    y_pred = m * x_data + c
    residuals = (y_data - y_pred) / sigma
    loglik = -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)
    return loglik

# Set up prior
prior_bounds = {
    "m": (0.0, 5.0),
    "c": (-2.0, 4.0),
    "sigma": (0.1, 2.0)
}

# Create particles and prior function
particles_dict, logprior_fn = ns_utils.uniform_prior(rng_key, 100, prior_bounds)

print(f"Particles type: {type(particles_dict)}")
print(f"Particles keys: {list(particles_dict.keys())}")

# Test likelihood with dict input
test_dict = {"m": true_m, "c": true_c, "sigma": true_sigma}
test_loglik = dict_loglikelihood(test_dict)
print(f"Test likelihood: {test_loglik:.3f}")

# Try to create sampler with correct dict format
try:
    sampler = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=dict_loglikelihood,
        num_delete=10,
        num_inner_steps=15
    )
    print("✓ Dict-based sampler created successfully")
    
    # Test initialization
    live_state = sampler.init(particles_dict)
    print("✓ Dict-based initialization successful")
    print(f"  Initial logZ: {live_state.logZ:.3f}")
    
    # Test one sampling step
    rng_key, subkey = jax.random.split(rng_key)
    jit_step = jax.jit(sampler.step)
    new_state, dead_info = jit_step(subkey, live_state)
    print("✓ Dict-based step successful")
    print(f"  New logZ: {new_state.logZ:.3f}")
    
    print("\n🎉 SUCCESS! Use dictionary format for particles and likelihood functions.")
    
except Exception as e:
    print(f"✗ Dict-based approach failed: {e}")
    import traceback
    traceback.print_exc()