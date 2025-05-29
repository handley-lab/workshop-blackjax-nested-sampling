#!/usr/bin/env python3
"""
Debug particles initialization in BlackJAX
"""

import jax
import jax.numpy as jnp
import blackjax.ns.utils as ns_utils

# Test particles initialization
prior_bounds = {
    "m": (0.0, 5.0),
    "c": (-2.0, 4.0),
    "sigma": (0.1, 2.0)
}

rng_key = jax.random.key(42)
particles, logprior_fn = ns_utils.uniform_prior(rng_key, 100, prior_bounds)

print(f"Particles type: {type(particles)}")
print(f"Particles: {particles}")

if isinstance(particles, dict):
    print(f"Keys: {list(particles.keys())}")
    for key, values in particles.items():
        print(f"  {key}: shape {values.shape}, type {type(values)}")
        print(f"    first few: {values[:5]}")
    
    # Try to convert to array format
    param_names = list(particles.keys())
    particles_array = jnp.stack([particles[name] for name in param_names], axis=1)
    print(f"\nConverted to array: {particles_array.shape}")
    print(f"First few rows:\n{particles_array[:5]}")

# Check what the sampler expects
import blackjax

@jax.jit
def dummy_loglik(params):
    return jnp.sum(params**2)

try:
    sampler = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=dummy_loglik,
        num_delete=10,
        num_inner_steps=15
    )
    print(f"\nSampler created successfully")
    
    # Try different formats
    try:
        live_state1 = sampler.init(particles)
        print("✓ Dict particles work")
    except Exception as e:
        print(f"✗ Dict particles failed: {e}")
        
    try:
        live_state2 = sampler.init(particles_array)
        print("✓ Array particles work")
    except Exception as e:
        print(f"✗ Array particles failed: {e}")
        
except Exception as e:
    print(f"Sampler creation failed: {e}")