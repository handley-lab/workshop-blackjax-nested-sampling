#!/usr/bin/env python3
"""
Test script to validate BlackJAX nested sampling workshop components
"""

# Test basic imports
print("Testing imports...")
try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from functools import partial
    import time
    print("✓ Basic imports successful")
except ImportError as e:
    print(f"✗ Basic import failed: {e}")
    exit(1)

# Test BlackJAX imports
try:
    import blackjax
    import blackjax.ns.utils as ns_utils
    print("✓ BlackJAX imports successful")
except ImportError as e:
    print(f"✗ BlackJAX import failed: {e}")
    print("  Try: pip install git+https://github.com/handley-lab/blackjax")
    exit(1)

# Test anesthetic import
try:
    from anesthetic import NestedSamples
    print("✓ Anesthetic import successful")
except ImportError as e:
    print(f"✗ Anesthetic import failed: {e}")
    print("  Try: pip install anesthetic")
    exit(1)

# JAX configuration
jax.config.update("jax_enable_x64", True)
print(f"✓ JAX configured with devices: {jax.devices()}")

# Test line fitting example
print("\nTesting line fitting example...")

# Generate data
np.random.seed(42)
n_data = 20
x_data = jnp.linspace(0, 10, n_data)
true_m, true_c, true_sigma = 2.5, 1.0, 0.5
true_y_clean = true_m * x_data + true_c
noise = true_sigma * np.random.normal(size=n_data)
y_data = true_y_clean + noise

# Define likelihood
@jax.jit
def line_loglikelihood(params):
    m, c, sigma = params
    y_pred = m * x_data + c
    residuals = (y_data - y_pred) / sigma
    loglik = -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)
    return loglik

# Test likelihood evaluation
test_params = jnp.array([true_m, true_c, true_sigma])
test_loglik = line_loglikelihood(test_params)
print(f"✓ Line likelihood at true params: {test_loglik:.3f}")

# Test prior setup
line_prior_bounds = {
    "m": (0.0, 5.0),
    "c": (-2.0, 4.0),
    "sigma": (0.1, 2.0)
}

rng_key = jax.random.key(42)
rng_key, subkey = jax.random.split(rng_key)

try:
    particles, logprior_fn = ns_utils.uniform_prior(subkey, 100, line_prior_bounds)
    print(f"✓ Prior setup successful: {type(particles)}")
    print(f"  Particles type: {type(particles)}")
    
    # Check if particles is a dict or array
    if isinstance(particles, dict):
        print(f"  Particles keys: {list(particles.keys())}")
        num_live = len(particles[list(particles.keys())[0]])
        num_dims = len(particles)
    else:
        print(f"  Particles shape: {particles.shape}")
        num_live, num_dims = particles.shape
        
    print(f"  Live points: {num_live}, Dimensions: {num_dims}")
    
except Exception as e:
    print(f"✗ Prior setup failed: {e}")
    exit(1)

# Test nested sampler creation
try:
    sampler = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=line_loglikelihood,
        num_delete=10,
        num_inner_steps=15
    )
    print("✓ Nested sampler creation successful")
except Exception as e:
    print(f"✗ Nested sampler creation failed: {e}")
    exit(1)

# Test initialization
try:
    live_state = sampler.init(particles)
    print(f"✓ Sampler initialization successful")
    print(f"  Initial logZ: {live_state.logZ:.3f}")
except Exception as e:
    print(f"✗ Sampler initialization failed: {e}")
    exit(1)

# Test one sampling step
try:
    rng_key, subkey = jax.random.split(rng_key)
    jit_step = jax.jit(sampler.step)
    new_live_state, dead_info = jit_step(subkey, live_state)
    print(f"✓ Sampling step successful")
    print(f"  New logZ: {new_live_state.logZ:.3f}")
except Exception as e:
    print(f"✗ Sampling step failed: {e}")
    exit(1)

print("\n🎉 All tests passed! The notebook should work correctly.")
print(f"\nNote: particles is a {type(particles)}")
if isinstance(particles, dict):
    print("Use len(particles[list(particles.keys())[0]]) for num_live")
    print("Use len(particles) for num_dims")
else:
    print("Use particles.shape[0] for num_live")
    print("Use particles.shape[1] for num_dims")