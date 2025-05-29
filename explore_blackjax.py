#!/usr/bin/env python3
"""
Explore BlackJAX nested sampling API properly
"""

import jax
import jax.numpy as jnp
import blackjax
import blackjax.ns.utils as ns_utils

# Check what the ns.utils.uniform_prior actually returns
prior_bounds = {
    "m": (0.0, 5.0),
    "c": (-2.0, 4.0),
    "sigma": (0.1, 2.0)
}

rng_key = jax.random.key(42)

# Let's see what the function signature and docstring say
print("uniform_prior function:")
print(ns_utils.uniform_prior.__doc__)

# Test with manual array construction instead
def create_particles_array(rng_key, num_live, bounds_dict):
    """Create particles as array."""
    param_names = list(bounds_dict.keys())
    num_dims = len(param_names)
    
    # Create random samples within bounds
    rng_key, *subkeys = jax.random.split(rng_key, num_dims + 1)
    
    particles = []
    for i, name in enumerate(param_names):
        low, high = bounds_dict[name]
        samples = jax.random.uniform(subkeys[i], (num_live,), minval=low, maxval=high)
        particles.append(samples)
    
    return jnp.stack(particles, axis=1)

# Test manual array creation
particles_array = create_particles_array(rng_key, 100, prior_bounds)
print(f"Manual particles array shape: {particles_array.shape}")

# Define simple likelihood that works with arrays
@jax.jit
def array_loglikelihood(params):
    """Likelihood that expects array input."""
    m, c, sigma = params
    # Simple quadratic for testing
    return -0.5 * (m**2 + c**2 + sigma**2)

# Define manual logprior for arrays
def array_logprior(params):
    """Uniform prior for array input."""
    m, c, sigma = params
    # Check bounds
    if (0.0 <= m <= 5.0 and -2.0 <= c <= 4.0 and 0.1 <= sigma <= 2.0):
        return 0.0  # Uniform prior
    else:
        return -jnp.inf

try:
    sampler = blackjax.nss(
        logprior_fn=array_logprior,
        loglikelihood_fn=array_loglikelihood,
        num_delete=10,
        num_inner_steps=15
    )
    print("✓ Array-based sampler created")
    
    live_state = sampler.init(particles_array)
    print("✓ Array-based initialization successful")
    print(f"  Initial logZ: {live_state.logZ}")
    
    # Test one step
    rng_key, subkey = jax.random.split(rng_key)
    new_state, dead_info = sampler.step(subkey, live_state)
    print("✓ Array-based step successful")
    print(f"  New logZ: {new_state.logZ}")
    
except Exception as e:
    print(f"✗ Array-based approach failed: {e}")
    import traceback
    traceback.print_exc()

# Let's also try the actual example from the BlackJAX repo
print("\nTrying exact BlackJAX API...")
try:
    # Maybe the issue is with the logprior function format
    particles_dict, logprior_fn = ns_utils.uniform_prior(rng_key, 100, prior_bounds)
    print(f"Dict particles: {type(particles_dict)}")
    
    # Let's see what logprior_fn expects
    test_input = jnp.array([2.0, 1.0, 0.5])
    logprior_result = logprior_fn(test_input)
    print(f"Logprior test: {logprior_result}")
    
    # Try with dict input to logprior
    test_dict = {"m": 2.0, "c": 1.0, "sigma": 0.5}
    try:
        logprior_result2 = logprior_fn(test_dict)
        print(f"Logprior dict test: {logprior_result2}")
    except Exception as e:
        print(f"Logprior dict test failed: {e}")
    
except Exception as e:
    print(f"Dict approach exploration failed: {e}")
    import traceback
    traceback.print_exc()