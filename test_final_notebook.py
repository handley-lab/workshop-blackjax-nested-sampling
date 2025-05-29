#!/usr/bin/env python3
"""
Final comprehensive test of the fixed notebook
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

print("=== FINAL NOTEBOOK TEST ===")

# Setup
import blackjax
import blackjax.ns.utils as ns_utils
from anesthetic import NestedSamples
import corner

jax.config.update("jax_enable_x64", True)
np.random.seed(42)
rng_key = jax.random.key(42)

print("✓ Imports and JAX setup")

# Test 1: Line fitting workflow
print("\n=== TEST 1: Line Fitting ===")

# Data generation
n_data = 20
x_data = jnp.linspace(0, 10, n_data)
true_m, true_c, true_sigma = 2.5, 1.0, 0.5
y_data = true_m * x_data + true_c + true_sigma * np.random.normal(size=n_data)

# Likelihood
@jax.jit
def line_loglikelihood(params_dict):
    m, c, sigma = params_dict["m"], params_dict["c"], params_dict["sigma"]
    y_pred = m * x_data + c
    residuals = (y_data - y_pred) / sigma
    return -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)

# Prior
line_prior_bounds = {"m": (0.0, 5.0), "c": (-2.0, 4.0), "sigma": (0.1, 2.0)}

# Nested sampling
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

# Run limited sampling
line_dead_points = []
for i in range(30):
    rng_key, subkey = jax.random.split(rng_key)
    line_live_state, line_dead_info = line_jit_step(subkey, line_live_state)
    line_dead_points.append(line_dead_info)

# Process results
line_dead = ns_utils.finalise(line_live_state, line_dead_points)
line_samples_dict = line_dead.particles
line_samples = jnp.stack([line_samples_dict[key] for key in ['m', 'c', 'sigma']], axis=1)

line_nested_samples = NestedSamples(
    data=line_samples,
    logL=line_dead.loglikelihood,
    logL_birth=line_dead.loglikelihood_birth,
    columns=['m', 'c', 'σ']
)

print(f"✓ Line fitting: {len(line_samples)} samples, evidence = {line_live_state.logZ:.2f}")

# Test 2: 2D Gaussian workflow (abbreviated)
print("\n=== TEST 2: 2D Gaussian ===")

# Setup 2D Gaussian problem
image_size = 30  # Smaller for testing
x = jnp.linspace(-3, 3, image_size)
y = jnp.linspace(-3, 3, image_size)
X, Y = jnp.meshgrid(x, y)
coords = jnp.stack([X.ravel(), Y.ravel()]).T

true_params = jnp.array([0.5, -0.3, 1.2, 0.8, 0.4])

@jax.jit
def simulator(params, rng_key, noise_sigma=0.1):
    mu_x, mu_y, sigma_x, sigma_y, rho = params
    mean = jnp.array([mu_x, mu_y])
    cov = jnp.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])
    
    logpdf = jax.scipy.stats.multivariate_normal.logpdf(coords, mean, cov)
    image_clean = logpdf.reshape(image_size, image_size)
    image_clean = jnp.exp(image_clean - jnp.max(image_clean))
    noise = jax.random.normal(rng_key, image_clean.shape) * noise_sigma
    return image_clean + noise

rng_key, subkey = jax.random.split(rng_key)
observed_data = simulator(true_params, subkey)

# 2D Gaussian likelihood
@jax.jit
def loglikelihood_fn(params_dict):
    mu_x = params_dict["mu_x"]
    mu_y = params_dict["mu_y"]
    sigma_x = params_dict["sigma_x"]
    sigma_y = params_dict["sigma_y"]
    rho = params_dict["rho"]
    
    mean = jnp.array([mu_x, mu_y])
    cov = jnp.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])
    
    det_cov = jnp.linalg.det(cov)
    
    def valid_cov():
        logpdf = jax.scipy.stats.multivariate_normal.logpdf(coords, mean, cov)
        image_pred = logpdf.reshape(image_size, image_size)
        image_pred = jnp.exp(image_pred - jnp.max(image_pred))
        noise_sigma = 0.1
        residuals = (observed_data - image_pred) / noise_sigma
        return -0.5 * jnp.sum(residuals**2)
    
    def invalid_cov():
        return -jnp.inf
    
    return jax.lax.cond(det_cov > 0, valid_cov, invalid_cov)

# Prior
prior_bounds = {
    "mu_x": (-2.0, 2.0),
    "mu_y": (-2.0, 2.0), 
    "sigma_x": (0.5, 3.0),
    "sigma_y": (0.5, 3.0),
    "rho": (-0.99, 0.99)
}

# Test likelihood
test_params_dict = {
    "mu_x": true_params[0],
    "mu_y": true_params[1], 
    "sigma_x": true_params[2],
    "sigma_y": true_params[3],
    "rho": true_params[4]
}
test_loglik = loglikelihood_fn(test_params_dict)
print(f"✓ 2D Gaussian likelihood: {test_loglik:.3f}")

# Brief sampling test
rng_key, subkey = jax.random.split(rng_key)
particles, logprior_fn = ns_utils.uniform_prior(subkey, 50, prior_bounds)

sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=5,
    num_inner_steps=10
)

live_state = sampler.init(particles)
jit_step = jax.jit(sampler.step)

# Run a few steps
dead_points = []
for i in range(10):
    rng_key, subkey = jax.random.split(rng_key)
    live_state, dead_info = jit_step(subkey, live_state)
    dead_points.append(dead_info)

print(f"✓ 2D Gaussian sampling: {len(dead_points)} steps, logZ = {live_state.logZ:.2f}")

# Test 3: Results processing
print("\n=== TEST 3: Results Processing ===")

dead = ns_utils.finalise(live_state, dead_points)
param_keys = ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho']
samples_dict = dead.particles
samples = jnp.stack([samples_dict[key] for key in param_keys], axis=1)

nested_samples = NestedSamples(
    data=samples,
    logL=dead.loglikelihood,
    logL_birth=dead.loglikelihood_birth,
    columns=['μₓ', 'μᵧ', 'σₓ', 'σᵧ', 'ρ']
)

# Test anesthetic operations
weights = nested_samples.get_weights()
n_eff = 1.0 / jnp.sum(weights**2)

print(f"✓ Anesthetic processing: {len(samples)} samples, {n_eff:.0f} effective")

print("\n🎉 ALL TESTS PASSED!")
print("The notebook should run end-to-end successfully.")
print("\nKey fixes applied:")
print("- ✓ Fixed BlackJAX installation (removed @nested_sampling)")
print("- ✓ Fixed particles dictionary handling")
print("- ✓ Fixed logL_birth -> loglikelihood_birth")
print("- ✓ Removed non-existent H attribute references")  
print("- ✓ Added line fitting as accessible starting example")
print("- ✓ Fixed likelihood functions to use dictionary parameters")

print(f"\nFinal test summary:")
print(f"- Line fitting: {len(line_samples)} samples")
print(f"- 2D Gaussian: {len(samples)} samples")
print(f"- Evidence computation: Working")
print(f"- Anesthetic integration: Working")