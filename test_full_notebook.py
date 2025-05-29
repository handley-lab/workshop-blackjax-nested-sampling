#!/usr/bin/env python3
"""
Test the complete notebook workflow end-to-end
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from functools import partial
import time

print("=== Cell 1: Setup and Installation ===")
try:
    # Imports
    import blackjax
    import blackjax.ns.utils as ns_utils
    from anesthetic import NestedSamples
    import corner
    
    # JAX configuration
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')
    
    # Set random seed
    np.random.seed(42)
    rng_key = jax.random.key(42)
    
    print("✓ Imports and setup successful")
except Exception as e:
    print(f"✗ Setup failed: {e}")
    exit(1)

print("\n=== Cell 2: Line Fitting Data Generation ===")
try:
    # Generate synthetic line data
    np.random.seed(42)
    n_data = 20
    x_data = jnp.linspace(0, 10, n_data)
    
    # True parameters for line: y = mx + c + noise
    true_m = 2.5    # slope
    true_c = 1.0    # intercept  
    true_sigma = 0.5 # noise level
    
    # Generate noisy observations
    true_y_clean = true_m * x_data + true_c
    noise = true_sigma * np.random.normal(size=n_data)
    y_data = true_y_clean + noise
    
    print(f"✓ Line data generated: {n_data} points")
    print(f"  True parameters: m={true_m}, c={true_c}, σ={true_sigma}")
except Exception as e:
    print(f"✗ Line data generation failed: {e}")
    exit(1)

print("\n=== Cell 3: Line Fitting Likelihood and Prior ===")
try:
    # Define likelihood and prior for line fitting
    @jax.jit
    def line_loglikelihood(params_dict):
        """Log-likelihood for linear regression."""
        m = params_dict["m"]
        c = params_dict["c"] 
        sigma = params_dict["sigma"]
        
        # Predicted y values
        y_pred = m * x_data + c
        
        # Gaussian likelihood
        residuals = (y_data - y_pred) / sigma
        loglik = -0.5 * jnp.sum(residuals**2) - n_data * jnp.log(sigma) - 0.5 * n_data * jnp.log(2 * jnp.pi)
        
        return loglik
    
    # Prior bounds for line fitting
    line_prior_bounds = {
        "m": (0.0, 5.0),      # slope
        "c": (-2.0, 4.0),     # intercept
        "sigma": (0.1, 2.0)   # noise level (must be positive)
    }
    
    # Test the likelihood
    test_params_line = {"m": true_m, "c": true_c, "sigma": true_sigma}
    test_loglik_line = line_loglikelihood(test_params_line)
    print(f"✓ Line likelihood defined, test value: {test_loglik_line:.3f}")
except Exception as e:
    print(f"✗ Line likelihood setup failed: {e}")
    exit(1)

print("\n=== Cell 4: Run Line Fitting Nested Sampling ===")
try:
    # Configuration for 3D problem
    line_num_live = 200  # Smaller for testing
    line_num_dims = 3
    line_num_inner_steps = line_num_dims * 5
    
    # Initialize prior and particles
    rng_key, subkey = jax.random.split(rng_key)
    line_particles, line_logprior_fn = ns_utils.uniform_prior(
        subkey, line_num_live, line_prior_bounds
    )
    
    # Create nested sampler
    line_sampler = blackjax.nss(
        logprior_fn=line_logprior_fn,
        loglikelihood_fn=line_loglikelihood,
        num_delete=20,  # Smaller for testing
        num_inner_steps=line_num_inner_steps
    )
    
    # Initialize and run
    line_live_state = line_sampler.init(line_particles)
    line_jit_step = jax.jit(line_sampler.step)
    
    print(f"✓ Line sampler initialized: {line_num_live} live points")
    
    # Run nested sampling (limited iterations for testing)
    line_dead_points = []
    line_iteration = 0
    start_time = time.time()
    max_iterations = 50  # Limit for testing
    
    while (line_live_state.logZ_live - line_live_state.logZ) > -3.0 and line_iteration < max_iterations:
        rng_key, subkey = jax.random.split(rng_key)
        
        line_live_state, line_dead_info = line_jit_step(subkey, line_live_state)
        line_dead_points.append(line_dead_info)
        
        line_iteration += 1
        
        if line_iteration % 10 == 0:
            remaining = line_live_state.logZ_live - line_live_state.logZ
            print(f"    Iteration {line_iteration:3d}: logZ = {line_live_state.logZ:.3f}, remaining = {remaining:.3f}")
    
    line_sampling_time = time.time() - start_time
    print(f"✓ Line sampling completed: {line_iteration} iterations, {line_sampling_time:.2f}s")
    
except Exception as e:
    print(f"✗ Line sampling failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== Cell 5: Process Line Fitting Results ===")
try:
    # Process and visualize line fitting results
    line_dead = ns_utils.finalise(line_live_state, line_dead_points)
    
    # Extract samples from dictionary format
    line_param_names = ['m', 'c', 'σ']
    param_keys = ['m', 'c', 'sigma']  # Keys in the particles dict
    
    # Convert dictionary particles to array for plotting
    line_samples_dict = line_dead.particles
    line_samples = jnp.stack([line_samples_dict[key] for key in param_keys], axis=1)
    
    # Create NestedSamples for anesthetic
    line_nested_samples = NestedSamples(
        data=line_samples,
        logL=line_dead.loglikelihood,
        logL_birth=line_dead.logL_birth,
        columns=line_param_names
    )
    
    # Print posterior summary
    print("Line fitting posterior summary:")
    line_true_params = jnp.array([true_m, true_c, true_sigma])
    for i, name in enumerate(line_param_names):
        mean = line_nested_samples[name].mean()
        std = line_nested_samples[name].std()
        true_val = line_true_params[i]
        print(f"  {name}: {mean:.3f} ± {std:.3f} (true: {true_val:.3f})")
    
    print(f"✓ Line results processed: {len(line_samples)} samples")
    
except Exception as e:
    print(f"✗ Line results processing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== Cell 6: 2D Gaussian Data Generation ===")
try:
    # Problem configuration
    image_size = 50
    x = jnp.linspace(-3, 3, image_size)
    y = jnp.linspace(-3, 3, image_size)
    X, Y = jnp.meshgrid(x, y)
    coords = jnp.stack([X.ravel(), Y.ravel()]).T  # (2500, 2)
    
    # True parameters for data generation
    true_params = jnp.array([0.5, -0.3, 1.2, 0.8, 0.4])  # [μₓ, μᵧ, σₓ, σᵧ, ρₓᵧ]
    
    @jax.jit
    def params_to_cov(params):
        """Convert parameters to mean vector and covariance matrix."""
        mu_x, mu_y, sigma_x, sigma_y, rho = params
        
        mean = jnp.array([mu_x, mu_y])
        
        # Covariance matrix
        cov = jnp.array([
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2]
        ])
        
        return mean, cov
    
    @jax.jit
    def simulator(params, rng_key, noise_sigma=0.1):
        """Simulate 2D Gaussian observation with noise."""
        mean, cov = params_to_cov(params)
        
        # Evaluate multivariate normal PDF on grid
        logpdf = jax.scipy.stats.multivariate_normal.logpdf(coords, mean, cov)
        image_clean = logpdf.reshape(image_size, image_size)
        image_clean = jnp.exp(image_clean - jnp.max(image_clean))  # Normalize
        
        # Add Gaussian noise
        noise = jax.random.normal(rng_key, image_clean.shape) * noise_sigma
        
        return image_clean + noise
    
    # Generate observed data
    rng_key, subkey = jax.random.split(rng_key)
    observed_data = simulator(true_params, subkey)
    
    print(f"✓ 2D Gaussian data generated: {image_size}×{image_size} image")
    print(f"  True parameters: μₓ={true_params[0]:.2f}, μᵧ={true_params[1]:.2f}, "
          f"σₓ={true_params[2]:.2f}, σᵧ={true_params[3]:.2f}, ρ={true_params[4]:.2f}")
    
except Exception as e:
    print(f"✗ 2D Gaussian data generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== Cell 7: 2D Gaussian Likelihood ===")
try:
    @jax.jit
    def loglikelihood_fn(params_dict):
        """Log-likelihood function for parameter inference."""
        # Extract parameters from dictionary
        mu_x = params_dict["mu_x"]
        mu_y = params_dict["mu_y"]
        sigma_x = params_dict["sigma_x"]
        sigma_y = params_dict["sigma_y"]
        rho = params_dict["rho"]
        
        # Convert to mean and covariance
        mean = jnp.array([mu_x, mu_y])
        cov = jnp.array([
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2]
        ])
        
        # Check if covariance matrix is positive definite
        det_cov = jnp.linalg.det(cov)
        
        # Return -inf if covariance is not positive definite
        def valid_cov():
            logpdf = jax.scipy.stats.multivariate_normal.logpdf(coords, mean, cov)
            image_pred = logpdf.reshape(image_size, image_size)
            image_pred = jnp.exp(image_pred - jnp.max(image_pred))
            
            # Gaussian likelihood (MSE)
            noise_sigma = 0.1
            residuals = (observed_data - image_pred) / noise_sigma
            return -0.5 * jnp.sum(residuals**2)
        
        def invalid_cov():
            return -jnp.inf
        
        return jax.lax.cond(det_cov > 0, valid_cov, invalid_cov)
    
    # Prior bounds
    prior_bounds = {
        "mu_x": (-2.0, 2.0),
        "mu_y": (-2.0, 2.0), 
        "sigma_x": (0.5, 3.0),
        "sigma_y": (0.5, 3.0),
        "rho": (-0.99, 0.99)  # Correlation must be in (-1, 1)
    }
    
    # Test likelihood function
    test_params_dict = {
        "mu_x": true_params[0],
        "mu_y": true_params[1], 
        "sigma_x": true_params[2],
        "sigma_y": true_params[3],
        "rho": true_params[4]
    }
    test_loglik = loglikelihood_fn(test_params_dict)
    print(f"✓ 2D Gaussian likelihood defined, test value: {test_loglik:.3f}")
    
except Exception as e:
    print(f"✗ 2D Gaussian likelihood setup failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== Cell 8: Test Small 2D Gaussian Sampling ===")
try:
    # Test with very small configuration
    print("Testing small 2D Gaussian nested sampling...")
    
    num_live_test = 100  # Very small for testing
    num_dims = 5
    
    # Initialize prior and particles
    rng_key, subkey = jax.random.split(rng_key)
    particles_test, logprior_fn_test = ns_utils.uniform_prior(
        subkey, num_live_test, prior_bounds
    )
    
    # Create nested sampler
    sampler_test = blackjax.nss(
        logprior_fn=logprior_fn_test,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=10,
        num_inner_steps=15
    )
    
    # Initialize
    live_state_test = sampler_test.init(particles_test)
    jit_step_test = jax.jit(sampler_test.step)
    
    # Run a few steps
    for i in range(5):
        rng_key, subkey = jax.random.split(rng_key)
        live_state_test, dead_info_test = jit_step_test(subkey, live_state_test)
        print(f"  Step {i+1}: logZ = {live_state_test.logZ:.3f}")
    
    print("✓ 2D Gaussian sampling test successful")
    
except Exception as e:
    print(f"✗ 2D Gaussian sampling test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 Full notebook test completed successfully!")
print("\nSummary:")
print(f"- Line fitting: {line_iteration} iterations, {line_sampling_time:.2f}s")
print(f"- Line samples: {len(line_samples)} final samples")
print(f"- 2D Gaussian: Basic sampling workflow tested")
print("\nThe notebook should run end-to-end!")