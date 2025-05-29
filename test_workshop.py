#!/usr/bin/env python3
"""
Test script to run through the BlackJAX nested sampling workshop notebook
and identify any runtime errors.
"""

def test_cell_1_setup():
    """Test initial setup and imports"""
    print("Testing Cell 1: Setup and imports...")
    try:
        # Install and setup (simulated - already done)
        import jax
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
        print("✓ Cell 1 passed")
        return True
    except Exception as e:
        print(f"✗ Cell 1 failed: {e}")
        return False

def test_cell_2_imports():
    """Test main imports"""
    print("\nTesting Cell 2: Main imports...")
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import matplotlib.pyplot as plt
        from functools import partial
        import time

        # BlackJAX imports
        import blackjax
        import blackjax.ns.utils as ns_utils

        # Visualization
        from anesthetic import NestedSamples
        import corner

        # JAX configuration for precision and reproducibility
        jax.config.update("jax_enable_x64", True)
        jax.config.update('jax_platform_name', 'cpu')  # Change to 'gpu' if available

        # Set random seed
        np.random.seed(42)
        rng_key = jax.random.key(42)
        
        print("✓ Cell 2 passed")
        return True, locals()
    except Exception as e:
        print(f"✗ Cell 2 failed: {e}")
        return False, {}

def test_cell_3_line_data():
    """Test line fitting data generation"""
    print("\nTesting Cell 3: Line data generation...")
    try:
        import jax.numpy as jnp
        import numpy as np
        import matplotlib.pyplot as plt
        
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

        print(f"Generated {n_data} data points")
        print(f"True parameters: m = {true_m}, c = {true_c}, σ = {true_sigma}")
        print("✓ Cell 3 passed")
        return True, {
            'x_data': x_data, 'y_data': y_data, 'n_data': n_data,
            'true_m': true_m, 'true_c': true_c, 'true_sigma': true_sigma
        }
    except Exception as e:
        print(f"✗ Cell 3 failed: {e}")
        return False, {}

def test_cell_4_line_likelihood(data):
    """Test line fitting likelihood"""
    print("\nTesting Cell 4: Line likelihood function...")
    try:
        import jax
        import jax.numpy as jnp
        
        x_data = data['x_data']
        y_data = data['y_data']
        n_data = data['n_data']
        true_m = data['true_m']
        true_c = data['true_c']
        true_sigma = data['true_sigma']
        
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
        print(f"Log-likelihood at true parameters: {test_loglik_line:.3f}")
        print("✓ Cell 4 passed")
        
        return True, {
            'line_loglikelihood': line_loglikelihood,
            'line_prior_bounds': line_prior_bounds,
            'test_loglik_line': test_loglik_line
        }
    except Exception as e:
        print(f"✗ Cell 4 failed: {e}")
        return False, {}

def test_cell_5_line_sampling(data, likelihood_data):
    """Test line fitting nested sampling"""
    print("\nTesting Cell 5: Line nested sampling...")
    try:
        import jax
        import jax.numpy as jnp
        import blackjax
        import blackjax.ns.utils as ns_utils
        import time
        
        line_loglikelihood = likelihood_data['line_loglikelihood']
        line_prior_bounds = likelihood_data['line_prior_bounds']
        
        print("Setting up nested sampling for line fitting...")

        # Configuration for 3D problem
        line_num_live = 100
        line_num_dims = 3
        line_num_inner_steps = line_num_dims * 5

        # Initialize prior and particles
        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)
        line_particles, line_logprior_fn = ns_utils.uniform_prior(
            subkey, line_num_live, line_prior_bounds
        )

        # Create nested sampler
        line_sampler = blackjax.nss(
            logprior_fn=line_logprior_fn,
            loglikelihood_fn=line_loglikelihood,
            num_delete=50,
            num_inner_steps=line_num_inner_steps
        )

        # Initialize and run
        line_live_state = line_sampler.init(line_particles)
        line_jit_step = jax.jit(line_sampler.step)

        print(f"Initialized {line_num_live} live points for 3D line fitting problem")

        # Run just a few iterations for testing
        line_dead_points = []
        line_iteration = 0
        start_time = time.time()
        max_iterations = 10  # Limit for testing

        while (line_live_state.logZ_live - line_live_state.logZ) > -3.0 and line_iteration < max_iterations:
            rng_key, subkey = jax.random.split(rng_key)
            
            line_live_state, line_dead_info = line_jit_step(subkey, line_live_state)
            line_dead_points.append(line_dead_info)
            
            line_iteration += 1
            
            if line_iteration % 5 == 0:
                remaining = line_live_state.logZ_live - line_live_state.logZ
                print(f"  Iteration {line_iteration:3d}: logZ = {line_live_state.logZ:.3f}, remaining = {remaining:.3f}")

        line_sampling_time = time.time() - start_time

        print(f"Line fitting test completed!")
        print(f"Test iterations: {line_iteration}")
        print(f"Sampling time: {line_sampling_time:.2f} seconds")
        print(f"Evidence: logZ = {line_live_state.logZ:.3f}")
        print("✓ Cell 5 passed")
        
        return True, {
            'line_live_state': line_live_state,
            'line_dead_points': line_dead_points,
            'line_iteration': line_iteration,
            'line_sampling_time': line_sampling_time,
            'rng_key': rng_key
        }
    except Exception as e:
        print(f"✗ Cell 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_cell_6_line_visualization(data, likelihood_data, sampling_data):
    """Test line fitting visualization"""
    print("\nTesting Cell 6: Line visualization...")
    try:
        import jax.numpy as jnp
        import blackjax.ns.utils as ns_utils
        from anesthetic import NestedSamples
        import matplotlib.pyplot as plt
        
        line_live_state = sampling_data['line_live_state']
        line_dead_points = sampling_data['line_dead_points']
        
        # Process and visualize line fitting results
        line_dead = ns_utils.finalise(line_live_state, line_dead_points)

        # Extract samples from dictionary format
        line_param_names = ['m', 'c', 'σ']
        param_keys = ['m', 'c', 'sigma']  # Keys in the particles dict

        # Convert dictionary particles to array for anesthetic
        line_samples_dict = line_dead.particles
        line_data = jnp.vstack([line_samples_dict[key] for key in param_keys]).T

        # Create NestedSamples for anesthetic
        line_nested_samples = NestedSamples(
            data=line_data,
            logL=line_dead.loglikelihood,
            logL_birth=line_dead.loglikelihood_birth,
            columns=line_param_names,
            labels=[r"$m$", r"$c$", r"$\sigma$"]
        )

        # Test basic properties
        print(f"Number of samples: {len(line_nested_samples)}")
        for name in line_param_names:
            mean = line_nested_samples[name].mean()
            std = line_nested_samples[name].std()
            print(f"  {name}: {mean:.3f} ± {std:.3f}")

        # Test evidence calculation
        evidence_mean = line_nested_samples.logZ()
        evidence_std = line_nested_samples.logZ(100).std()  # Reduce samples for testing
        print(f"Evidence: logZ = {evidence_mean:.3f} ± {evidence_std:.3f}")

        print("✓ Cell 6 passed")
        return True, {'line_nested_samples': line_nested_samples}
    except Exception as e:
        print(f"✗ Cell 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Run all tests"""
    print("=" * 60)
    print("BlackJAX Nested Sampling Workshop Test")
    print("=" * 60)
    
    # Test basic setup
    if not test_cell_1_setup():
        return False
        
    # Test imports
    success, imports = test_cell_2_imports()
    if not success:
        return False
        
    # Test line data
    success, data = test_cell_3_line_data()
    if not success:
        return False
        
    # Test likelihood
    success, likelihood_data = test_cell_4_line_likelihood(data)
    if not success:
        return False
        
    # Test sampling
    success, sampling_data = test_cell_5_line_sampling(data, likelihood_data)
    if not success:
        return False
        
    # Test visualization
    success, viz_data = test_cell_6_line_visualization(data, likelihood_data, sampling_data)
    if not success:
        return False
        
    print("\n" + "=" * 60)
    print("✓ All line fitting tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()