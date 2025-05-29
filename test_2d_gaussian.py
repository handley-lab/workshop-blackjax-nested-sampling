#!/usr/bin/env python3
"""
Test script for the 2D Gaussian parameter inference section
"""

def test_2d_gaussian_setup():
    """Test 2D Gaussian data generation"""
    print("Testing 2D Gaussian setup...")
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        
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
        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)
        observed_data = simulator(true_params, subkey)

        print(f"Generated {image_size}x{image_size} image")
        print(f"True parameters: μₓ={true_params[0]:.2f}, μᵧ={true_params[1]:.2f}, "
              f"σₓ={true_params[2]:.2f}, σᵧ={true_params[3]:.2f}, ρ={true_params[4]:.2f}")
        print("✓ 2D Gaussian setup passed")
        
        return True, {
            'image_size': image_size,
            'coords': coords,
            'true_params': true_params,
            'observed_data': observed_data,
            'rng_key': rng_key
        }
    except Exception as e:
        print(f"✗ 2D Gaussian setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_2d_gaussian_likelihood(data):
    """Test 2D Gaussian likelihood function"""
    print("\nTesting 2D Gaussian likelihood...")
    try:
        import jax
        import jax.numpy as jnp
        
        coords = data['coords']
        observed_data = data['observed_data']
        image_size = data['image_size']
        true_params = data['true_params']
        
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
        print(f"Log-likelihood at true parameters: {test_loglik:.3f}")
        print("✓ 2D Gaussian likelihood passed")
        
        return True, {
            'loglikelihood_fn': loglikelihood_fn,
            'prior_bounds': prior_bounds,
            'test_loglik': test_loglik
        }
    except Exception as e:
        print(f"✗ 2D Gaussian likelihood failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_2d_gaussian_sampling(data, likelihood_data):
    """Test 2D Gaussian nested sampling"""
    print("\nTesting 2D Gaussian nested sampling...")
    try:
        import jax
        import blackjax
        import blackjax.ns.utils as ns_utils
        import time
        
        loglikelihood_fn = likelihood_data['loglikelihood_fn']
        prior_bounds = likelihood_data['prior_bounds']
        rng_key = data['rng_key']
        
        # Nested sampling configuration
        num_live = 100  # Reduced for testing
        num_dims = 5
        num_inner_steps = num_dims * 5
        num_delete = 50

        print(f"Nested sampling configuration:")
        print(f"  Live points: {num_live}")
        print(f"  Inner MCMC steps: {num_inner_steps}")
        print(f"  Parallel deletion: {num_delete}")

        # Initialize uniform prior and live points
        rng_key, subkey = jax.random.split(rng_key)
        particles, logprior_fn = ns_utils.uniform_prior(
            subkey, num_live, prior_bounds
        )

        # Create nested sampler
        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps
        )

        # Initialize nested sampling state
        live_state = nested_sampler.init(particles)

        # JIT compile the sampling step for efficiency
        jit_step = jax.jit(nested_sampler.step)

        print(f"Initial evidence estimate: {live_state.logZ:.3f}")
        print(f"Initial live evidence: {live_state.logZ_live:.3f}")

        # Run limited sampling for testing
        dead_points = []
        iteration = 0
        convergence_threshold = -3.0
        max_iterations = 20  # Limit for testing

        start_time = time.time()

        while (live_state.logZ_live - live_state.logZ) > convergence_threshold and iteration < max_iterations:
            rng_key, subkey = jax.random.split(rng_key)
            
            # Take nested sampling step
            live_state, dead_info = jit_step(subkey, live_state)
            dead_points.append(dead_info)
            
            iteration += 1
            
            # Progress updates
            if iteration % 10 == 0:
                remaining_evidence = live_state.logZ_live - live_state.logZ
                print(f"Iteration {iteration:4d}: logZ = {live_state.logZ:.3f}, "
                      f"remaining = {remaining_evidence:.3f}")

        sampling_time = time.time() - start_time

        print(f"2D Gaussian sampling test completed!")
        print(f"Test iterations: {iteration}")
        print(f"Sampling time: {sampling_time:.2f} seconds")
        print(f"Final evidence: logZ = {live_state.logZ:.3f}")
        print("✓ 2D Gaussian sampling passed")
        
        return True, {
            'live_state': live_state,
            'dead_points': dead_points,
            'iteration': iteration,
            'sampling_time': sampling_time
        }
    except Exception as e:
        print(f"✗ 2D Gaussian sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_2d_gaussian_processing(data, likelihood_data, sampling_data):
    """Test 2D Gaussian result processing"""
    print("\nTesting 2D Gaussian result processing...")
    try:
        import jax.numpy as jnp
        import blackjax.ns.utils as ns_utils
        from anesthetic import NestedSamples
        
        live_state = sampling_data['live_state']
        dead_points = sampling_data['dead_points']
        true_params = data['true_params']
        
        # Process results and create anesthetic NestedSamples
        dead = ns_utils.finalise(live_state, dead_points)

        # Extract samples from dictionary format for anesthetic
        param_names = ['μₓ', 'μᵧ', 'σₓ', 'σᵧ', 'ρ']
        param_keys = ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho']

        # Convert dictionary particles to array
        samples_dict = dead.particles
        samples = jnp.vstack([samples_dict[key] for key in param_keys]).T

        # Create NestedSamples for anesthetic
        nested_samples = NestedSamples(
            data=samples,
            logL=dead.loglikelihood,
            logL_birth=dead.loglikelihood_birth,
            columns=param_names,
            labels=[r"$\mu_x$", r"$\mu_y$", r"$\sigma_x$", r"$\sigma_y$", r"$\rho$"]
        )

        # Print posterior summary
        print("2D Gaussian parameter inference results:")
        for i, name in enumerate(param_names):
            mean = nested_samples[name].mean()
            std = nested_samples[name].std()
            true_val = true_params[i]
            print(f"  {name}: {mean:.3f} ± {std:.3f} (true: {true_val:.3f})")

        # Evidence calculation with error estimate
        evidence_mean = nested_samples.logZ()
        evidence_std = nested_samples.logZ(100).std()  # Reduced for testing
        print(f"Evidence: logZ = {evidence_mean:.3f} ± {evidence_std:.3f}")
        
        print("✓ 2D Gaussian processing passed")
        return True, {'nested_samples': nested_samples}
    except Exception as e:
        print(f"✗ 2D Gaussian processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Run 2D Gaussian tests"""
    print("=" * 60)
    print("2D Gaussian Parameter Inference Test")
    print("=" * 60)
    
    # Test setup
    success, data = test_2d_gaussian_setup()
    if not success:
        return False
        
    # Test likelihood
    success, likelihood_data = test_2d_gaussian_likelihood(data)
    if not success:
        return False
        
    # Test sampling
    success, sampling_data = test_2d_gaussian_sampling(data, likelihood_data)
    if not success:
        return False
        
    # Test processing
    success, processing_data = test_2d_gaussian_processing(data, likelihood_data, sampling_data)
    if not success:
        return False
        
    print("\n" + "=" * 60)
    print("✓ All 2D Gaussian tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()