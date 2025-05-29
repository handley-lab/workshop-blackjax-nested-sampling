#!/usr/bin/env python3
"""
Test script for the performance comparison section
"""

def test_nuts_comparison():
    """Test NUTS comparison functionality"""
    print("Testing NUTS comparison setup...")
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        
        # First create some dummy data for the comparison
        true_params = jnp.array([0.5, -0.3, 1.2, 0.8, 0.4])
        
        # Test transform functions
        def transform_to_unconstrained(params):
            """Transform to unconstrained space for NUTS."""
            mu_x, mu_y, sigma_x, sigma_y, rho = params
            
            # Log transform for positive parameters
            log_sigma_x = jnp.log(sigma_x)
            log_sigma_y = jnp.log(sigma_y)
            
            # Logit transform for correlation [-1, 1] -> R
            logit_rho = jnp.log((rho + 1) / (2 - rho))
            
            return jnp.array([mu_x, mu_y, log_sigma_x, log_sigma_y, logit_rho])

        def transform_to_constrained(unconstrained_params):
            """Transform back to constrained space."""
            mu_x, mu_y, log_sigma_x, log_sigma_y, logit_rho = unconstrained_params
            
            sigma_x = jnp.exp(log_sigma_x)
            sigma_y = jnp.exp(log_sigma_y)
            rho = 2 * jax.nn.sigmoid(logit_rho) - 1
            
            return jnp.array([mu_x, mu_y, sigma_x, sigma_y, rho])

        # Test transforms
        unconstrained = transform_to_unconstrained(true_params)
        reconstructed = transform_to_constrained(unconstrained)
        
        print(f"Original: {true_params}")
        print(f"Unconstrained: {unconstrained}")
        print(f"Reconstructed: {reconstructed}")
        print(f"Transform error: {jnp.max(jnp.abs(true_params - reconstructed)):.6f}")
        
        # Test NUTS kernel setup
        from blackjax.mcmc.nuts import build_kernel
        
        @jax.jit
        def dummy_logdensity(unconstrained_params):
            """Dummy log-density for testing."""
            constrained_params = transform_to_constrained(unconstrained_params)
            mu_x, mu_y, sigma_x, sigma_y, rho = constrained_params
            
            # Simple quadratic for testing
            return -0.5 * jnp.sum((constrained_params - true_params)**2)

        nuts_kernel = build_kernel(dummy_logdensity)
        print("NUTS kernel created successfully")
        print("✓ NUTS comparison setup passed")
        
        return True
    except Exception as e:
        print(f"✗ NUTS comparison setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_aies_comparison():
    """Test AIES comparison functionality"""
    print("\nTesting AIES comparison setup...")
    try:
        import jax
        import jax.numpy as jnp
        
        # Try to import AIES
        try:
            from blackjax.mcmc.aies import init, build_kernel as aies_build_kernel
            aies_available = True
            print("AIES import successful")
        except ImportError:
            print("AIES not available in this BlackJAX version - this is expected")
            aies_available = False
            
        if aies_available:
            # Test AIES setup
            @jax.jit
            def dummy_logdensity(params):
                """Dummy log-density for AIES testing."""
                return -0.5 * jnp.sum(params**2)

            aies_kernel = aies_build_kernel(dummy_logdensity)
            print("AIES kernel created successfully")
            
        print("✓ AIES comparison setup passed")
        return True, aies_available
    except Exception as e:
        print(f"✗ AIES comparison setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def test_model_comparison():
    """Test model comparison functionality"""
    print("\nTesting model comparison setup...")
    try:
        import jax
        import jax.numpy as jnp
        import blackjax
        import blackjax.ns.utils as ns_utils
        
        # Create dummy likelihood for circular Gaussian model
        @jax.jit
        def circular_gaussian_loglikelihood(params_3d_dict):
            """Likelihood for circular Gaussian model (σₓ = σᵧ, ρ = 0)."""
            mu_x = params_3d_dict["mu_x"]
            mu_y = params_3d_dict["mu_y"] 
            sigma = params_3d_dict["sigma"]
            
            # Simple test likelihood
            params = jnp.array([mu_x, mu_y, sigma])
            return -0.5 * jnp.sum(params**2)

        # Model 2 prior bounds (3D)
        prior_bounds_3d = {
            "mu_x": (-2.0, 2.0),
            "mu_y": (-2.0, 2.0),
            "sigma": (0.5, 3.0)
        }

        # Test setup
        rng_key = jax.random.key(42)
        num_live_simple = 50  # Very small for testing
        rng_key, subkey = jax.random.split(rng_key)
        particles_3d, logprior_fn_3d = ns_utils.uniform_prior(
            subkey, num_live_simple, prior_bounds_3d
        )

        nested_sampler_3d = blackjax.nss(
            logprior_fn=logprior_fn_3d,
            loglikelihood_fn=circular_gaussian_loglikelihood,
            num_delete=25,
            num_inner_steps=15
        )

        live_state_3d = nested_sampler_3d.init(particles_3d)
        print(f"Model comparison setup with {num_live_simple} live points")
        print(f"Initial evidence: {live_state_3d.logZ:.3f}")
        
        print("✓ Model comparison setup passed")
        return True
    except Exception as e:
        print(f"✗ Model comparison setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comparison tests"""
    print("=" * 60)
    print("Performance Comparison Test")
    print("=" * 60)
    
    # Test NUTS
    if not test_nuts_comparison():
        return False
        
    # Test AIES
    success, aies_available = test_aies_comparison()
    if not success:
        return False
        
    # Test model comparison
    if not test_model_comparison():
        return False
        
    print("\n" + "=" * 60)
    print("✓ All comparison tests passed!")
    if not aies_available:
        print("  (Note: AIES not available - notebook will handle this gracefully)")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()