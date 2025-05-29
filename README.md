# BlackJAX Nested Sampling Workshop

**SBI Galev 2025 Conference**  
*Will Handley, University of Cambridge*

A hands-on workshop demonstrating GPU-native Bayesian inference with BlackJAX nested sampling.

## Overview

This workshop introduces **BlackJAX nested sampling** - a modern, GPU-accelerated implementation of nested sampling that leverages JAX's automatic differentiation and JIT compilation for scientific inference.

### Learning Objectives

- Understand when nested sampling excels over traditional MCMC methods
- Implement nested sampling with BlackJAX's high-level API
- Visualize results with Anesthetic (specialized nested sampling plots)
- Compare performance: nested sampling vs. NUTS vs. affine invariant ensemble sampling
- Apply GPU acceleration to scientific inference problems
- Perform Bayesian model comparison with evidence computation

### Key Features Demonstrated

- **GPU-native inference**: Full JAX integration with autodiff + JIT compilation
- **Multimodal handling**: Robust exploration of complex posterior geometries  
- **Evidence computation**: Essential for SBI model comparison workflows
- **Performance benchmarking**: Head-to-head comparison with MCMC methods
- **Open-source ecosystem**: Community-driven alternative to legacy Fortran tools

## Quick Start

### Option 1: Google Colab (Recommended)

Click the badge below to run the workshop in Google Colab with zero setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[username]/workshop-blackjax-nested-sampling/blob/master/blackjax_nested_sampling_workshop.ipynb)

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/[username]/workshop-blackjax-nested-sampling.git
cd workshop-blackjax-nested-sampling

# Install dependencies
pip install git+https://github.com/handley-lab/blackjax@nested_sampling
pip install anesthetic matplotlib corner tqdm jupyter

# Launch Jupyter
jupyter notebook blackjax_nested_sampling_workshop.ipynb
```

### Option 3: Conda Environment

```bash
# Create environment
conda create -n blackjax-workshop python=3.9
conda activate blackjax-workshop

# Install JAX (CPU or GPU)
pip install jax jaxlib  # CPU version
# pip install jax[cuda12_pip]  # GPU version (if CUDA available)

# Install BlackJAX and visualization tools
pip install git+https://github.com/handley-lab/blackjax@nested_sampling
pip install anesthetic matplotlib corner tqdm jupyter

# Run workshop
jupyter notebook blackjax_nested_sampling_workshop.ipynb
```

## Workshop Structure

### 1. **Setup and Installation** (5 min)
- BlackJAX nested sampling branch installation
- JAX configuration for GPU acceleration
- Environment verification

### 2. **Problem Setup** (10 min)
- 2D Gaussian parameter inference from noisy images
- Building on Viraj Pandya's JAX/SciML workshop content
- Likelihood and prior specification

### 3. **BlackJAX Nested Sampling** (15 min)
- High-level API walkthrough (`blackjax.nss`)
- Configuration and initialization
- Running nested sampling to convergence
- Understanding evidence computation

### 4. **Results and Visualization** (10 min)
- Processing nested sampling output
- Anesthetic corner plots and diagnostics
- Posterior summary statistics
- Evidence evolution plots

### 5. **Performance Comparison** (10 min)
- Head-to-head: Nested Sampling vs. NUTS vs. AIES
- Timing benchmarks and accuracy assessment
- When to choose each method

### 6. **Advanced Features** (5 min)
- Bayesian model comparison demonstration
- GPU performance scaling
- Integration with SBI workflows

### 7. **Your Turn** (Remaining time)
- Apply BlackJAX to your own research problems
- Experiment with different configurations
- Template for custom likelihood functions

## Key Problem: 2D Gaussian Parameter Inference

The workshop centers on a **5-dimensional parameter inference problem**:

- **Parameters**: [μₓ, μᵧ, σₓ, σᵧ, ρₓᵧ] (position, scale, correlation)
- **Data**: 50×50 pixel noisy observations of 2D Gaussians
- **Challenge**: Multimodal posterior due to parameter symmetries and correlations
- **Ground truth**: Known parameters for validation

This problem highlights nested sampling's advantages:
- **Complex geometry**: Correlation parameter creates constrained parameter space
- **Potential multimodality**: Symmetries can create multiple posterior modes
- **MCMC challenges**: Traditional methods may struggle with mode mixing
- **Evidence computation**: Natural for model comparison scenarios

## Dependencies

### Core Requirements
- **Python**: ≥3.8
- **JAX**: ≥0.4.16 (automatic differentiation + JIT compilation)
- **BlackJAX**: nested_sampling branch (GPU-native MCMC/nested sampling)
- **Anesthetic**: ≥2.0 (nested sampling visualization)

### Visualization
- **Matplotlib**: General plotting
- **Corner**: Corner plots for parameter inference
- **NumPy**: Numerical operations

### Development Tools
- **Jupyter**: Interactive notebook environment
- **tqdm**: Progress bars for long-running samplers

## Technical Highlights

### BlackJAX Nested Sampling API

```python
import blackjax

# High-level nested sampling interface
sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=50,        # GPU parallelization parameter
    num_inner_steps=25    # MCMC steps per NS iteration
)

# Initialize and run
live_state = sampler.init(particles)
while not_converged:
    live_state, dead_info = sampler.step(rng_key, live_state)
```

### Key Configuration Parameters

- **`num_live`**: Live points (precision scales as √num_live, runtime linear)
- **`num_inner_steps`**: MCMC steps per iteration (recommend 3-5 × dimension)  
- **`num_delete`**: Parallel deletion for GPU efficiency
- **Termination**: Evidence contribution threshold (typically `logZ_live - logZ < -3`)

### JAX Configuration for Performance

```python
import jax
jax.config.update("jax_enable_x64", True)     # Higher precision
jax.config.update('jax_platform_name', 'gpu') # GPU acceleration
jax.config.update('jax_num_cpu_devices', 8)   # Multi-core CPU fallback
```

## Comparison with Alternatives

| Method | Strengths | Limitations | Best For |
|--------|-----------|-------------|----------|
| **BlackJAX NS** | Multimodal, evidence, GPU-native | More setup than MCMC | Model comparison, complex posteriors |
| **NUTS** | Fast for unimodal, gradients | Gradient-only, mode mixing | Smooth posteriors, fast sampling |
| **AIES (emcee)** | Gradient-free, simple | CPU-only, slow convergence | Legacy compatibility, simple problems |
| **dynesty** | Mature NS, CPU | Python overhead, no GPU | Standard NS applications |
| **PolyChord** | Robust, battle-tested | Fortran, no autodiff | Production NS with legacy code |

## Performance Characteristics

### Scaling Properties
- **Evidence precision**: ∝ 1/√num_live
- **Runtime**: ∝ num_live × num_inner_steps × likelihood_cost
- **GPU acceleration**: Significant speedup for large num_live
- **Memory**: Manageable for problems up to ~100 dimensions

### When to Use BlackJAX Nested Sampling

✅ **Excellent for:**
- Simulation-based inference (NLE, NRE, NJE workflows)
- Multimodal posteriors (phase transitions, symmetries)
- Model comparison (Bayesian evidence computation)
- GPU-accelerated scientific computing
- High-dimensional problems (>10 parameters)
- Integration with JAX-based modeling

❌ **Consider alternatives for:**
- Simple unimodal posteriors (NUTS may be faster)
- Very low-dimensional problems (<3 parameters)
- CPU-only environments (dynesty, emcee)
- When only posterior samples needed (not evidence)

## Resources and Further Reading

### BlackJAX Ecosystem
- **BlackJAX Repository**: https://github.com/handley-lab/blackjax
- **Documentation**: https://blackjax-devs.github.io/blackjax/
- **Issue Tracker**: https://github.com/handley-lab/blackjax/issues

### Nested Sampling Visualization
- **Anesthetic**: https://anesthetic.readthedocs.io/en/latest/plotting.html
- **Corner plots**: Parameter inference visualization
- **Evidence evolution**: Nested sampling diagnostics

### JAX Ecosystem
- **JAX Documentation**: https://jax.readthedocs.io/
- **NumPyro**: https://num.pyro.ai/ (Probabilistic programming)
- **Optax**: https://optax.readthedocs.io/ (Gradient-based optimization)
- **Flax**: https://flax.readthedocs.io/ (Neural networks)

### Scientific Background
- **Nested Sampling Review**: Skilling (2006), "Nested sampling for general Bayesian computation"
- **SBI Review**: Cranmer et al. (2020), "The frontier of simulation-based inference"
- **JAX**: Bradbury et al. (2018), "JAX: composable transformations of Python+NumPy programs"

## Support and Community

### Getting Help
- **Workshop Issues**: Open GitHub issue in this repository
- **BlackJAX Questions**: Use BlackJAX issue tracker or discussion forums
- **JAX Questions**: JAX GitHub discussions or Stack Overflow

### Contributing
BlackJAX is a community-driven project. Contributions welcome:
- **Bug reports**: Help improve reliability
- **Feature requests**: Suggest new capabilities  
- **Documentation**: Improve tutorials and examples
- **Code contributions**: Implement new samplers or features

### Citation

If you use BlackJAX nested sampling in your research, please cite:

```bibtex
@software{blackjax2024,
  title={BlackJAX: A Sampling Library for JAX},
  author={BlackJAX Developers},
  url={https://github.com/handley-lab/blackjax},
  year={2024}
}
```

---

## Workshop Outcomes

By the end of this workshop, you will:

1. **Understand** when nested sampling provides advantages over MCMC methods
2. **Implement** nested sampling with BlackJAX's modern, GPU-native API
3. **Visualize** results with specialized nested sampling diagnostic plots
4. **Benchmark** performance against traditional sampling methods
5. **Apply** these techniques to your own scientific inference problems
6. **Integrate** nested sampling into SBI and JAX-based modeling workflows

**The future of scientific inference is GPU-native, open-source, and community-driven.**

---

*Workshop developed for SBI Galev 2025 by Will Handley, University of Cambridge*