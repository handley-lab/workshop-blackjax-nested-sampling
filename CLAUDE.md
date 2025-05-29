# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains materials for a workshop on nested sampling in BlackJAX, part of the SBI Galaxy Evolution 2025 conference. The project structure includes:

- **Workshop materials** (`prompt-materials/`) from related SBI, JAX, and machine learning workshops
- **Talk preparation** materials for a 1-hour session (15-minute talk + 45-minute hands-on workshop)
- **Target deliverable**: A Jupyter notebook workshop runnable in Google Colab

## Architecture

### Core Focus Areas

1. **BlackJAX Nested Sampling**: GPU-native nested sampling implementation
2. **JAX Integration**: Leveraging automatic differentiation and JIT compilation
3. **Simulation-Based Inference (SBI)**: Modern Bayesian inference techniques
4. **Educational Workshop Design**: Hands-on learning with minimal setup friction

### Key Dependencies & Frameworks

- **JAX**: Primary computational framework (autodiff + JIT compilation)
- **BlackJAX**: MCMC and nested sampling library
- **Anesthetic**: Visualization library for nested sampling results
- **NumPy/SciPy**: Core scientific computing
- **Matplotlib**: Plotting and visualization
- **Jupyter**: Interactive notebook environment

### Workshop Content Structure

**Talk Component** (~5 slides, 10-15 minutes):
- Position nested sampling as essential for most SBI methods (except NPE)
- Contrast BlackJAX (GPU-native, open-source) with legacy implementations
- Emphasize JAX's dual strengths: autodiff + JIT compilation

**Workshop Component** (~45 minutes):
- Hands-on nested sampling with BlackJAX
- Anesthetic visualization workflows
- Performance comparison: nested sampling vs. affine invariant ensemble sampling

## Development Commands

### Jupyter Notebooks
```bash
# Launch Jupyter for workshop development
jupyter notebook

# For Google Colab compatibility, ensure notebooks use:
# !pip install blackjax anesthetic
```

### LaTeX Presentations
```bash
# Build presentation materials
pdflatex will_handley.tex
bibtex will_handley
pdflatex will_handley.tex
pdflatex will_handley.tex
```

### JAX Configuration
```python
# Standard JAX setup for notebooks
import jax
jax.config.update("jax_enable_x64", True)  # Higher precision
jax.config.update('jax_num_cpu_devices', 8)  # Multi-core
```

## Key Implementation Patterns

### BlackJAX Nested Sampling Workflow
1. Define likelihood function (JAX-compatible)
2. Specify prior distribution
3. Configure nested sampling algorithm
4. Run sampling with JIT compilation
5. Visualize results with Anesthetic

### Educational Notebook Structure
- **Setup**: Minimal pip installs for Colab compatibility
- **Theory**: Brief conceptual introduction
- **Practice**: Hands-on coding exercises
- **Comparison**: Performance benchmarking against alternatives
- **Extension**: Encourage experimentation with user data

## Reference Materials

### Primary Documentation
- **BlackJAX repository**: https://github.com/handley-lab/blackjax
- **Anesthetic documentation**: https://anesthetic.readthedocs.io/en/latest/plotting.html
- **JAX documentation**: https://jax.readthedocs.io/

### Workshop-Specific Materials
- **`prompt-materials/blackjax/`**: BlackJAX examples and patterns
  - `line.py`: Reference implementation for line fitting with proper anesthetic integration
  - `docs/examples/nested_sampling.py`: Advanced nested sampling examples
- **`prompt-materials/anesthetic/`**: Anesthetic library documentation and examples
  - `docs/source/plotting.rst`: Comprehensive plotting guide
  - `anesthetic/examples/perfect_ns.py`: Perfect nested sampling generators
  - `tests/test_examples.py`: Usage examples and API patterns
- **Previous talks** (in `prompt-materials/talks/`): Source material for adapted content
- **Workshop context** (in `prompt-materials/`): SBI, JAX, and ILI reference materials

### Key API Patterns Learned
- **NestedSamples class**: Proper post-processing of BlackJAX results
- **Evidence computation**: `samples.logZ()` and `samples.logZ(nsamples).std()`
- **Anesthetic plotting**: `plot_2d()` with kinds parameter for visualization
- **Transform functions**: Proper arctanh/tanh transforms for constrained parameters

## Workshop Development Notes

- Target audience: Researchers familiar with JAX and SBI concepts
- Delivery platform: Google Colab (minimize installation friction)
- Duration: 1 hour total (15-min talk + 45-min hands-on)
- Key comparison: BlackJAX nested sampling vs. existing tools (dynesty, emcee)
- Integration opportunity: Build on Viraj Pandya's JAX/SciML workshop content

### Performance Configuration
- **Recommended settings**: 100 live points, num_delete=50 for workshop timing
- **Educational progression**: Line fitting → 2D Gaussian → Performance comparison
- **Error handling**: Proper covariance matrix validation and parameter transforms