# BlackJAX Nested Sampling Workshop

**GPU-Native Bayesian Inference with JAX and BlackJAX**

📖 **Essential Reading**: For the authoritative reference on blackjax nested sampling theory and applications, see the [Nested Sampling Book](https://handley-lab.co.uk/nested-sampling-book) by David Yallup.

> *"Nested sampling is a Bayesian computational technique that solves the key problem of evidence evaluation"* — from the [Nested Sampling Book](https://handley-lab.co.uk/nested-sampling-book)

## 🚀 Workshop Notebooks

### Interactive Workshop (Clean)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/handley-lab/workshop-blackjax-nested-sampling/blob/master/workshop_nested_sampling.ipynb)

**👆 Click here to run the workshop interactively in Google Colab**

### Preview with Results (Executed)
[![View on GitHub](https://img.shields.io/badge/view-GitHub-blue?logo=github)](https://github.com/handley-lab/workshop-blackjax-nested-sampling/blob/master/workshop_nested_sampling_executed.ipynb)

**👆 Click here to preview the workshop with all plots and outputs**

---

This interactive workshop demonstrates modern nested sampling using BlackJAX, a GPU-native probabilistic programming library built on JAX. The modular design allows flexible delivery from a 20-minute core workshop to a comprehensive 110-minute session covering advanced topics. Learn how to leverage automatic differentiation and JIT compilation for high-performance Bayesian inference.

## 📁 Repository Structure

```
workshop-blackjax-nested-sampling/
├── workshop_nested_sampling.py          # Source script (development)
├── workshop_nested_sampling.ipynb       # Clean interactive notebook
├── workshop_nested_sampling_executed.ipynb  # Pre-executed with outputs
├── CLAUDE.md                            # Claude development guidance
├── CLAUDE_WORKSHOP_TEMPLATE.md          # Workshop development template
├── README.md                            # This file
└── development/                         # Development materials
    ├── docs/                           # Development documentation
    ├── history/                        # Development conversation logs
    ├── reference-materials/            # Source materials and examples
    └── scripts/                        # Helper scripts and utilities
```

## 🎯 Workshop Overview

**Core Duration:** 20 minutes (suitable for talks)  
**Full Duration:** 110 minutes (core + extensions)  
**Format:** Hands-on Jupyter notebook  
**Platform:** Runnable in Google Colab (no local installation required)

### What You'll Learn

**Core Workshop (20 minutes):**
1. **GPU-Native Nested Sampling** with BlackJAX
2. **JAX Integration** for automatic differentiation and JIT compilation  
3. **Anesthetic Visualization** for professional nested sampling post-processing
4. **Performance Comparisons** between different sampling algorithms

**Advanced Extensions (90 minutes optional):**
5. **Custom Sampler Development** using BlackJAX's modular components
6. **JAX Ecosystem Integration** with gradient descent and optimization
7. **Simulation-Based Inference** with neural posterior estimation

## 📚 Workshop Content

### 🎯 Core Workshop (20 minutes)

#### Part 1: Linear Regression (5 minutes)
- Basic nested sampling workflow
- Evidence computation and uncertainty quantification
- Posterior visualization with true value overlays

#### Part 2: 2D Gaussian Inference (5 minutes)  
- Multivariate parameter estimation
- Correlation coefficient inference with proper transforms
- Advanced anesthetic plotting techniques

#### Part 3: Performance Comparison (10 minutes)
- BlackJAX nested sampling vs. NUTS (Hamiltonian Monte Carlo)
- Timing benchmarks and sampler trade-offs
- When to use nested sampling vs. other methods

### 🚀 Advanced Extensions (90 minutes optional)

#### Part 4: Building Custom Nested Samplers (30 minutes)
- Understanding BlackJAX's modular architecture
- Implementing custom MCMC kernels and adaptive schemes
- Research applications and specialized sampling strategies

#### Part 5: JAX Ecosystem Integration (30 minutes)
- Gradient-based optimization with Optax
- Image-based inference problems
- Complementary strengths of different approaches

#### Part 6: Simulation-Based Inference (SBI) (30 minutes)
- Neural posterior estimation with Flax
- Amortized inference and training workflows
- Modern SBI vs. traditional Bayesian methods

## 🚀 Quick Start

### Option 1: Interactive Workshop in Google Colab (Recommended)
Click the **"Open in Colab"** badge above to run the clean, interactive workshop in your browser. No installation required!

### Option 2: Preview with Results
Click the **"View on GitHub"** badge to see the workshop with all plots and outputs pre-executed for quick reference.

### Option 3: Local Installation
```bash
# Clone the repository
git clone https://github.com/handley-lab/workshop-blackjax-nested-sampling.git
cd workshop-blackjax-nested-sampling

# Core dependencies (required for Parts 1-3)
pip install git+https://github.com/handley-lab/blackjax
pip install anesthetic tqdm matplotlib jupyter

# Advanced extensions (optional for Parts 4-6)
pip install optax flax

# Launch the notebook (clean version)
jupyter notebook workshop_nested_sampling.ipynb

# Or view the executed version with plots
jupyter notebook workshop_nested_sampling_executed.ipynb
```

### Option 4: Run Python Script
```bash
# Run the standalone Python script
python workshop_nested_sampling.py
```

## 🔧 Key Dependencies

### Core Workshop
- **JAX**: Automatic differentiation and JIT compilation
- **BlackJAX**: GPU-native MCMC and nested sampling  
- **Anesthetic**: Nested sampling visualization and analysis
- **NumPy/SciPy**: Scientific computing foundations
- **Matplotlib**: Plotting and visualization

### Advanced Extensions (Optional)
- **Optax**: Gradient-based optimization (Part 5)
- **Flax**: Neural networks and SBI (Part 6)
- Additional JAX ecosystem packages

## 📊 What You'll Build

The workshop generates several publication-ready visualizations:

- **Data Visualization**: Synthetic datasets with true model overlays
- **Posterior Plots**: Corner plots with true parameter markers  
- **Performance Comparisons**: Sampler timing and accuracy benchmarks
- **Evidence Computation**: Bayesian model comparison metrics

## 🎓 Prerequisites

- **Python Experience**: Basic familiarity with NumPy/SciPy
- **Bayesian Inference**: Understanding of posteriors, priors, and likelihoods
- **Optional**: Previous exposure to MCMC methods (helpful but not required)

## 🌟 Key Features

### GPU-Native Performance
- **JAX Backend**: Automatic vectorization and GPU acceleration
- **JIT Compilation**: Near-compiled performance from Python code
- **Automatic Differentiation**: Efficient gradient computation

### Professional Workflows  
- **Anesthetic Integration**: Industry-standard nested sampling post-processing
- **Evidence Computation**: Natural Bayesian model comparison
- **Parameter Transforms**: Proper handling of constrained parameters

### Educational Design
- **Progressive Complexity**: From simple line fitting to multivariate inference
- **Hands-on Examples**: Interactive code cells with immediate feedback
- **Performance Insights**: Real timing comparisons between methods

## 🔗 Related Resources

### BlackJAX Documentation
- **Main Repository**: [handley-lab/blackjax](https://github.com/handley-lab/blackjax)
- **Nested Sampling Branch**: Focus on the `nested_sampling` branch for latest features

### Anesthetic Documentation  
- **Documentation**: [anesthetic.readthedocs.io](https://anesthetic.readthedocs.io/en/latest/plotting.html)
- **Plotting Examples**: Comprehensive visualization gallery

### JAX Ecosystem
- **JAX Documentation**: [jax.readthedocs.io](https://jax.readthedocs.io/)
- **Scientific Computing**: Auto-differentiation and JIT compilation tutorials

## 🤝 Contributing

This workshop was developed for the **SBI Galaxy Evolution 2025** conference. Contributions and improvements are welcome!

### Workshop Context
- **Event**: SBI Galev 2025 ([sbi-galev.github.io/2025](https://sbi-galev.github.io/2025/))
- **Session**: Nested sampling for simulation-based inference
- **Builds On**: JAX/SciML workshop content by Viraj Pandya

## 📄 License

This workshop is open-source and available for educational use. Please see individual dependency licenses for JAX, BlackJAX, and Anesthetic.

## 💡 Next Steps

After completing this workshop, consider:

1. **Apply to Your Data**: Use BlackJAX nested sampling on your research problems
2. **Explore Other Samplers**: Try BlackJAX's HMC, NUTS, and MALA implementations  
3. **GPU Acceleration**: Run on TPUs/GPUs for large-scale inference problems
4. **Model Comparison**: Use evidence computation for Bayesian model selection
5. **Community**: Join discussions in the BlackJAX and JAX communities

---

**Workshop Development**: Generated with [Claude Code](https://claude.ai/code) • **Author**: Will Handley • **Institution**: University of Cambridge
