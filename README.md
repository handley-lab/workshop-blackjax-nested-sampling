# BlackJAX Nested Sampling Workshop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/handley-lab/workshop-blackjax-nested-sampling/blob/master/blackjax_nested_sampling_workshop.ipynb)

## GPU-Native Nested Sampling for Modern Simulation-Based Inference

Welcome to the hands-on workshop on nested sampling with BlackJAX! This repository contains materials for a 45-minute workshop delivered at the SBI Galaxy Evolution 2025 conference.

## 🎯 Workshop Overview

**Duration:** ~45 minutes  
**Level:** Intermediate (assumes familiarity with JAX and SBI concepts)  
**Platform:** Google Colab (zero installation required)

### What You'll Learn

- Why nested sampling is essential for most SBI methods
- How to use BlackJAX for GPU-native nested sampling
- Professional visualization with Anesthetic
- Performance comparison: nested sampling vs. ensemble samplers
- Hands-on experience with real examples

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above to run the workshop directly in your browser with zero setup.

### Option 2: Local Installation
```bash
git clone https://github.com/handley-lab/workshop-blackjax-nested-sampling.git
cd workshop-blackjax-nested-sampling
pip install blackjax[all] anesthetic emcee corner jupyter
jupyter notebook blackjax_nested_sampling_workshop.ipynb
```

## 📚 Workshop Content

### 1. Setup & Installation (5 min)
- Quick package installation
- JAX configuration for optimal performance

### 2. Theory: Why Nested Sampling for SBI? (5 min)
- Most SBI methods (NLE, NRE, NJE) require samplers
- BlackJAX advantages: GPU-native, open source, community-driven

### 3. Example 1: Linear Regression (15 min)
- Bayesian linear regression with nested sampling
- Model definition, prior specification, sampling
- Results analysis and interpretation

### 4. Visualization with Anesthetic (10 min)
- Professional corner plots
- Posterior predictive checks
- Publication-quality figures

### 5. Example 2: Multimodal Distribution (5 min)
- Demonstrates nested sampling's strength with complex posteriors
- Mixture of Gaussians example

### 6. Performance Comparison (10 min)
- BlackJAX nested sampling vs. emcee (AIES)
- Timing benchmarks and sample quality
- Model evidence as a bonus feature

### 7. Your Turn: Experiment! (Remaining time)
- Modify examples with your own problems
- Integration with JAX/SciML workflows
- Advanced BlackJAX features

## 🔧 Key Technologies

- **[BlackJAX](https://github.com/handley-lab/blackjax)**: GPU-native MCMC and nested sampling
- **[JAX](https://jax.readthedocs.io/)**: Automatic differentiation + JIT compilation
- **[Anesthetic](https://anesthetic.readthedocs.io/)**: Nested sampling visualization
- **[emcee](https://emcee.readthedocs.io/)**: Ensemble sampling for comparison

## 💡 Core Messages

### BlackJAX Advantages
1. **GPU-native**: Leverages modern HPC infrastructure
2. **Open source**: Community-driven development
3. **JAX integration**: Autodiff + JIT compilation
4. **Model evidence**: Bonus feature for model comparison

### When to Use Nested Sampling
- Model comparison (via evidence)
- Multimodal posteriors
- High-dimensional parameter spaces
- When you need robust, general-purpose sampling

## 🎓 Learning Outcomes

By the end of this workshop, you'll be able to:
- Set up and run BlackJAX nested sampling
- Interpret nested sampling results
- Create professional visualizations with Anesthetic
- Compare different sampling methods
- Apply nested sampling to your own SBI problems

## 📁 Repository Contents

- `blackjax_nested_sampling_workshop.ipynb` - Main workshop notebook
- `README.md` - This documentation  
- `CLAUDE.md` - Development instructions for Claude Code
- `pre-prompt.md` - Initial workshop requirements and context
- `prompt.md` - Detailed project specifications
- `history/` - Development history and conversation logs
- `.gitignore` - Git ignore patterns

## 📖 Further Reading

- [BlackJAX Documentation](https://github.com/handley-lab/blackjax)
- [Anesthetic Plotting Guide](https://anesthetic.readthedocs.io/en/latest/plotting.html)
- [JAX Scientific Computing](https://jax.readthedocs.io/en/latest/)
- [Nested Sampling Review](https://arxiv.org/abs/2205.15570)

## 🤝 Contributing

Found an issue or have suggestions? Please:
1. Open an issue on GitHub
2. Submit a pull request
3. Join the discussion in BlackJAX community

## 🙏 Acknowledgments

- **BlackJAX Team**: For the amazing GPU-native sampling library
- **SBI Galaxy Evolution 2025**: For hosting this workshop
- **JAX Team**: For the computational foundation
- **Anesthetic Developers**: For beautiful visualization tools

---

**Ready to explore GPU-native nested sampling? Let's dive in!** 🚀