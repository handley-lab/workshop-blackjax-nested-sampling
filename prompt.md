**Project Goal:** Develop a 1-hour short talk (approx. 10-15 minutes, ~5 slides) and an accompanying hands-on workshop (approx. 45-50 minutes) on my new nested sampling algorithm in BlackJAX.

**Your Role:** You are an AI assistant tasked with helping me conceptualize, outline, and draft content for this talk and workshop. You should synthesize information from the provided materials, propose structures, and help generate content. We will build a Git repository for the workshop materials together.

**Event Context:**
*   **Workshop I'm Attending:** SBI Galev 2025 (Details: https://sbi-galev.github.io/2025/)
*   **My Session Slot:** 1 hour total.

**Key Input Materials & How to Use Them:**

You will need to synthesize information from the following `prompt-materials/` and external links. Assume you can access and process the content within these resources based on my descriptions.

1.  **`prompt-materials/intro_jax_sciml/`:**
    *   **Content:** Workshop on JAX for scientific machine learning (by Viraj Pandya, Wednesday).
    *   **Your Task:** Refer to this directly for all relevant content on JAX and scientific machine learning. Use it to understand what foundational JAX knowledge the audience might have gained.

2.  **`prompt-materials/ltu-ili/`:**
    *   **Content:** Workshop on implicit likelihood inference in the "learning the universe" framework (by Matt Ho, Tuesday).
    *   **Your Task:** Use this to understand the broader conference content and identify potential example problem domains that resonate with the audience.

3.  **`prompt-materials/sbi-talk/`:**
    *   **Content:** Introduction to simulation-based inference (talk by Maximilian von Wietersheim-Kramsta, Tuesday), including LaTeX and figures.
    *   **Your Task:** Refer to this for material already covered on SBI, ensuring my talk builds upon it without excessive repetition.

4.  **`prompt-materials/talks/` (Git Repository - My Previous Talks):**
    *   **Content:** A Git repository of my past talks. The `README.md` lists available talk branches.
    *   **Your Task:** I need you to conceptually "explore" these branches (as if you were checking them out and reading `will_handley.tex` for each) to identify relevant content for the new talk. The goal is to synthesize and adapt existing material onto a new branch `bristol_2025`.
        *   `cosmoverse_2025`: GPU machinery & LLMs.
        *   `vietnam_2025`: Most up-to-date views on SBI machinery.
        *   `imperial_2024`: Linear simulation-based inference.
        *   `india_2024`: Polyswyft.
        *   `sydney_2024`: Bayesian inference and nested sampling.
        *   `phystat_2024`: More controversial views on SBI and density estimation.
    *   **Focus:** Identify slides/concepts from these past talks that align with the core messages for the new talk (detailed below).

5.  **`prompt-materials/blackjax/` (BlackJAX Repository):**
    *   **Content:** The BlackJAX repository.
    *   **Your Task:** Focus on these key areas to inform the workshop content and examples:
        *   Nested sampling source files: `blackjax/ns/`, `blackjax/mcmc/ss/`
        *   Example script: `line.py` (fitting a straight line with BlackJAX NS).
        *   Example script: `nsmcmc.py` (implementing a custom NS algorithm).
        *   Example script: `docs/examples/nested_sampling.py` (advanced usage patterns).

6.  **`prompt-materials/anesthetic/` (Anesthetic Library Documentation):**
    *   **Content:** Complete anesthetic library documentation and examples.
    *   **Your Task:** Use these materials to understand proper nested sampling post-processing:
        *   `docs/source/plotting.rst`: Comprehensive plotting guide with API examples
        *   `anesthetic/examples/perfect_ns.py`: Perfect nested sampling generators and patterns
        *   `tests/test_examples.py`: Real-world usage examples and API patterns
        *   Key patterns: `NestedSamples` class, `plot_2d()` methods, evidence computation

**Deliverables We Will Create Together:**

1.  **Short Talk (~5 slides, 10-15 minutes):**
    *   Concise, high-impact introduction to nested sampling in BlackJAX.
2.  **Workshop (Python Notebook & README, 45-50 minutes):**
    *   Hands-on, runnable in Google Colab (minimize installation friction).
    *   Clear `README.md` for the workshop repository.
3.  **Git Repository:** For all workshop materials (notebook, README, any data files).

**Key Links for Reference:**
*   **BlackJAX:** https://github.com/handley-lab/blackjax
*   **Anesthetic (for plotting NS results):** https://anesthetic.readthedocs.io/en/latest/plotting.html

**Core Ideas to Convey in the Talk:**

*   Most SBI methods (NLE, NRE, NJE, etc.), excluding NPE, require a sampler.
*   Nested sampling is a general-purpose sampler, but existing implementations are often legacy Fortran (MultiNest, PolyChord) or slower Python (dynesty, ultranest, nautilus).
*   **Our Contribution:** BlackJAX offers a GPU-native, open-source nested sampling implementation, aiming for community ownership via the BlackJAX repository.
*   The future of HPC is GPU-driven, heavily influenced by machine learning.
*   Alternative tools (e.g., harmonic, floz) derive strength from GPUs; a good workshop exercise could be comparing BlackJAX NS performance against these.
*   **JAX Explained Simply:**
    *   Automatic differentiation.
    *   Just-in-time (JIT) compilation (especially for GPUs).
    *   These are distinct but equally powerful features.
*   The rise of AI in scientific research (this collaborative talk creation is an example!).

**Core Ideas/Tasks for the Workshop:**

*   Demonstrate how to run nested sampling using BlackJAX.
*   Show how to use `anesthetic` for visualizing results and proper post-processing.
*   Guide users through comparing Nested Sampling (NS) with NUTS and Affine Invariant Ensemble Sampler (AIES) on a simple problem.
*   Educational progression: Line fitting → 2D Gaussian parameter inference → Performance comparison
*   Encourage attendees to try the BlackJAX NS implementation with their own JAX code, potentially building on Viraj Pandya's JAX/SciML workshop material.

**Workshop Implementation Notes (from Development):**

*   **Performance Configuration**: Use 100 live points and num_delete=50 for optimal workshop timing
*   **Anesthetic Integration**: Use `NestedSamples` class with `plot_2d()` methods for proper visualization
*   **Error Handling**: Include proper covariance matrix validation and parameter transform examples
*   **API Patterns**: Demonstrate `samples.logZ()` and `samples.logZ(nsamples).std()` for evidence computation
*   **Transform Functions**: Show proper arctanh/tanh transforms for constrained parameters (e.g., correlation coefficient)
*   **Educational Structure**: Start simple (line fitting) then build complexity (multivariate parameter inference)

**Initial Request & How to Proceed:**

1.  **Propose an Outline:** Based on all the above, please first propose a high-level outline for:
    *   The ~5-slide talk structure.
    *   The sections/flow of the Python workshop notebook.
2.  **Identify Key Content:** From my past talks (`prompt-materials/talks/`), which specific talks/slides seem most promising to adapt for the `bristol_2025` talk?
3.  **Suggest Workshop Examples:** What simple, illustrative examples could we use in the Colab notebook, keeping in mind the audience and potential integration with Viraj's JAX/SciML content? The `line.py` example is a good starting point, but what else?

I am very open to your ideas on content, structure, and examples for both the talk and the workshop. Let's start with the outlines and content identification.
