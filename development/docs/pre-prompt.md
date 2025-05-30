I'm writing a short talk and workshop on my new nested sampling algorithm in blackjax.

The workshop I'm attending is detailed here:
https://sbi-galev.github.io/2025/

Key material:

prompt-materials/intro_jax_sciml/: this is a workshop on using jax for scientific machine learning, which is highly relevant to my talk, and you should refer to directly for any and all relevant content on jax and scientific machine learning. Workshop given on Wednesday by Viraj Pandya

prompt-materials/ltu-ili/: This is a workshop on using implicit likelihood inference in the learning the universe framework, downloaded from google drive, and gives an idea of the content of the conference, and possibly some examples we can run. Workshop given on Tuesday by Matt Ho

prompt-materials/sbi-talk/: This was an excellent introduction to simulation based inference in this talk, which has the latex and figures here. You should refer to it throughout for material which has been covered in the workshop on simulation based inference. Talk on Tuesday by Maximilian von Wietersheim-Kramsta

prompt-materials/talks: This is my back catalogue of my previous talks as a git repository. The README.md gives a back list with the branches of the talks available, which you should be prepared to explore directly in order to construct my short talk. You should check out the branch, view will_handley.tex, assess which content is relevant for this talk, and bring it together onto the new branch bristol_2025
  - cosmoverse_2025:
    - talk on GPU machinery & LLMs
  - vietnam_2025:
    - most up-to-date views on SBI machinery
  - imperial_2024:
    - talk on linear simulation-based inference
  - india_2024:
    - talk on polyswyft
  - sydney_2024:
    - talk on bayesian inference and nested sampling
  - phystat_2024:
    - my more controversial views on SBI and density estimation.

prompt-materials/blackjax/: This is the blackjax repository. The relevant files are:
- source files for nested sampling
  - blackjax/ns/
  - blackjax/mcmc/ss
- line.py: an example script for fitting a straight line to data using blackjax nested sampling
- nsmcmc.py: an example script for how to implement ones own nested sampling algorithm

I have 1h, of which 10-15 minutes should be about 5 slides, and the rest should be a workshop on nested sampling in blackjax, which encourages people to try out our new nested sampling implementation with their jax code, possibly on code that was provided yesterday by viraj

We are going to build a git repository for the workshop, which should have a readme and a python notebook, which we will work on together.

key links:
- blackjax: https://github.com/handley-lab/blackjax
- anesthetic: https://anesthetic.readthedocs.io/en/latest/plotting.html


- Core ideas to get across in the talk:
  - Other than NPE, all other methods (NLE, NRE, NJE etc) all need a sampler. 
  - nested sampling is a general purpose sampler, but either wrapped up in legacy fortran codes (MultiNest, PolyChord), or in slow python codes (dynesty, ultranest, nautilus)
  - this code is gpu native, open source, and an attempt to dissociate the authors via the blackjax repo so that it can be a community project.
  - Emphasise that the future is GPU, whether we like it or not because all future HPC will be heavily weighted to that via machine learning
  - emphasise that alternative tools like harmonic or floz gather their strength from the gpu rather than the sampling method, so a good exercise this workshop would be to compare the performance of nested sampling in blackjax with these other tools.
  - jax does two things:
    - it does automatic differentiation
    - it does just-in-time compilation, for GPUs
    - people often conflate these two things, but they are separate, and glorious.
  - I want to also get across the upcoming surge of ai methods for doing scientific research
    - the writing of this talk in one day is a testament to that 
- Core ideas to get across in the workshop:
  - how to run nested sampling
  - how to use anesthetic
  - compare NS to AIES (affine invariant ensemble sampler)

The workshop should be in python notebook format, and ideally run in google colab, so that people can run it without needing to install anything.

Very free to hear your ideas on what things could be included in the workshop.
