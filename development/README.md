# Development Materials

This directory contains development materials and reference content used during the creation of the BlackJAX nested sampling workshop.

## Directory Structure

### `docs/`
- **DEVELOPMENT.md** - Development notes and technical decisions
- **prompt.md** - Main development prompts and specifications 
- **pre-prompt.md** - Initial workshop concept and requirements

### `history/`
- **Conversation logs** - Complete development history and iterations
- **Progressive refinements** - Shows evolution of workshop content
- **Decision tracking** - Rationale for technical choices

### `reference-materials/`
- **blackjax/** - BlackJAX library examples and documentation
- **anesthetic/** - Anesthetic visualization library materials
- **py2nb/** - Enhanced py2nb tool development (now on PyPI)
- **talks/** - Reference presentations and prior art
- **sbi-talk/**, **ltu-ili/**, etc. - Related workshop materials

### `scripts/`
- **execute_notebook.py** - Helper for notebook execution
- **fix_notebook.py** - Notebook formatting utilities
- **fix_notebook_complete.py** - Enhanced notebook repair tools

## Usage Notes

- **Reference materials** are excluded from git tracking (see .gitignore)
- **Scripts** are development utilities, not part of main workshop
- **History** provides complete development audit trail
- **Documentation** explains technical decisions and methodology

## Workshop Development Methodology

The workshop was developed using a **script-first approach**:

1. **Python script development** (`workshop_nested_sampling.py`)
2. **py2nb conversion** to create clean notebooks
3. **Execution testing** to generate output versions
4. **Iterative refinement** based on testing and feedback

This methodology is documented in the template files in the parent directory and proven effective for creating high-quality educational materials.