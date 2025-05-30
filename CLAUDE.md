# CLAUDE.md - Workshop Development Template

This file provides guidance to Claude Code (claude.ai/code) when developing educational workshops and tutorials.

## Workshop Development Best Practices

### Recommended Workflow

1. **Start with a Python Script** (`workshop_name.py`)
   - Develop the complete workshop as a flat Python script first
   - Use appropriate comment syntax for py2nb conversion (see below)
   - Test all code blocks end-to-end before notebook conversion
   - This allows for easier debugging and iterative development

2. **Convert to Notebook** using `py2nb`
   - Use the enhanced py2nb script to convert script to notebook
   - Use `--execute` option to create both clean and executed versions
   - Maintains proper cell structure and metadata

3. **Post-process Notebooks**
   - Validate notebook JSON structure
   - Ensure proper execution counts and cell metadata
   - Test notebook execution in target environments (Jupyter, Colab)

### Comment Syntax for py2nb

```python
#| # Title - Markdown Header
#| This creates a markdown cell with content
#| 
#| - Bullet points work
#| - Math: $E = mc^2$

# Regular Python code goes here
import numpy as np

#- # This starts a new code cell

more_code = "in new cell"

#! # This creates a command cell (new syntax)
#! pip install package_name
#! pip install another_package

# Continue with regular code
```

**Note**: The `#!` command syntax supports any shell commands, not just pip installs. This provides flexibility for workshop setup, data downloads, system configuration, and more.

### File Structure for Workshops

```
workshop_name/
├── CLAUDE.md                          # Project-specific instructions
├── workshop_name.py                   # Main script (development)
├── workshop_name.ipynb                # Clean interactive notebook  
├── workshop_name_executed.ipynb       # Pre-executed with outputs (py2nb --execute)
├── README.md                          # User documentation
└── prompt-materials/                  # Reference materials
    ├── dependency_docs/
    ├── example_notebooks/
    └── reference_implementations/
```

## Educational Workshop Architecture

### Core Design Principles

1. **Modular Structure**: Break content into logical parts (15-20 min each)
2. **Progressive Complexity**: Start simple, build to advanced concepts
3. **Executable Examples**: Every concept demonstrated with working code
4. **Multiple Delivery Modes**: Support different time allocations
5. **Platform Compatibility**: Work in Jupyter, Colab, and local environments

### Dependency Management

- **Core Dependencies**: Install minimal requirements at start
- **Advanced Dependencies**: Install when needed using `#!` command syntax
- **Platform Compatibility**: Prefer packages available in Colab
- **Version Pinning**: Specify versions for reproducibility when needed

### Content Organization

```python
#| # Workshop Title
#| 
#| Brief description and learning objectives
#| 
#| 📖 **Essential Reading**: Link to authoritative references

#! pip install core_dependency1 core_dependency2

import core_dependency1
import standard_libraries

#| ## Part 1: Core Concepts (20 minutes)
#| Essential material that everyone should complete

# Core implementation here

#| ## Part 2: Intermediate Applications (30 minutes) 
#| Building on Part 1 with real examples

# More code here

#- # New cell for complex example

example_code()

#| ## Part 3: Advanced Extensions (Optional - 30+ minutes)
#| Research-level techniques and integrations

#! pip install advanced_dependency

# Advanced implementations
```

## Key Dependencies & Frameworks

### Scientific Computing Stack
- **JAX**: Automatic differentiation and JIT compilation
- **NumPy/SciPy**: Core scientific computing foundations
- **Matplotlib**: Plotting and visualization

### Specialized Libraries (Install When Needed)
- **Optax**: Gradient-based optimization (`#!` command before Part 5)
- **Flax**: Neural networks (`#!` command before Part 6)
- **Domain-specific packages**: Install in relevant sections

## Validation and Testing

### Pre-Deployment Checklist

1. **Script Development**:
   - [ ] All code blocks execute successfully
   - [ ] Examples produce expected outputs
   - [ ] Error handling for common issues
   - [ ] Timing estimates for each section

2. **Notebook Conversion**:
   - [ ] py2nb conversion successful
   - [ ] JSON validation passes
   - [ ] All cells have proper metadata
   - [ ] Install blocks in correct locations

3. **Platform Testing**:
   - [ ] Jupyter Notebook execution
   - [ ] Google Colab compatibility  
   - [ ] Dependency installation works
   - [ ] All outputs render correctly

4. **Documentation**:
   - [ ] README with clear instructions
   - [ ] Learning objectives stated
   - [ ] Time estimates provided
   - [ ] Prerequisites listed

## Common Patterns

### Command Blocks
```python
#! # Core Dependencies (Workshop Start)
#! pip install main_package
#! pip install visualization_package

#! # Advanced Dependencies (Part N)
#! pip install advanced_package
```

### Progress Indicators
```python
#| ### Step N.M: Descriptive Title
#| 
#| Brief explanation of what this step accomplishes
#| and why it's important for the overall workflow.

# Implementation here
print("✓ Step completed successfully!")
```

### Educational Scaffolding
```python
#| 🎯 **Learning Objective**: By the end of this section, you will understand...
#| 
#| 📋 **Prerequisites**: This section assumes familiarity with...
#| 
#| ⏱️ **Estimated Time**: 15-20 minutes

# Teaching code here with extensive comments
```

## Anti-Patterns to Avoid

1. **Monolithic Code Blocks**: Break large implementations into digestible pieces
2. **Missing Installation Blocks**: Always specify when dependencies are needed
3. **Untested Examples**: Every code block should be verified to work
4. **Platform-Specific Code**: Avoid assumptions about local vs. cloud environments
5. **Missing Learning Context**: Always explain why each step matters

## Success Metrics

A successful workshop should:
- Execute completely in target environments (Jupyter, Colab)
- Provide clear learning progression from basic to advanced
- Include sufficient explanation for self-guided learning
- Offer modular sections for different time constraints
- Demonstrate best practices in the domain
- Include proper attribution and references

## Troubleshooting Common Issues

### Notebook Validation Errors
- Ensure markdown cells don't have `outputs` fields
- Check that `execution_count` is properly set (null for markdown, int/null for code)
- Validate JSON structure with `python -c "import json; json.load(open('notebook.ipynb'))"`

### Dependency Issues
- Test installations in fresh environments
- Use `#!` command blocks to install dependencies when needed
- Provide fallback instructions for manual installation

### Platform Compatibility
- Test in both Jupyter and Colab environments
- Avoid file system dependencies that don't work in cloud environments
- Provide alternative data sources when needed

---

**Workshop Development**: Generated with [Claude Code](https://claude.ai/code) • **Template Version**: 1.0