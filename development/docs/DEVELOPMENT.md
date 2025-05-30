# Development Tools and Scripts

This document describes the helper scripts created during workshop development to maintain the dual notebook setup and ensure proper GitHub display.

## Overview

The workshop maintains two notebook versions:
1. **Clean Interactive** (`workshop_nested_sampling.ipynb`) - For Google Colab
2. **Pre-executed Preview** (`workshop_nested_sampling_executed.ipynb`) - For GitHub browsing

## Helper Scripts

### `fix_notebook_complete.py`

**Purpose**: Fixes notebook validation issues for GitHub display by adding required metadata.

**What it does**:
- Adds `execution_count` fields to all code cells (required by GitHub)
- Adds unique `id` fields to all cells (prevents validation warnings)
- Ensures notebooks pass GitHub's validation requirements

**Usage**:
```bash
python fix_notebook_complete.py
```

**When to use**: After any notebook edits or before committing notebooks to fix validation errors.

### `execute_notebook.py`

**Purpose**: Executes notebooks with proper matplotlib backend for embedded plot generation.

**What it does**:
- Sets `MPLBACKEND=module://matplotlib_inline.backend_inline` for proper plot embedding
- Executes notebook using jupyter nbconvert
- Generates executed notebook with embedded PNG plots for GitHub display

**Usage**:
```bash
# Must be run from within activated virtual environment
source workshop_env/bin/activate
python execute_notebook.py
```

**When to use**: To create the pre-executed notebook version with embedded plots for GitHub preview.

## Development Workflow

### 1. Source of Truth
- **Edit** `workshop_nested_sampling.py` as the primary source
- Use py2nb to convert to clean notebook: `python prompt-materials/py2nb/py2nb workshop_nested_sampling.py > workshop_nested_sampling.ipynb`

### 2. Clean Notebook Maintenance
Always ensure the clean notebook has:
- Executable `!pip install` cell as second cell
- Installation note markdown cell
- `%matplotlib inline` in imports
- Combined plotting cells (no split creation/plotting)

### 3. Executed Notebook Generation
```bash
# Fix clean notebook structure
python fix_notebook_complete.py

# Execute with embedded plots
source workshop_env/bin/activate
python execute_notebook.py

# Copy to executed version
cp workshop_nested_sampling.ipynb workshop_nested_sampling_executed.ipynb

# Regenerate clean version
python prompt-materials/py2nb/py2nb workshop_nested_sampling.py > workshop_nested_sampling.ipynb
python fix_notebook_complete.py
```

## Common Issues and Solutions

### Problem: "Invalid Notebook 'execution_count' is a required property"
**Solution**: Run `python fix_notebook_complete.py`

### Problem: Plots don't embed in executed notebook
**Solution**: Use `execute_notebook.py` instead of direct jupyter execution

### Problem: Split visualization cells show empty plots
**Solution**: Combine figure creation and plotting in single cells

### Problem: Colab can't install packages
**Solution**: Ensure clean notebook has executable `!pip install` cell

## Virtual Environment Setup

```bash
python -m venv workshop_env
source workshop_env/bin/activate
pip install git+https://github.com/handley-lab/blackjax anesthetic tqdm jupyter matplotlib-inline
```

## Quality Checklist

Before committing:
- [ ] Clean notebook has install cell and runs in Colab
- [ ] Executed notebook displays all plots on GitHub
- [ ] Both notebooks validate without errors
- [ ] README.md links point to correct versions
- [ ] All helper scripts documented and committed

## Script Implementation Details

### fix_notebook_complete.py Implementation
```python
import json
import uuid

with open('workshop_nested_sampling.ipynb', 'r') as f:
    notebook = json.load(f)

execution_count = 1
for cell in notebook['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
    if cell['cell_type'] == 'code':
        cell['execution_count'] = execution_count
        execution_count += 1

with open('workshop_nested_sampling.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
```

### execute_notebook.py Implementation  
```python
import os
import subprocess
import sys

os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'

cmd = [
    sys.executable, '-m', 'nbconvert',
    '--to', 'notebook', '--execute', '--inplace',
    '--ExecutePreprocessor.kernel_name=python3',
    'workshop_nested_sampling.ipynb'
]

subprocess.run(cmd, cwd='/path/to/workshop')
```