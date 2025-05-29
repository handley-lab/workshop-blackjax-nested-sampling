#!/usr/bin/env python3
"""Fix notebook by adding execution_count fields to code cells."""

import json

# Read the notebook
with open('workshop_nested_sampling.ipynb', 'r') as f:
    notebook = json.load(f)

# Add execution_count to all code cells
execution_count = 1
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = execution_count
        cell['outputs'] = []  # Empty outputs for now
        execution_count += 1

# Write the fixed notebook
with open('workshop_nested_sampling.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Fixed notebook execution_count fields")