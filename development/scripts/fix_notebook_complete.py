#!/usr/bin/env python3
"""Fix notebook by adding execution_count fields and cell IDs."""

import json
import uuid

# Read the notebook
with open('workshop_nested_sampling.ipynb', 'r') as f:
    notebook = json.load(f)

# Add execution_count and cell IDs to all code cells
execution_count = 1
for cell in notebook['cells']:
    # Add cell ID if missing
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
    
    if cell['cell_type'] == 'code':
        cell['execution_count'] = execution_count
        if 'outputs' not in cell:
            cell['outputs'] = []  # Empty outputs for now
        execution_count += 1

# Write the fixed notebook
with open('workshop_nested_sampling.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Fixed notebook execution_count fields and cell IDs")