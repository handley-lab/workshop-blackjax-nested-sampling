#!/usr/bin/env python3
"""Execute notebook with proper matplotlib inline display."""

import os
import subprocess
import sys

# Set matplotlib backend environment variable
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'

# Execute the notebook
cmd = [
    sys.executable, '-m', 'nbconvert',
    '--to', 'notebook',
    '--execute',
    '--inplace',
    '--ExecutePreprocessor.kernel_name=python3',
    'workshop_nested_sampling.ipynb'
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/will/documents/workshop-blackjax-nested-sampling')
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("Return code:", result.returncode)
except Exception as e:
    print(f"Error: {e}")