#!/bin/bash
# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Uninstall and reinstall critical packages cleanly
pip uninstall -y numpy scipy scikit-learn
pip install --no-cache-dir numpy scipy scikit-learn
