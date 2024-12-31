#!/bin/bash

set -e
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
pip uninstall transformers
pip install transformers[torch]
pip install protobuf==3.20.*
