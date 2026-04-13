#!/usr/bin/env python3
"""Entry point for the OpenAI-compatible proxy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.openai_compat import create_proxy_app
import uvicorn
uvicorn.run(create_proxy_app(), host="0.0.0.0", port=8421)
