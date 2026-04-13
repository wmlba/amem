#!/usr/bin/env python3
"""Entry point for the API server."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.app import create_app
import uvicorn
uvicorn.run(create_app(), host="0.0.0.0", port=8420)
