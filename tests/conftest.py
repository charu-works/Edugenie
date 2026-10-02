import os
import sys
from pathlib import Path

os.environ["PRELOAD_LOCAL_MODEL"] = "0"   # don't load the 3 GB model during tests
os.environ["EXPLAIN_BACKEND"] = "gemini"  # tests never touch torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))