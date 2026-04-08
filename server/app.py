# server/app.py
from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import app

# This file is required by Scaler's checker