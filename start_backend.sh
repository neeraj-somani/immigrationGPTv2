#!/bin/bash
echo "Starting ImmigrationGPT Backend..."
cd backend
echo "Installing/updating Python dependencies..."
pip install -r requirements.txt
echo "Starting FastAPI server..."
python main.py
