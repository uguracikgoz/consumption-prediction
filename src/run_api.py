#!/usr/bin/env python3
"""
Run the Pod Prediction API
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Pod Prediction API")
    parser.add_argument("--init", action="store_true", help="Initialize models before starting API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the API to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser.parse_args()

def init_models():
    """Run the model initialization script"""
    print("Initializing models...")
    init_script = Path(__file__).parent / "init_model.py"
    result = subprocess.run([sys.executable, str(init_script)], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Model initialization failed:")
        print(result.stderr)
        return False
        
    print(result.stdout)
    return True

def run_api(host, port, reload):
    """Run the FastAPI application"""
    api_script = Path(__file__).parent / "src" / "api" / "main.py"
    cmd = [sys.executable, str(api_script)]
    
    # Set environment variables
    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(port)
    
    print(f"Starting API on {host}:{port}...")
    os.execve(sys.executable, cmd, env)

def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize models if requested
    if args.init:
        success = init_models()
        if not success:
            sys.exit(1)
    
    # Check if models exist
    model_dir = Path(__file__).parent / "src" / "model" / "saved_models"
    if not model_dir.exists() or not any(model_dir.glob("*_model.joblib")):
        print("No models found. Running model initialization...")
        success = init_models()
        if not success:
            sys.exit(1)
    
    # Run API
    run_api(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
