"""
main.py
"""
from src.ec4ai.cli import get_cli
from src.ec4ai.cloud.cloud_inference_client import cloud_inference_client
from src.ec4ai.config import load_config
from src.ec4ai.edge.edge_inference import edge_inference
from src.ec4ai.sensors.mic_stream import start_recording
from src.ec4ai.utils.env_loader import load_env_file

import os

def main():
    
    args = get_cli()
    config = load_config(args.config)
    
    load_env_file(".env")
    API_URL = os.getenv("CLOUD_API_URL")
    
    audio_path = start_recording(args, config)

    print("Cloud Inference:\n")
    cloud_inference_client(audio_path, API_URL)
    
    print("Egde Inference:\n")
    edge_inference(audio_path)

if __name__  == "__main__":
    main()