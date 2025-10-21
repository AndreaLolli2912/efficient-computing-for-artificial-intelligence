"""
main.py
"""
from src.ec4ai.cli import get_cli
from src.ec4ai.config import load_config
from src.ec4ai.sensors.mic_stream import start_recording
from src.ec4ai.cloud.cloud_inference_client import cloud_inference_client
def main():
    args = get_cli()
    config = load_config(args.config)

    file_path = start_recording(args, config)
    cloud_inference_client(file_path)
if __name__  == "__main__":
    main()