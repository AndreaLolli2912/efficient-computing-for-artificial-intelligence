"""
main.py
"""
from src.ec4ai.cli import get_cli
from src.ec4ai.config import load_config
from src.ec4ai.sensors.mic_stream import start_recording

def main():
    args = get_cli()
    config = load_config(args.config)

    start_recording(args, config)

if __name__  == "__main__":
    main()