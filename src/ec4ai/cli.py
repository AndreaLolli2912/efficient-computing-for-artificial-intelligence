"""
cli.py
"""
import argparse

def get_cli():
    parser = argparse.ArgumentParser(
        description="Efficient Computing 4 Artificial Intelligence"
    )

    parser.add_argument(
        "--bit-depth",
        type=str,
        choices=["int16", "int32"],
        default="int16",
        help=(
            "Audio sample precision:\n"
            "  int16: 16-bit depth (2 bytes per sample, range: -32,768 to +32,767, ~96 dB dynamic range). "
            "  int32: 32-bit depth (4 bytes per sample, range: ±2.1e9, ~144 dB dynamic range). "
            "Default: int16."
        )
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        choices=[48_000, 44_100],
        default=48_000,
        help=(
            "Sampling frequency (samples per second):\n"
            "  44,100 Hz\n"
            "  48,000 Hz\n"
            "Higher rates increase accuracy but also CPU and storage use.\n"
            "Default: 48,000 Hz."
        )
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1,
        help=(
            "Recording duration in seconds (int)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml"
    )

    return parser.parse_args()
