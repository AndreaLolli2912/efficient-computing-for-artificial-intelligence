import argparse
import logging
import re
import time
import uuid

import adafruit_dht
from board import D4
import redis
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import sounddevice as sd

def get_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="Redis Cloud host."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Redis Cloud port."
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Redis Cloud username."
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Redis Cloud password."
    )
    return parser.parse_args()

def establish_cloud_connection(args):
    redis_client = redis.Redis(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password
    )

    assert redis_client.ping(), "Redis Cloud Connection Failed"

    return redis_client

def load_whisper_model(model_name='openai/whisper-tiny.en'):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    return model, processor

def callback(indata, frames, time, status):
    global system_state
    # Convert the recorded audio to a PyTorch tensor of type float32.
    tensor_float32 = torch.tensor(indata.copy(), dtype=torch.float32)
    # Change the data layout from channel-last to channel-first format.
    tensor_float32 = torch.swapaxes(tensor_float32, 0, 1)
    # Normalize the waveform values to the range [−1,1].
    waveform_norm = tensor_float32 / 2 ** (16 - 1)
    # Downsample the signal to 16kHz.
    waveform_16k = torchaudio.functional.resample(waveform_norm, SAMPLING_RATE, 16_000)
    # Remove the channel dimension.
    waveform_16k = torch.squeeze(waveform_16k)
    # Feed the resulting tensor to the Whisper pipeline.
    inputs = processor(waveform_16k, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features
    generated_ids = model.generate(input_features)
    # Transcribe the output, removing spaces and punctuation.
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    transcription = re.sub(r'[^a-z0-9\s]', '', transcription.strip().lower())
    # control logic
    if "up" in transcription and system_state == DISABLED:
        system_state = ENABLED
        logger.info("Voice command detected: ENABLE data collection")
    elif "stop" in transcription and system_state == ENABLED:
        system_state = DISABLED
        logger.info("Voice command detected: DISABLE data collection")

if __name__ == "__main__":
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S")

    logger = logging.getLogger(__name__)

    args = get_cli()

    # Initialize the DHT-11 sensor to collect temperature and humidity data.
    mac_address = hex(uuid.getnode())
    logger.info("Initializing DHT-11 sensor...")
    dht_device = adafruit_dht.DHT11(D4)
    
    # Establish a connection to the Redis Cloud database using the redis-py API.
    logger.info("Connecting to Redis...")
    redis_client = establish_cloud_connection(args)
    # Time series creation
    try:
        redis_client.ts().create(f"{mac_address}:temperature")
    except redis.ResponseError:
        pass # Time series already exists
    
    try:
        redis_client.ts().create(f"{mac_address}:humidity")
    except redis.ResponseError:
        pass # Time series already exists

    # Load the pretrained Whisper tiny model for voice command recognition.
    logger.info("Loading Whisper tiny model...")
    model, processor = load_whisper_model()

    # Set the system state to disabled (data collection off).
    ENABLED, DISABLED = 1, 0
    system_state = DISABLED

    # Configure the recording parameters
    CHANNELS = 1
    BIT_DEPTH = "int16"
    SAMPLING_RATE = 48_000

    # Implement command recognition.
    logger.info("Starting audio stream...")
    with sd.InputStream(
        samplerate=SAMPLING_RATE,
        blocksize=48_000, # calls 'callback function every 1s'
        device=1,
        channels=CHANNELS,
        dtype=BIT_DEPTH,
        callback=callback
    ):
        last_read_time = 0
        while True:
            if system_state == ENABLED:
                try:
                    timestamp = time.time()
                    timestamp_ms = int(timestamp * 1000) # Convert Unix time in milliseconds and cast it to integer

                    temperature = float(dht_device.temperature)
                    humidity    = float(dht_device.humidity)
                    logger.info(f"Reading: Temperature: {temperature} | Humidity: {humidity}")
                    
                    redis_client.ts().add(f"{mac_address}:temperature", timestamp_ms, temperature)
                    redis_client.ts().add(f"{mac_address}:humidity",    timestamp_ms, humidity   )
                    logger.info("Uploaded to Redis timestamp=%d", timestamp_ms)
                    
                    time.sleep(5)

                except Exception as e:
                    logger.warning("Sensor read failure")
                    dht_device.exit()
                    dht_device = adafruit_dht.DHT11(D4)
                    continue

            elif system_state == DISABLED:
                continue