import argparse
import logging
import time
import uuid
import os
import zipfile
import numpy as np

import adafruit_dht
from board import D4
import redis
import torch, torchaudio
import sounddevice as sd

from onnxruntime import InferenceSession


def get_cli()->argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        parser (argparse.Namespace): The parsed arguments."""
    
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
        help="Redis Cloud username.",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Redis Cloud password."
    )
    return parser.parse_args()


def establish_cloud_connection(args):
    """Establish a connection to Redis Cloud.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    
    Raises:
        AssertionError: If the connection to Redis Cloud fails.
        
    Returns:
        redis_client (redis.Redis): The Redis client instance.
    """
    redis_client = redis.Redis(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password
    )

    assert redis_client.ping(), "Redis Cloud Connection Failed"

    return redis_client


def loadFrontendAndModel(frontendPath:str, modelPath:str):
    """Load the custom Key Word Spotting model and the expected frontend.
    
    Args:
        frontendPath (str): The path to the frontend model.
        modelPath (str): The path to the custom KWS model.
        
    Returns:
        frontend (InferenceSession): The loaded frontend model.
        model (InferenceSession): The loaded custom KWS model.
    """
    if not os.path.exists(frontendPath):
        raise FileNotFoundError(f"Frontend model not found at {frontendPath}")
    
    if not  os.path.exists(modelPath):
        raise FileNotFoundError(f"Custom KWS model not found at {modelPath}")
    
    if modelPath.lower().endswith(".zip"):
        with zipfile.ZipFile(modelPath, "r") as z:
            modelPath = z.namelist()[0]
            z.extract(modelPath, ".")   
    
    return InferenceSession(frontendPath), InferenceSession(modelPath)


def callback(indata, frames, time, status, EnableThreshold:float=0.999):
    global system_state
    
    # Convert the recorded audio to a PyTorch tensor of type float32.
    tensor_float32 = torch.tensor(indata.copy(), dtype=torch.float32)
    
    # Change the data layout from channel-last to channel-first format.
    tensor_float32 = torch.swapaxes(tensor_float32, 0, 1)
    
    # Normalize the waveform values to the range [−1,1].
    waveform_norm = tensor_float32 / 32_768.0 #   32_768 = 2 ** (16 - 1)
    
    # Downsample the signal to 16kHz.
    waveform_16k = torchaudio.functional.resample(waveform_norm, SAMPLING_RATE, 16_000)
    
    # Feed the preprocessed audio to the frontend and then to the model and get the prediction
    inputs = frontend.run(None, {"input": np.expand_dims(waveform_16k.numpy(), axis=0)})[0]
    outputs = model.run(None, {"input": inputs})[0][0]
    
    # Post-process the model outputs to get the predicted class and its probability
    exp = np.exp(outputs)
    pStop, pUp = exp / np.sum(exp) 
    pred = np.argmax(outputs).item()
    
    # control logic
    if pred and pUp >= EnableThreshold and system_state == DISABLED:
        system_state = ENABLED
        logger.info("Voice command detected: ENABLE data collection")
        
    elif not pred and pStop >= EnableThreshold and system_state == ENABLED:
        system_state = DISABLED
        logger.info("Voice command detected: DISABLE data collection")
        

if __name__ == "__main__":
    FRONTEND_PATH = './HW2/model/Group9_frontend.onnx'
    MODEL_PATH = './HW2/model/Group9_model.onnx.zip'
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Create a logger instance
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
    frontend, model = loadFrontendAndModel(FRONTEND_PATH, MODEL_PATH)

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
        # Main loop to read sensor data and upload to Redis Cloud.
        while True:
            # Read sensor data and upload them only if the system state is enabled.
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