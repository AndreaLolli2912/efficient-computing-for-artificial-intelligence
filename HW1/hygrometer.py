import argparse
import hex
import re
import uuid

import adafruit_dht
from board import D4 as D4
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
        default="redis-15750.c135.eu-central-1-1.ec2.redns.redis-cloud.com",
        help="Redis Cloud host."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=15750,
        help="Redis Cloud port."
    )
    parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="Redis Cloud username."
    )
    parser.add_argument(
        "--password",
        type=str,
        default="r35F7Gez05k66A86KA9JcSfqdZL9ekrG",
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
    print("input_features.shape", input_features.shape)
    generated_ids = model.generate(input_features)
    print("generated_ids.shape", generated_ids.shape)
    # Transcribe the output, removing spaces and punctuation.
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    transcription = re.sub(r'[^a-z0-9\s]', '', transcription.strip().lower())
    print("transcription", transcription)
    # control logic
    if transcription == "up":
        system_state = ENABLED
        print("up")
    elif transcription == "stop":
        system_state = DISABLED
        print("down")

if __name__ == "__main__":

    args = get_cli()

    # Initialize the DHT-11 sensor to collect temperature and humidity data.
    dht_device = adafruit_dht.DHT11(D4)

    # Establish a connection to the Redis Cloud database using the redis-py API.
    redis_client = establish_cloud_connection(args)

    # Load the pretrained Whisper tiny model for voice command recognition.
    model, processor = load_whisper_model()

    # Set the system state to disabled (data collection off).
    ENABLED, DISABLED = 1, 0
    system_state = DISABLED

    # Configure the recording parameters
    CHANNELS = 1
    BIT_DEPTH = "int16"
    SAMPLING_RATE = 48_000

    # Implement command recognition.
    with sd.InputStream(
        samplerate=SAMPLING_RATE,
        blocksize=48_000,
        device=1,
        channels=CHANNELS,
        dtype=BIT_DEPTH,
        callback=callback
    ):
        while True:
            mac_address = hex(uuid.getnode())
            if system_state == ENABLED:
                try:
                    timestamp = time.time()
                    timestamp = int(timestamp * 1000) # Convert Unix time in milliseconds and cast it to integer

                    temperature = dht_device.temperature
                    humidity = dht_device.humidity

                    try:
                        redis_client.ts().create('temperature')
                    except redis.ResponseError:
                        pass
                    
                    try:
                        redis_client.ts().create("humidity")
                    except redis.ResponseError:
                        pass
                    
                    redis_client.ts().add("temperatue", timestamp, temperature)
                    redis_client.ts().add("humidity",   timestamp, humidity   )
                    
                    time.sleep(5)
                except:
                    dht_device.exit()
                    dht_device = adafruit_dht.DHT11(D4)
            elif system_state == DISABLED:
                continue