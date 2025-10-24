import os ##
import logging
from time import time

import adafruit_dht
import argparse
from board import D4 as D4
import numpy as np
import redis
from scipy.io.wavfile import write
import sounddevice as sd
from torchaudio import transforms as T
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def get_logger():
    # instantiate logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # define handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    )

    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

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

class SensorManager:
    def __init__(self, logger):
        self.device = adafruit_dht.DHT11(D4)
        self.logger = logger

        self.logger.info("device is OK.")

    def read(self):
        return {
            "temperature": self.device.temperature,
            "humidity": self.device.humidity
        }

class CloudClient:
    def __init__(self, logger, host: str, port: int, user: str, password: str):
        self.client = redis.Redis(
            host=host,
            port=port,
            username=user,
            password=password
        )
        assert self.client.ping(), "Failed to connect to Redis"
        self.logger = logger

        self.logger.info("cloud is OK.")

    def store(self, key: str, value):
        pass
        # self.client.set(key, value)

class ModelManager:
    def __init__(self, logger):
        # self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.logger = logger

        self.logger.info("model is OK.")

class System:
    def __init__(self, args, logger):
        self.sensor = SensorManager(logger)
        self.cloud = CloudClient(logger, args.host, args.port, args.user, args.password)
        self.model = ModelManager(logger)
        # self.vui = VUI()
        self.state = 0
        self.logger = logger

        self.logger.info("system ready")

class VUI():
    def __init__(self, logger, system, device: int = 1, channels: int = 1, dtype = "int16", samplerate: int = 48_000):
        self.audio_buffer = list()
        self.channels     = channels
        self.device       = device 
        self.dtype        = dtype
        self.logger       = logger
        self.samplerate   = samplerate
        self.system       = system 

        self.logger.info("vui ready")

    def _audio_pipeline(self):
        # convert the recorded audio to PyTorch tensor of type float32
        audio_data = torch.tensor(self.audio_buffer, dtype=torch.float32)
        # change the data layout from channel-last to channel-first format
        swapped_audio_data = torch.swap(audio_data, 0, 1)
        # normalize the waveform values to the range [−1,1].
        normalized_audio_data = swapped_audio_data / torch.pow(2, 15) # 2^{B-1} where B is the bit depth
        # downsample the signal to 16kHz.
        transform = T.Resample(self.samplerate, 16_00)
        downsampled_audio_data = transform(normalized_audio_data)
        # remove the channel dimension
        return torch.squeeze(downsampled_audio_data)


    def callback(self, indata, frames, callback_time, status):
        self.audio_buffer.append(indata.copy())

    def record_audio(self):
        self.logger.info("recording")
        with sd.InputStream(
            device=self.device,
            channels=self.channels,
            dtype=self.dtype,
            samplerate=self.samplerate,
            callback=self.callback):
            while True:
                sd.sleep(1000)
                audio_data = np.concatenate(self.audio_buffer, axis=0 ,dtype=self.dtype)
                # fai tutto e pulisci
                current_filename = "%f.wav"%time()
                output_dir = "data/audio"
                os.makedirs(output_dir, exist_ok=True)
                audio_path = os.path.join(output_dir, current_filename)
                write(audio_path, self.samplerate, audio_data)
                self.audio_buffer.clear()
                break

if __name__ == "__main__":
    logger = get_logger()
    # Retrieve arguments from command line interface
    args = get_cli()
    # Initialize the system
    system = System(args, logger)
    # Configure the audio acquisition
    vui = VUI(logger, system)
    vui.record_audio()