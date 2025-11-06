import collections
import logging
import queue
import re
import threading
import time

import adafruit_dht
import argparse
from board import D4 as D4
import numpy as np
import redis
from scipy.io.wavfile import write
import sounddevice as sd

import torch
from torchaudio import transforms as T
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# audio global vars
BIT_DEPTH = "int16"
CHANNELS = 1
DEVICE = 1
SAMPLERATE = 48_000

# system state vars
ENABLED, DISABLED = 1, 0

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

# Voice User Interface Components
class AudioCapture:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self._ring = collections.deque(maxlen=self.samplerate)
    
    def callback(self, indata, frames, callback_time, status):
        self._ring.extend(indata[:, 0].copy())

    def get_last_second(self):
        if len(self._ring) == self.samplerate:
            return np.array(self._ring)
        return None

class AudioPreprocessor:
    def __init__(self, sr_in, sr_out):
        self.resample = T.Resample(orig_freq=sr_in, new_freq=sr_out)
    
    def __call__(self, arr_int16):
        """arr_int16 : np.array (48_000, ), dtype = 'int16' """
        # Convert the recorded audio to a PyTorch tensor of type float32.
        x = torch.tensor(arr_int16, dtype=torch.float32)
        # print(x.min(), x.max(), x.mean())
        # Change the data layout from channel-last to channel-first format.
        # Normalize the waveform values to the range [−1,1].
        x = x / 32_768.0
        # print(x.min(), x.max(), x.mean())
        # Downsample the signal to 16kHz.
        x_16k = self.resample(x)
        # Remove the channel dimension.
        return x_16k

class WhisperModel:
    def __init__(self, logger):
        self.logger    = logger
        self.model     = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

        self.logger.info("whisper model loaded")

    def __call__(self, x_16):
        "x_16: torch.float32 at 16 kHz"
        inputs = self.processor(x_16, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features
        generated_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return transcription

class CommandRecognizer:

    def sanitize(self, text):
        return re.sub(r'[^a-z0-9\s]', '', text.strip().lower())

    def detect(self, text_clean):
        """Returns 'up', 'stop', or None."""
        if "up" in text_clean:
            return "up"
        elif "stop" in text_clean:
            return "stop"
        return None

class VUI:
    def __init__(self, logger, on_command, **vui_kwargs):
        self.logger     = logger
        self.on_command = on_command # callback for command events from system class

        self.channels   = vui_kwargs["channels"]
        self.device     = vui_kwargs["device"]        
        self.dtype      = vui_kwargs["dtype"]
        self.samplerate = vui_kwargs["samplerate"]

        # inject components
        self.asr          = WhisperModel(self.logger)
        self.capture      = AudioCapture(self.samplerate)
        self.preprocessor = AudioPreprocessor(sr_in=self.samplerate, sr_out=16_000)
        self.recog        = CommandRecognizer()

        # queue for 1s audio chunks
        self._audio_queue = queue.Queue(maxsize=2)

        # thread controll
        self._stop = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop) #, daemon=True
        self._process_thread = threading.Thread(target=self._process_loop) #, daemon=True

        self.logger.info("vui ready")
    
    def start(self):
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self):
        self._stop.set()
        self._capture_thread.join(timeout=2.0)
        self._process_thread.join(timeout=2.0)

    def _capture_loop(self):
        self.logger.info("vui recording started")
        next_tick = time.time()
        with sd.InputStream(
            samplerate=self.samplerate,
            device=self.device,
            channels=self.channels,
            blocksize=0,
            dtype=self.dtype,
            latency="low",
            callback=self.capture.callback):
            while not self._stop.is_set():
                now = time.time()
                if now > next_tick:
                    chunk = self.capture.get_last_second()
                    next_tick += 1.0 # add 1 second
                    if chunk is not None:
                        if self._audio_queue.full():
                            self.logger.info("queue full")
                            try:
                                _ = self._audio_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self.logger.info("Adding new shits")
                            self._audio_queue.put_nowait(chunk)
                        except queue.Full:
                            pass
                # time.sleep(0.005)
    
    def _process_loop(self):
        self.logger.info("vui preprocessing started")
        cooldown_until = 0.0
        while not self._stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
                print(chunk[:10])
                print(chunk.max(), chunk.min(), chunk.mean())
                print("#" * 100)
            except queue.Empty:
                continue

            # Pipeline: preprocess -> ASR -> sanitize -> detect
            x_16k = self.preprocessor(chunk)
            self.logger.info(x_16.min(), x_16.max(), x_16k.mean(), x_16k.shape)
            print("#" * 100)
            transcription = self.asr(x_16k)
            clean = self.recog.sanitize(transcription)
            cmd = self.recog.detect(clean)

            self.logger.info("vui command is %s"%clean) #NOTE
            now = time.time()
            if cmd and now > cooldown_until:
                self.logger.info("vui command:%s"%cmd)
                # emit command via callback
                self.on_command(cmd)
                cooldown_until = now + 1.0
            
            self._audio_queue.task_done()

class SensorManager:
    def __init__(self, logger):
        self.logger = logger
        self.device = adafruit_dht.DHT11(D4)

        self.logger.info("sensor ready")

    def read(self):
        while True:
            temperature = self.device.temperature
            humidity    = self.device.humidity
            if temperature is not None and humidity is not None:
                    return {"temperature": temperature, "humidity": humidity}

class CloudClient:
    def __init__(self, logger, host: str, port: int, user: str, password: str):
        self.logger = logger
        self.client = redis.Redis(
            host=host,
            port=port,
            username=user,
            password=password
        )
        assert self.client.ping(), "Failed to connect to Redis"
        

        self.logger.info("cloud ready")

    def store(self, key: str, value):
        pass
        # self.client.set(key, value)

class System:

    def __init__(self, args, logger, **kwargs):
        
        self.logger = logger
        self.state = DISABLED
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # vui
        self.vui = VUI(self.logger, self._handle_command, **kwargs)

        # sensor and cloud
        self.sensor = SensorManager(logger)
        self.cloud = CloudClient(logger, args.host, args.port, args.user, args.password)

        # sensor thread
        self._sensor_thread = threading.Thread(target=self._sensor_loop) #, daemon=True

        self.logger.info("system ready")
    
    def get_state(self):
        with self._lock:
            return self.state

    def set_state(self, state):
        with self._lock:
            self.state = state

    def run(self):
        self.vui.start()
        self._sensor_thread.start()

    def stop(self):
        self._stop.set()
        self.vui.stop()
        self._sensor_thread.join(timeout=2.0)

    def _handle_command(self, cmd):
        """Called by VUI when 'up' or 'stop' detected."""
        if cmd == "up":
            self.set_state(ENABLED)
            self.logger.info("system: UP -> ENABLED")
        elif cmd == "stop":
            self.set_state(DISABLED)
            self.logger.info("system: STOP -> DISABLED")
    
    def _sensor_loop(self):
        next_sample = time.time()
        while not self._stop.is_set():
            now = time.time()
            if self.get_state() == ENABLED and now >= next_sample:
                next_sample += 5.0
                record = self.sensor.read()
                print(record)
    
if __name__ == "__main__":
    logger = get_logger()
    # Retrieve arguments from command line interface
    args = get_cli()
    # Initialize the system and configure the audio acquisition
    # system = System(args, logger)
    vui_kwargs = {
        "channels"  : CHANNELS, 
        "device"    : DEVICE,
        "dtype"     : BIT_DEPTH,
        "samplerate": SAMPLERATE,
    }

    system = System(args, logger, **vui_kwargs)
    try:
        system.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()