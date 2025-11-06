import collections
import logging
import queue
import re
import multiprocessing as mp
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

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    )
    handler.setFormatter(formatter)
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
        x = torch.tensor(arr_int16, dtype=torch.float32)
        x = x / 32_768.0
        x_16k = self.resample(x)
        return x_16k

class WhisperModel:
    def __init__(self, logger):
        self.logger = logger
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
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
    def __init__(self, logger, command_queue, **vui_kwargs):
        self.logger = logger
        self.command_queue = command_queue  # For sending commands to main process

        self.channels = vui_kwargs["channels"]
        self.device = vui_kwargs["device"]        
        self.dtype = vui_kwargs["dtype"]
        self.samplerate = vui_kwargs["samplerate"]

        # inject components
        self.asr = WhisperModel(self.logger)
        self.capture = AudioCapture(self.samplerate)
        self.preprocessor = AudioPreprocessor(sr_in=self.samplerate, sr_out=16_000)
        self.recog = CommandRecognizer()

        # queue for 1s audio chunks
        self._audio_queue = mp.Queue(maxsize=2)

        # process control
        self._stop = mp.Event()
        self._capture_process = None
        self._process_process = None

        self.logger.info("vui ready")
    
    def start(self):
        self._capture_process = mp.Process(target=self._capture_loop)
        self._process_process = mp.Process(target=self._process_loop)
        
        self._capture_process.start()
        self._process_process.start()

    def stop(self):
        self._stop.set()
        
        if self._capture_process:
            self._capture_process.join(timeout=2.0)
            if self._capture_process.is_alive():
                self._capture_process.terminate()
                
        if self._process_process:
            self._process_process.join(timeout=2.0)
            if self._process_process.is_alive():
                self._process_process.terminate()

    def _capture_loop(self):
        # Re-initialize logger in subprocess
        logger = get_logger('vui.capture')
        logger.info("vui recording started")
        
        # Create capture object in this process
        capture = AudioCapture(self.samplerate)
        next_tick = time.time()
        
        with sd.InputStream(
            samplerate=self.samplerate,
            device=self.device,
            channels=self.channels,
            blocksize=0,
            dtype=self.dtype,
            latency="low",
            callback=capture.callback):
            
            while not self._stop.is_set():
                now = time.time()
                if now >= next_tick:
                    chunk = capture.get_last_second()
                    next_tick += 1.0
                    if chunk is not None:
                        try:
                            self._audio_queue.put(chunk, block=False)
                        except:
                            # Drop oldest if queue full
                            try:
                                _ = self._audio_queue.get_nowait()
                                self._audio_queue.put(chunk, block=False)
                            except:
                                pass
                time.sleep(0.01)
    
    def _process_loop(self):
        # Re-initialize components in subprocess
        logger = get_logger('vui.process')
        logger.info("vui preprocessing started")
        
        # Create model objects in this process
        asr = WhisperModel(logger)
        preprocessor = AudioPreprocessor(sr_in=self.samplerate, sr_out=16_000)
        recog = CommandRecognizer()
        
        cooldown_until = 0.0
        
        while not self._stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except:
                continue

            # Pipeline: preprocess -> ASR -> sanitize -> detect
            x_16k = preprocessor(chunk)
            transcription = asr(x_16k)
            clean = recog.sanitize(transcription)
            cmd = recog.detect(clean)

            logger.info("vui transcription: %s" % clean)
            now = time.time()
            if cmd and now > cooldown_until:
                logger.info("vui command: %s" % cmd)
                # Send command to main process via queue
                try:
                    self.command_queue.put(cmd, block=False)
                except:
                    pass
                cooldown_until = now + 1.0

class SensorManager:
    def __init__(self, logger, state_value, state_lock, stop_event, args):
        self.logger = logger
        self.state_value = state_value
        self.state_lock = state_lock
        self.stop_event = stop_event
        self.args = args
        
        # Process reference
        self._sensor_process = None
        
        self.logger.info("sensor manager ready")
    
    def start(self):
        self._sensor_process = mp.Process(target=self._sensor_loop)
        self._sensor_process.start()
    
    def stop(self):
        self.stop_event.set()
        if self._sensor_process:
            self._sensor_process.join(timeout=2.0)
            if self._sensor_process.is_alive():
                self._sensor_process.terminate()

    def _sensor_loop(self):
        # Re-initialize components in subprocess
        logger = get_logger('sensor')
        logger.info("sensor process started")
        
        # Initialize hardware in this process
        device = adafruit_dht.DHT11(D4)
        
        # Initialize cloud client in this process
        try:
            client = redis.Redis(
                host=self.args.host,
                port=self.args.port,
                username=self.args.user,
                password=self.args.password
            )
            assert client.ping(), "Failed to connect to Redis"
            logger.info("cloud ready")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            client = None
        
        next_sample = time.time()
        
        while not self.stop_event.is_set():
            now = time.time()
            
            # Check state
            with self.state_lock:
                current_state = self.state_value.value
            
            if current_state == ENABLED and now >= next_sample:
                next_sample += 5.0
                try:
                    temperature = device.temperature
                    humidity = device.humidity
                    if temperature is not None and humidity is not None:
                        record = {"temperature": temperature, "humidity": humidity}
                        logger.info(f"sensor reading: {record}")
                        # Optionally store to Redis
                        # if client:
                        #     client.set(f"sensor:{int(now)}", str(record))
                except RuntimeError as e:
                    logger.warning(f"sensor read error: {e}")
            
            time.sleep(0.1)

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
        self.args = args
        
        # Shared state using multiprocessing primitives
        self.state_value = mp.Value('i', DISABLED)
        self.state_lock = mp.Lock()
        self._stop = mp.Event()
        
        # Command queue for VUI -> System communication
        self.command_queue = mp.Queue(maxsize=5)

        # VUI with command queue
        self.vui = VUI(self.logger, self.command_queue, **kwargs)

        # Sensor manager
        self.sensor = SensorManager(self.logger, self.state_value, 
                                    self.state_lock, self._stop, args)

        self.logger.info("system ready")
    
    def get_state(self):
        with self.state_lock:
            return self.state_value.value

    def set_state(self, state):
        with self.state_lock:
            self.state_value.value = state

    def run(self):
        self.vui.start()
        self.sensor.start()
        
        self.logger.info("system running")
        
        # Main loop: handle commands from VUI
        while not self._stop.is_set():
            try:
                cmd = self.command_queue.get(timeout=0.5)
                self._handle_command(cmd)
            except:
                continue

    def stop(self):
        self.logger.info("stopping system...")
        self._stop.set()
        self.vui.stop()
        self.sensor.stop()
        self.logger.info("system stopped")

    def _handle_command(self, cmd):
        """Called when 'up' or 'stop' detected."""
        if cmd == "up":
            self.set_state(ENABLED)
            self.logger.info("system: UP -> ENABLED")
        elif cmd == "stop":
            self.set_state(DISABLED)
            self.logger.info("system: STOP -> DISABLED")

if __name__ == "__main__":
    # Set multiprocessing start method (important for library compatibility)
    mp.set_start_method('spawn')
    
    logger = get_logger()
    # Retrieve arguments from command line interface
    args = get_cli()
    # Initialize the system and configure the audio acquisition
    vui_kwargs = {
        "channels": CHANNELS, 
        "device": DEVICE,
        "dtype": BIT_DEPTH,
        "samplerate": SAMPLERATE,
    }

    system = System(args, logger, **vui_kwargs)
    try:
        system.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()