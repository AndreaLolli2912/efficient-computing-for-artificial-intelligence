import threading, time, collections, queue, re
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import adafruit_dht
from board import D4
import redis

DISABLED, ENABLED = 0, 1

# ========== AUDIO COMPONENTS ==========

class AudioCapture:
    """Manages sounddevice InputStream and 1s ring buffer at 48 kHz mono int16."""
    def __init__(self, device, samplerate=48_000, channels=1, dtype="int16", logger=None):
        self.device = device
        self.sr = samplerate
        self.channels = channels
        self.dtype = dtype
        self.logger = logger
        self._ring = collections.deque(maxlen=self.sr)

    def callback(self, indata, frames, time_info, status):
        """Minimal callback: append mono int16 frames to ring."""
        if status and self.logger:
            self.logger.warning(f"audio status: {status}")
        self._ring.extend(indata[:, 0].copy().astype(np.int16))

    def get_last_second(self):
        """Snapshot exactly 1s (48k samples) if available."""
        if len(self._ring) == self.sr:
            return np.frombuffer(np.array(self._ring, dtype=np.int16).tobytes(), dtype=np.int16)
        return None


class AudioPreprocessor:
    """Converts int16 -> float32, normalizes, resamples 48k->16k, squeezes channel."""
    def __init__(self, sr_in=48_000, sr_out=16_000):
        self.resample = T.Resample(orig_freq=sr_in, new_freq=sr_out)

    def run(self, arr_int16):
        """Returns torch.float32 [T] at 16 kHz."""
        x = torch.tensor(arr_int16, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x / 32768.0
        x16 = self.resample(x)
        x16 = x16.squeeze(0)
        return x16


class WhisperModel:
    """Loads Whisper tiny and exposes transcribe()."""
    def __init__(self, logger=None):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.logger = logger
        if self.logger:
            self.logger.info("whisper model loaded")

    def transcribe(self, x_16k):
        """x_16k: torch.float32 [T] at 16 kHz -> return raw text."""
        with torch.no_grad():
            feats = self.processor(x_16k, sampling_rate=16_000, return_tensors="pt").input_features
            ids = self.model.generate(feats)
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text


class CommandRecognizer:
    """Sanitizes text and detects command words."""
    def sanitize(self, text):
        return re.sub(r'[^a-z0-9\s]', '', text.strip().lower())

    def detect(self, text_clean):
        """Returns 'up', 'stop', or None."""
        if "up" in text_clean:
            return "up"
        elif "stop" in text_clean:
            return "stop"
        return None


# ========== VUI CLASS (encapsulates voice interface) ==========

class VUI:
    """Voice User Interface: records continuously, processes every 1s, emits commands."""
    def __init__(self, device, logger=None, on_command=None):
        """
        Args:
            device: sounddevice device ID
            logger: logger instance
            on_command: callback(cmd: str) called when 'up' or 'stop' detected
        """
        self.logger = logger
        self.on_command = on_command  # callback for command events

        # Inject components
        self.capture = AudioCapture(device=device, logger=logger)
        self.preproc = AudioPreprocessor()
        self.asr = WhisperModel(logger=logger)
        self.recog = CommandRecognizer()

        # Queue for 1s audio chunks (capture -> processor)
        self._audio_queue = queue.Queue(maxsize=2)

        # Thread control
        self._stop = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)

        if self.logger:
            self.logger.info("vui ready")

    def start(self):
        """Start capture and processing threads."""
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self):
        """Stop threads and cleanup."""
        self._stop.set()
        self._capture_thread.join(timeout=2.0)
        self._process_thread.join(timeout=2.0)

    # ---- Capture thread: records continuously, enqueues 1s every second ----
    def _capture_loop(self):
        if self.logger:
            self.logger.info("vui capture started")
        next_tick = time.time()
        with sd.InputStream(
            samplerate=self.capture.sr,
            blocksize=0,  # adaptive
            device=self.capture.device,
            channels=self.capture.channels,
            dtype=self.capture.dtype,
            callback=self.capture.callback
        ):
            while not self._stop.is_set():
                now = time.time()
                if now >= next_tick:
                    next_tick += 1.0
                    chunk = self.capture.get_last_second()
                    if chunk is not None:
                        # Keep latest 1s in queue
                        if self._audio_queue.full():
                            try:
                                _ = self._audio_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self._audio_queue.put_nowait(chunk)
                        except queue.Full:
                            pass
                # time.sleep(0.005)

    # ---- Process thread: preprocesses, runs Whisper, emits commands ----
    def _process_loop(self):
        if self.logger:
            self.logger.info("vui processor started")
        cooldown_until = 0.0
        while not self._stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Pipeline: preprocess -> ASR -> sanitize -> detect
            x16 = self.preproc.run(chunk)
            text = self.asr.transcribe(x16)
            clean = self.recog.sanitize(text)
            cmd = self.recog.detect(clean)

            now = time.time()
            if cmd and now >= cooldown_until:
                if self.logger:
                    self.logger.info(f"vui command: {cmd.upper()}")
                # Emit command via callback
                if self.on_command:
                    self.on_command(cmd)
                cooldown_until = now + 1.0

            self._audio_queue.task_done()


# ========== SENSOR COMPONENTS ==========

class SensorManager:
    """Reads DHT11 with retries."""
    def __init__(self, logger=None):
        self.device = adafruit_dht.DHT11(D4)
        self.logger = logger
        if self.logger:
            self.logger.info("sensor ready")

    def read_retry(self, retries=3, delay=2.0):
        for _ in range(retries):
            try:
                t = self.device.temperature
                h = self.device.humidity
                if t is not None and h is not None:
                    return t, h
            except RuntimeError:
                pass
            time.sleep(delay)
        raise RuntimeError("DHT11 read failed after retries")


class CloudClient:
    """Writes readings to Redis."""
    def __init__(self, host, port, user, password, logger=None):
        self.client = redis.Redis(host=host, port=port, username=user, password=password)
        assert self.client.ping(), "Failed to connect to Redis"
        self.logger = logger
        if self.logger:
            self.logger.info("cloud ready")

    def store(self, key, value_dict):
        self.client.hset(key, mapping=value_dict)


# ========== SYSTEM ORCHESTRATOR ==========

class System:
    """Composition root: wires VUI + sensor + cloud, manages state machine."""
    def __init__(self, args, logger):
        self.logger = logger
        self.state = DISABLED
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Inject VUI with command callback
        self.vui = VUI(device=args.device, logger=logger, on_command=self._handle_command)

        # Inject sensor + cloud
        self.sensor = SensorManager(logger=logger)
        self.cloud = CloudClient(args.host, args.port, args.user, args.password, logger=logger)

        # Sensor thread
        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)

        self.logger.info("system ready")

    def get_state(self):
        with self._lock:
            return self.state

    def set_state(self, s):
        with self._lock:
            self.state = s

    def start(self):
        """Start VUI and sensor threads."""
        self.vui.start()
        self._sensor_thread.start()

    def stop(self):
        """Stop all threads."""
        self._stop.set()
        self.vui.stop()
        self._sensor_thread.join(timeout=2.0)

    # ---- Command callback: VUI -> System state transitions ----
    def _handle_command(self, cmd):
        """Called by VUI when 'up' or 'stop' detected."""
        if cmd == "up":
            self.set_state(ENABLED)
            self.logger.info("system: UP -> ENABLED")
        elif cmd == "stop":
            self.set_state(DISABLED)
            self.logger.info("system: STOP -> DISABLED")

    # ---- Sensor thread: samples every 5s only when ENABLED ----
    def _sensor_loop(self):
        self.logger.info("sensor loop started")
        next_sample = time.time()
        while not self._stop.is_set():
            now = time.time()
            if self.get_state() == ENABLED and now >= next_sample:
                next_sample += 5.0
                try:
                    t, h = self.sensor.read_retry()
                    key = f"hygro:{int(now)}"
                    self.cloud.store(key, {"temperature": t, "humidity": h})
                    self.logger.info(f"stored {key}: T={t}, H={h}")
                except Exception as e:
                    self.logger.error(f"sensor/upload error: {e}")
            time.sleep(0.05)


# ========== MAIN ==========

if __name__ == "__main__":
    import logging
    import argparse

    def get_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--host", type=str, default="redis-host")
    parser.add_argument("--port", type=int, default=15750)
    parser.add_argument("--user", type=str, default="default")
    parser.add_argument("--password", type=str, default="your-password")
    args = parser.parse_args()

    logger = get_logger()
    system = System(args, logger)

    try:
        system.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("shutting down")
        system.stop()
