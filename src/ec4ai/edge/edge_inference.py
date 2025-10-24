from src.ec4ai.utils.audio_utils import load_audio

from subprocess import Popen
from time import sleep, time

import numpy as np
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MAX_SAMPLING_RATE = 16_000

def load_model_and_processor(model_name: str = 'openai/whisper-tiny.en'):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return model, processor

def edge_inference(audio_path: str, n_trials:int = 20):
    # Fix the CPU frequency to its maximum value (1.5 GHz)
    Popen(
        'sudo sh -c "echo performance >'
        '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
        shell=True,
    ).wait()

    # Load model and processor
    model, processor = load_model_and_processor()

    # Load test WAV file
    audio_data, sampling_rate = load_audio(audio_path, MAX_SAMPLING_RATE)
    audio_data = audio_data.squeeze(0)

    times = []

    for i in range(n_trials):
        start = time()
        
        inputs = processor(
            audio_data, 
            sampling_rate=MAX_SAMPLING_RATE, 
            return_tensors="pt"
        )

        predicted_ids = model.generate(input_features = inputs.input_features)

        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )

        print(transcription)
        end = time()
        times.append(end - start)
        sleep(0.1)

    median_time = np.median(times)
    std_time = np.std(times)

    print(f'Latency: {median_time:.2f}+/-{std_time:.2f}s')