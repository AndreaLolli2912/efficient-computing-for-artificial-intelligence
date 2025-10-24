from subprocess import Popen
from time import sleep, time

import numpy as np
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Fix the CPU frequency to its maximum value (1.5 GHz)
Popen(
    'sudo sh -c "echo performance >'
    '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
    shell=True,
).wait()

# Load model and processor

# Load test WAV file
filename = 'labs/HOMEWORK_1/audio/inferenceAudio/stop_0b40aa8e_nohash_0.wav'
x, sampling_rate = torchaudio.load(filename)
x = x.squeeze(0)


for name in [
    # 'tiny', 
    # 'base', 
    # 'small', 
    'medium', 
    'large', 
    'largev2'
    ]:
    model_name = f'openai/whisper-{name}.en'

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    times = []

    print(f"Running edge inference on model {model_name}...")
    for i in range(20):
        start = time()
        predicted_ids = model.generate(
            processor(
            x, sampling_rate=16000, return_tensors="pt"
            ).input_features)
        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )
        times.append(time() - start)
        sleep(0.1)
    
    print("\n"*2, "-"*50)
    print(f'Model: {model_name}')
    print(f'Latency: {np.median(times):.2f}+/-{np.std(times):.2f}s')
    print("\n"*3)