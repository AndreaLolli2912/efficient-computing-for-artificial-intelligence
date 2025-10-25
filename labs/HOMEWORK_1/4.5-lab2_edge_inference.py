from subprocess import Popen
from time import sleep, time

from numpy import median, std
from torchaudio import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# DELETE ME
from os import getpid
from psutil import Process

# Fix the CPU frequency to its maximum value (1.5 GHz)
Popen(
    'sudo sh -c "echo performance >'
    '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
    shell=True,
).wait()

# Load test WAV file
x, _ = load('labs/HOMEWORK_1/audio/stop_0b40aa8e_nohash_0.wav')
x = x.squeeze(0)


name = 'tiny' # 'tiny', 'base', 'small', 'medium', 'large', 'largev2'    


# Memory before loading the model
print(f"Memory used: {Process(getpid()).memory_info().rss / (1024 ** 2):.2f} MB")


processor = WhisperProcessor.from_pretrained(f'openai/whisper-{name}.en')
model = WhisperForConditionalGeneration.from_pretrained(f'openai/whisper-{name}.en')

# Memory after loading the model
print(f"Memory used: {Process(getpid()).memory_info().rss / (1024 ** 2):.2f} MB")

# Parameter count and approximate memory usage
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Approx. memory: {total_params * 4 / (1024 ** 2):.2f} MB")  # assuming 32-bit floats    


    
print(f"Running edge inference on model {name}...")
times = []
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
print(f'Model: {name}')
print(f'Latency: {median(times):.2f}+/-{std(times):.2f}s')
print("\n"*3)