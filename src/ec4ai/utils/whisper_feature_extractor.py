import torch
import torchaudio
from torchaudio import transforms as T
from audio_utils import load_audio

def whisper_processor(input, n_fft, inp):
    pass


if __name__ == "__main__":
    MAX_SAMPLING_RATE = 16_000
    audio_path = "data/audio/1761128443.202978.wav"
    x, sampling_rate = load_audio(audio_path, MAX_SAMPLING_RATE)

    T.MelSpectrogram()
    x = torch.stft(x, n_fft = 400)
    print(x)