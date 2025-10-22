"src/ec4ai/utils/audio_utils.py"
import torchaudio
from torchaudio import transforms as T


def load_audio(audio_path, sampling_rate):
    # load audio
    audio_data, s_rate = torchaudio.load(audio_path)
    # check if audio's sampling rate differs from the one specified
    if s_rate != sampling_rate:
        # apply resampling
        audio_data, s_rate = resample_audio(audio_data, s_rate, sampling_rate)

    return audio_data, s_rate

def resample_audio(audio_data, old_sampling_rate, new_sampling_rate):
    # create transform instance and resample audio
    transform = T.Resample(old_sampling_rate, new_sampling_rate)
    audio_data = transform(audio_data).clone()
    return audio_data, new_sampling_rate
