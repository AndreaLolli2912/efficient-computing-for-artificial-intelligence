from subprocess import Popen
from time import sleep

import zipfile
import os
import numpy as np
import onnxruntime as ort
import pandas as pd

# Fix the CPU frequency to its maximum value (1.5 GHz)
Popen(
    'sudo sh -c "echo performance >'
    '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
    shell=True,
).wait()

x_test = np.random.normal(size=(1, 1, 16000)).astype(np.float32)

# Change the Group ID
GROUP_ID = 9

frontend_file = f'./HW2/model/Group{GROUP_ID}_frontend.onnx'
model_file =    f'./HW2/model/Group{GROUP_ID}_model_INT8.onnx.zip'

frontend_size = os.path.getsize(frontend_file)
model_size = os.path.getsize(model_file)

print(f'Feature Extraction Size: {frontend_size / 2**10:.1f}KB')
print(f'Model Size: {model_size / 2**10:.1f}KB')
print(f'Total Size: {(frontend_size + model_size) / 2**10:.1f}KB')


sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1
sess_opt.enable_profiling = True

ort_frontend = ort.InferenceSession(frontend_file, sess_options=sess_opt)
# ORT profile file names use the timestamp.
# Sleep 1 minute to generate two different file names.
sleep(60)
if model_file.lower().endswith(".zip"):
    with zipfile.ZipFile(model_file, "r") as z:
        inner_name = z.namelist()[0]
        extract_dir = os.path.split(os.path.dirname(model_file))[0]

        # Estrai il file nella cartella corretta
        z.extract(inner_name, extract_dir)

        # Percorso finale corretto
        model_file = os.path.join(extract_dir, inner_name)



ort_model = ort.InferenceSession(model_file, sess_options=sess_opt)

tot_latencies = []
for i in range(100):
    features = ort_frontend.run(None, {'input': x_test})[0]
    outputs = ort_model.run(None, {'input': features})[0]
    sleep(0.1)

frontend_profile = ort_frontend.end_profiling()
model_profile = ort_model.end_profiling()


def print_stats(profile_file):
    df = pd.read_json(profile_file)
    df_filtered = df[['name', 'dur']]
    name_order = df_filtered.drop_duplicates('name')['name']

    # Group by 'name' and calculate median/std 'dur'
    stats = (
        df_filtered.groupby('name')['dur']
        .agg(['median', 'std', 'min', 'max'])
        .reset_index()
    )
    stats['name'] = pd.Categorical(
        stats['name'], categories=name_order, ordered=True
    )
    stats = stats.sort_values('name')

    return stats['median'].iloc[-1] / 1000


frontend_latency = print_stats(frontend_profile)
model_latency = print_stats(model_profile)
total_latency = frontend_latency + model_latency


print(f'Feature Extraction Latency: {frontend_latency:.1f} ms')
print(f'Model Latency: {model_latency:.1f} ms')
print(f'Total Latency: {total_latency:.1f} ms')