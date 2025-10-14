# EC4AI - raspberrypi

## Script

```bash
python main.py --bit-depth int16 --sampling-rate 48000 --duration 1

```

| Parameter         | Type  | Allowed Values       | Description                                                              |
| ----------------- | ----- | -------------------- | ------------------------------------------------------------------------ |
| `--bit-depth`     | `str` | `int16`, `int32`     | Audio sample precision. 16-bit is standard; 32-bit for higher precision. |
| `--sampling-rate` | `int` | `48000`, `44100`     | Audio sampling frequency in Hz.                                          |
| `--duration`      | `int` | Any positive integer | Recording duration in seconds.                                           |

