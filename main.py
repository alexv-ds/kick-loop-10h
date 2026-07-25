#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import soundfile as sf
import typing
import random


def add_sample(
    pos: int,
    out: np.ndarray[
        tuple[int], np.dtype[np.float32 | np.float64 | np.int32 | np.int16]
    ],
    sample: sf.AudioData | sf.AudioData_2d,
):
    n = min(len(sample), len(out) - pos)
    out[pos : pos + n] += sample[:n]

def multipile_sample(
    pos: int,
    out: np.ndarray[
        tuple[int], np.dtype[np.float32 | np.float64 | np.int32 | np.int16]
    ],
    sample: sf.AudioData | sf.AudioData_2d,
):
    n = min(len(sample), len(out) - pos)
    out[pos : pos + n] *= sample[:n]

def gen_poisson(duration: float, rate: float) -> npt.NDArray[np.float64]:
    if duration <= 0:
        return np.empty(0, dtype=float)

    if rate <= 0:
        raise ValueError("rate must be positive")

    timestamps: typing.List[float] = []
    t = 0.0

    while True:
        t += np.random.exponential(1.0 / rate)

        if t >= duration:
            break

        timestamps.append(t)

    return np.asarray(timestamps, dtype=np.float64)


def gen_markov(
    duration: float,
    S: typing.Tuple[float, float],
    M: typing.Tuple[float, float],
    L: typing.Tuple[float, float],
) -> npt.NDArray[np.float64]:
    current_state = "M"
    states = {
        "S": S,
        "M": M,
        "L": L,
    }

    transitions = {
        "S": {"S": 0.7, "M": 0.25, "L": 0.05},
        "M": {"S": 0.2, "M": 0.6, "L": 0.2},
        "L": {"S": 0.05, "M": 0.3, "L": 0.65},
    }

    timestamps: typing.List[float] = []
    t = 0.0

    while True:
        mean, sigma = states[current_state]
        interval = random.gauss(mean, sigma)

        if interval <= 0:
            print("Discard interval: {}", interval)
            continue

        t += interval
        current_state = random.choices(
            list(transitions[current_state].keys()),
            weights=list(transitions[current_state].values()),
        )[0]

        if t >= duration:
            break
        timestamps.append(t)

    return np.asarray(timestamps, dtype=np.float64)


def fill_audio(
    out: np.ndarray[
        tuple[int], np.dtype[np.float32 | np.float64 | np.int32 | np.int16]
    ],
    sample: sf.AudioData | sf.AudioData_2d,
    sr: int,
    events: npt.NDArray[np.float64],
):
    for pos in (events * sr).astype(int):
        if pos >= len(out):
            print(f"Skip event at position {pos}")
            continue
        add_sample(pos, out, sample)


def fill_silence(
    out: np.ndarray[
        tuple[int], np.dtype[np.float32 | np.float64 | np.int32 | np.int16]
    ],
    sr: int,
    rate: float,
    silense_range: range,
    fade: float = 0.5
):
    for pos in (gen_poisson(len(out) / sr, rate) * sr).astype(int):
        duration = random.randrange(
            silense_range.start, silense_range.stop, silense_range.step
        )

        n = duration * sr
        silence = np.zeros(n, dtype=out.dtype)

        fade_len = int(fade * sr)
        if fade_len > 0 and n > 0:
            fade_len = min(fade_len, n // 2)  # чтобы атака и релиз не пересекались
            silence[:fade_len] = np.linspace(1.0, 0.0, fade_len).astype(out.dtype)
            silence[-fade_len:] = np.linspace(0.0, 1.0, fade_len).astype(out.dtype)

        multipile_sample(pos, out, silence)

# ######################## #
# ######################## #
# ######################## #

sample = "kick1.mp3"
output = "generated.mp3"
duration_sec = 60

print(f"Prepare - {sample}")

kick, sr = sf.read(sample)
if kick.ndim > 1:
    kick = kick.mean(axis=1)

print(f"Sample rate: {sr}")
print(f"Sample duration: {(len(kick) / sr):.1f} sec")


audio = np.zeros(duration_sec * sr, kick.dtype)

print(f"Fill: {(len(audio) / sr):.1f} sec")


events = gen_poisson(duration_sec, 0.1)
fill_audio(audio, kick, sr, events)

events = gen_markov(
    duration_sec,
    S=(300.0 / 1000.0, 15.0 / 1000.0),
    M=(700.0 / 1000.0, 50.0 / 1000.0),
    L=(1200.0 / 1000.0, 200.0 / 1000.0),
)
fill_audio(audio, kick, sr, events)

fill_silence(audio, sr, 0.05, range(2, 5))


np.clip(audio, -0.3, 0.3, out=audio)
audio *= 0.97 / 0.3


print(f"Save - {output}")
sf.write(output, audio, sr)
