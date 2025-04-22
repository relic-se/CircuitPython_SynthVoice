# SPDX-FileCopyrightText: Copyright (c) 2025 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import math

import synthio
import ulab.numpy as np

import relic_synthvoice

try:
    from circuitpython_typing import ReadableBuffer
except ImportError:
    pass

_LOG_2 = math.log(2)


class Drone(relic_synthvoice.Voice):

    def __init__(self, synthesizer: synthio.Synthesizer, voices: int = 3, root: float = 130.81):
        self._synthesizer = synthesizer

        self._notenum = -1
        self._velocity = 0.0

        self._root = root
        self._tune = 0.0
        self._detune = 0.0

        self._waveform = None

        self._attack_time = 0.5
        self._release_time = 0.5

        self._amplitude = synthio.Math(
            synthio.MathOperation.SUM,
            1 / voices,
            synthio.Math(
                synthio.MathOperation.PRODUCT,
                synthio.LFO(  # Tremolo synthio.LFO
                    waveform=None,
                    rate=1.0,
                    scale=0.0,
                ),
                synthio.LFO(  # Tremolo Delay
                    waveform=np.array([0, 32767], dtype=np.int16),
                    rate=1 / 0.001,
                    once=True,
                ),
            ),
            0.0,
        )

        self._freq_lerp = relic_synthvoice.LerpBlockInput(
            rate=0.0,
            value=0.0,
        )

        self._bend = synthio.Math(
            synthio.MathOperation.SUM,
            self._freq_lerp.block,  # Frequency Lerp
            synthio.Math(
                synthio.MathOperation.PRODUCT,
                synthio.LFO(  # Vibrato synthio.LFO
                    waveform=None, rate=1.0, scale=0.0, offset=0.0
                ),
                synthio.LFO(  # Vibrato Delay
                    waveform=np.array([0, 32767], dtype=np.int16),
                    rate=1 / 0.001,
                    once=True,
                ),
            ),
            0.0,
        )

        self._notes = tuple([
            synthio.Note(
                frequency=root,
                waveform=self._waveform,
                amplitude=self._amplitude,
                bend=self._bend,
            ) for i in range(voices)
        ])

        self._filter_envelope = relic_synthvoice.AREnvelope(
            attack_time=0.0,
            release_time=0.0,
            amount=0.0,
        )

        self._filter_frequency = synthio.Math(
            synthio.MathOperation.MAX,
            synthio.Math(
                synthio.MathOperation.SUM,
                synthesizer.sample_rate / 2,
                self._filter_envelope.block,
                synthio.Math(
                    synthio.MathOperation.PRODUCT,
                    synthio.LFO(  # Filter synthio.LFO
                        waveform=None,
                        rate=1.0,
                        scale=0.0,
                        offset=0.0,
                    ),
                    synthio.LFO(  # Filter Delay
                        waveform=np.array([0, 32767], dtype=np.int16),
                        rate=1 / 0.001,
                        once=True,
                    ),
                ),
            ),
            40.0,  # Minimum allowed frequency
        )

        self._update_biquad(frequency=self._filter_frequency)
        self._update_envelope()

        self._append_blocks()

    @property
    def notes(self) -> tuple[synthio.Note]:
        return self._notes

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        return self._filter_envelope.blocks + self._freq_lerp.blocks

    def press(self, notenum: int | None = None, velocity: float | int = 1.0) -> bool:
        if not super().press(1 if notenum is None else notenum, velocity):
            return False
        self.frequency = self._root if notenum is None else synthio.midi_to_hz(notenum)
        self._filter_envelope.press()
        self._filter_frequency.a.c.b.retrigger()
        self._amplitude.b.b.retrigger()  # Tremolo Delay
        self._bend.b.b.retrigger()  # Vibrato Delay
        return True
    
    def release(self) -> bool:
        if not super().release():
            return False
        self._filter_envelope.release()
        return True

    def _update_root(self):
        for i, note in enumerate(self._notes):
            tune = 0

            if type(self._detune) is tuple:
                tune = float(self._detune[i % len(self._detune)])
            else:
                tune = float(self._detune) * i / (len(self._notes) - 1)

            if type(self._tune) is tuple:
                tune += float(self._tune[i % len(self._tune)])
            else:
                tune += float(self._tune)

            note.frequency = self._root * pow(2, tune)

    @property
    def frequency(self) -> float:
        return math.exp(self._freq_lerp.value * _LOG_2) * self._root

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._freq_lerp.value = math.log(value / self._root) / _LOG_2

    @property
    def tune(self) -> float|tuple:
        return self._tune
    
    @tune.setter
    def tune(self, value: float|tuple) -> None:
        self._tune = value
        self._update_root()

    @property
    def detune(self) -> float|tuple:
        return self._detune
    
    @detune.setter
    def detune(self, value: float|tuple) -> None:
        self._detune = value
        self._update_root()

    @property
    def glide(self) -> float:
        return self._freq_lerp.rate

    @glide.setter
    def glide(self, value: float) -> None:
        self._freq_lerp.rate = value
    
    @property
    def vibrato_rate(self) -> float:
        return self._bend.b.a.rate

    @vibrato_rate.setter
    def vibrato_rate(self, value: float) -> None:
        self._bend.b.a.rate = value

    @property
    def vibrato_depth(self) -> float:
        return self._bend.b.a.scale

    @vibrato_depth.setter
    def vibrato_depth(self, value: float) -> None:
        self._bend.b.a.scale = value

    @property
    def vibrato_delay(self) -> float:
        return 1 / self._bend.b.b.rate

    @vibrato_delay.setter
    def vibrato_delay(self, value: float) -> None:
        self._bend.b.b.rate = 1 / max(value, 0.001)

    @property
    def waveform(self) -> ReadableBuffer | None:
        return self._waveform

    @waveform.setter
    def waveform(self, value: ReadableBuffer | None) -> None:
        self._waveform = value
        for note in self._notes:
            note.waveform = self._waveform

    @property
    def amplitude(self) -> float:
        return self._amplitude.a

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude.a = value

    @property
    def tremolo_rate(self) -> float:
        return self._amplitude.b.a.rate

    @tremolo_rate.setter
    def tremolo_rate(self, value: float) -> None:
        self._amplitude.b.a.rate = value

    @property
    def tremolo_depth(self) -> float:
        return self._amplitude.b.a.scale

    @tremolo_depth.setter
    def tremolo_depth(self, value: float) -> None:
        self._amplitude.b.a.scale = value

    @property
    def tremolo_delay(self) -> float:
        return 1 / self._amplitude.b.b.rate

    @tremolo_delay.setter
    def tremolo_delay(self, value: float) -> None:
        self._amplitude.b.b.rate = 1 / max(value, 0.001)

    def _update_envelope(self):
        self._envelope = synthio.Envelope(
            attack_time=self._attack_time,
            attack_level=1.0,
            decay_time=0.0,
            sustain_level=1.0,
            release_time=self._release_time,
        )
        for note in self._notes:
            note.envelope = self._envelope

    @property
    def attack_time(self) -> float:
        return self._envelope.attack_time

    @attack_time.setter
    def attack_time(self, value: float) -> None:
        self._attack_time = max(value, 0.0)
        self._update_envelope()

    @property
    def release_time(self) -> float:
        return self._release_time

    @release_time.setter
    def release_time(self, value: float) -> None:
        self._release_time = max(value, 0.0)
        self._update_envelope()

    @property
    def filter_frequency(self) -> float:
        return self._filter_frequency.a.a

    @filter_frequency.setter
    def filter_frequency(self, value: float) -> None:
        self._filter_frequency.a.a = min(max(value, 1), self._synthesizer.sample_rate / 2)

    @property
    def filter_attack_time(self) -> float:
        return self._filter_envelope.attack_time

    @filter_attack_time.setter
    def filter_attack_time(self, value: float) -> None:
        self._filter_envelope.attack_time = value

    @property
    def filter_amount(self) -> float:
        return self._filter_envelope.amount

    @filter_amount.setter
    def filter_amount(self, value: float) -> None:
        self._filter_envelope.amount = value

    @property
    def filter_release_time(self) -> float:
        return self._filter_envelope.release_time

    @filter_release_time.setter
    def filter_release_time(self, value: float) -> None:
        self._filter_envelope.release_time = value

    @property
    def filter_rate(self) -> float:
        return self._filter_frequency.a.c.a.rate

    @filter_rate.setter
    def filter_rate(self, value: float) -> None:
        self._filter_frequency.a.c.a.rate = value

    @property
    def filter_depth(self) -> float:
        return self._filter_frequency.a.c.a.scale

    @filter_depth.setter
    def filter_depth(self, value: float) -> None:
        self._filter_frequency.a.c.a.scale = value

    @property
    def filter_delay(self) -> float:
        return 1 / self._filter_frequency.a.c.b.rate

    @filter_delay.setter
    def filter_delay(self, value: float) -> None:
        self._filter_frequency.a.c.b.rate = 1 / max(value, 0.001)
