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
    """A multi-oscillator voice intended to generate "droning" synthesizer sounds with the
    following features:
    
    * per-oscillator tuning and detuning
    * amplitude & filter envelopes
    * LFOs (low-frequency oscillators) for amplitude (tremolo), filter, & pitch (vibrato)
    * pitch glide

    :param synthesizer: The :class:`synthio.Synthesizer` object this voice will be used with.
    :param oscillators: The number of oscillators to control with this voice.
    :param root: The root frequency used to calculate tuning. Defaults to 440.0hz. Changing this
        value will affect tuning properties.
    """

    def __init__(
        self, synthesizer: synthio.Synthesizer, oscillators: int = 3, root: float = 130.81
    ):
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
            1 / oscillators,
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

        self._notes = tuple(
            [
                synthio.Note(
                    frequency=root,
                    waveform=self._waveform,
                    amplitude=self._amplitude,
                    bend=self._bend,
                )
                for i in range(oscillators)
            ]
        )

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
        """Get all :class:`synthio.Note` objects attributed to this voice."""
        return self._notes

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        """Get all :class:`synthio.BlockInput` objects attributed to this voice."""
        return self._filter_envelope.blocks + self._freq_lerp.blocks

    def press(self, notenum: int | None = None, velocity: float | int = 1.0) -> bool:
        """Update the voice to be "pressed" with a specific MIDI note number and velocity. Returns
        whether or not a new note is received to avoid unnecessary retriggering. The envelope is
        updated with the new velocity value regardless.

        :param notenum: The MIDI note number representing the note frequency. If this parameter is
            not provided, the root frequency will be used.
        :param velocity: The strength at which the note was received, between 0.0 and 1.0. Although,
            velocity is not utilized by this voice.
        """
        if not super().press(1 if notenum is None else notenum, velocity):
            return False
        self.frequency = self._root if notenum is None else synthio.midi_to_hz(notenum)
        self._filter_envelope.press()
        self._filter_frequency.a.c.b.retrigger()
        self._amplitude.b.b.retrigger()  # Tremolo Delay
        self._bend.b.b.retrigger()  # Vibrato Delay
        return True

    def release(self) -> bool:
        """Release the voice if a note is currently being played. Returns `True` if a note was
        released and `False` if not.
        """
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
    def tune(self) -> float | tuple:
        """The amount of tuning from the root frequency of the oscillators (typically 440.0hz) in
        octaves. Ie: 1.0 = 880.hz, -2.0 = 110.0hz. To assign individual values for each drone voice,
        provide a tuple of float values. Defaults to 0.0.
        """
        return self._tune

    @tune.setter
    def tune(self, value: float | tuple) -> None:
        self._tune = value
        self._update_root()

    @property
    def detune(self) -> float | tuple:
        """The amount of detuning from the root frequency and tune of the oscillators (typically
        440.0hz) in octaves. Ie: 1.0 = 880.hz, -2.0 = 110.0hz. To assign individual values for each
        drone voice, provide a tuple of float values. Defaults to 0.0.
        """
        return self._detune

    @detune.setter
    def detune(self, value: float | tuple) -> None:
        self._detune = value
        self._update_root()

    @property
    def frequency(self) -> float:
        """The frequency in hertz to set the oscillators to. Updating this value will activate the
        frequency lerp block to gradually change the note frequency based on the glide settings of
        this voice.
        """
        return math.exp(self._freq_lerp.value * _LOG_2) * self._root

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._freq_lerp.value = math.log(value / self._root) / _LOG_2

    @property
    def glide(self) -> float:
        """The length of time it takes for the oscillators to "glide" (transition) between
        frequencies in seconds.
        """
        return self._freq_lerp.rate

    @glide.setter
    def glide(self, value: float) -> None:
        self._freq_lerp.rate = value

    @property
    def vibrato_rate(self) -> float:
        """The rate of the frequency LFO in hertz. Defaults to 1.0hz."""
        return self._bend.b.a.rate

    @vibrato_rate.setter
    def vibrato_rate(self, value: float) -> None:
        self._bend.b.a.rate = value

    @property
    def vibrato_depth(self) -> float:
        """The depth of the frequency LFO in octaves relative to the current note frequency and
        :attr:`bend`. Defaults to 0.0.
        """
        return self._bend.b.a.scale

    @vibrato_depth.setter
    def vibrato_depth(self, value: float) -> None:
        self._bend.b.a.scale = value

    @property
    def vibrato_delay(self) -> float:
        """The amount of time to gradually increase the depth of the frequency LFO in seconds. Must
        be greater than 0.0s. Defaults to 0.001s.
        """
        return 1 / self._bend.b.b.rate

    @vibrato_delay.setter
    def vibrato_delay(self, value: float) -> None:
        self._bend.b.b.rate = 1 / max(value, 0.001)

    @property
    def waveform(self) -> ReadableBuffer | None:
        """The waveform of the oscillators."""
        return self._waveform

    @waveform.setter
    def waveform(self, value: ReadableBuffer | None) -> None:
        self._waveform = value
        for note in self._notes:
            note.waveform = self._waveform

    @property
    def amplitude(self) -> float:
        """The relative amplitude of the oscillators from 0.0 to 1.0. An amplitude of 0 makes the
        oscillators inaudible. Defaults to 1 divided by the number of oscillators (ie: 0.333 if
        using 3 oscillators).
        """
        return self._amplitude.a

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude.a = value

    @property
    def tremolo_rate(self) -> float:
        """The rate of the amplitude LFO in hertz. Defaults to 1.0hz."""
        return self._amplitude.b.a.rate

    @tremolo_rate.setter
    def tremolo_rate(self, value: float) -> None:
        self._amplitude.b.a.rate = value

    @property
    def tremolo_depth(self) -> float:
        """The depth of the amplitude LFO. This value is added to :attr:`amplitude`. Defaults to
        0.0.
        """
        return self._amplitude.b.a.scale

    @tremolo_depth.setter
    def tremolo_depth(self, value: float) -> None:
        self._amplitude.b.a.scale = value

    @property
    def tremolo_delay(self) -> float:
        """The amount of time to gradually increase the depth of the amplitude LFO in seconds. Must
        be greater than 0.0s. Defaults to 0.001s.
        """
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
        """The rate of attack of the amplitude envelope to 1.0 after :meth:`press` is called in
        seconds. Must be greater than 0.0s. Defaults to 0.001s.
        """
        return self._envelope.attack_time

    @attack_time.setter
    def attack_time(self, value: float) -> None:
        self._attack_time = max(value, 0.0)
        self._update_envelope()

    @property
    def release_time(self) -> float:
        """The rate of decay of the amplitude envelope to 0.0 after :meth:`release` is called in
        seconds. Must be greater than 0.0s. Defaults to 0.001s.
        """
        return self._release_time

    @release_time.setter
    def release_time(self, value: float) -> None:
        self._release_time = max(value, 0.0)
        self._update_envelope()

    @property
    def filter_frequency(self) -> float:
        """The frequency of the filter in hertz. The maximum value allowed and default is half of
        the sample rate (the Nyquist frequency).
        """
        return self._filter_frequency.a.a

    @filter_frequency.setter
    def filter_frequency(self, value: float) -> None:
        self._filter_frequency.a.a = min(max(value, 1), self._synthesizer.sample_rate / 2)

    @property
    def filter_attack_time(self) -> float:
        """The rate of attack of the filter frequency envelope from :attr:`filter_frequency` to
        :attr:`filter_frequency` plus :attr:`filter_amount` in seconds. Must be greater than 0.0s.
        Defaults to 0.001s.
        """
        return self._filter_envelope.attack_time

    @filter_attack_time.setter
    def filter_attack_time(self, value: float) -> None:
        self._filter_envelope.attack_time = value

    @property
    def filter_amount(self) -> float:
        """The level to add to the :attr:`filter_frequency` in hertz after the filter envelope
        attack time has passed. This value will be sustained until :meth:`release` is called.
        Defaults to 0hz.
        """
        return self._filter_envelope.amount

    @filter_amount.setter
    def filter_amount(self, value: float) -> None:
        self._filter_envelope.amount = value

    @property
    def filter_release_time(self) -> float:
        """The rate of release of the filter frequency envelope back to :attr:`filter_frequency` in
        seconds. Must be greater than 0.0s. Defaults to 0.001s.
        """
        return self._filter_envelope.release_time

    @filter_release_time.setter
    def filter_release_time(self, value: float) -> None:
        self._filter_envelope.release_time = value

    @property
    def filter_rate(self) -> float:
        """The rate in hertz of the filter frequency LFO. Defaults to 1.0hz."""
        return self._filter_frequency.a.c.a.rate

    @filter_rate.setter
    def filter_rate(self, value: float) -> None:
        self._filter_frequency.a.c.a.rate = value

    @property
    def filter_depth(self) -> float:
        """The maximum level of the filter LFO to add to :attr:`filter_frequency` in hertz in both
        positive and negative directions. Defaults to 0.0hz.
        """
        return self._filter_frequency.a.c.a.scale

    @filter_depth.setter
    def filter_depth(self, value: float) -> None:
        self._filter_frequency.a.c.a.scale = value

    @property
    def filter_delay(self) -> float:
        """The amount of time to gradually increase the depth of the filter LFO in seconds. Must be
        greater than 0.0s. Defaults to 0.001s.
        """
        return 1 / self._filter_frequency.a.c.b.rate

    @filter_delay.setter
    def filter_delay(self, value: float) -> None:
        self._filter_frequency.a.c.b.rate = 1 / max(value, 0.001)
