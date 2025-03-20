# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
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


class Oscillator(relic_synthvoice.Voice):
    """A complex single-voice Oscillator with the following features:
    - amplitude & filter envelopes
    - LFOs (low-frequency oscillators) for amplitude (tremolo), filter, pitch (vibrato), & panning
    - pitch glide
    - waveform looping

    :param synthesizer: The :class:`synthio.Synthesizer` object this voice will be used with.
    :param root: The root frequency used to calculate tuning. Defaults to 440.0hz. Changing this
        value will effect tuning properties.
    """

    def __init__(self, synthesizer: synthio.Synthesizer, root: float = 440.0):
        self._synthesizer = synthesizer

        self._notenum = -1
        self._velocity = 0.0

        self._velocity_amount = 1.0

        self._root = root
        self._coarse_tune = 0.0
        self._fine_tune = 0.0
        self._bend_range = 0.0
        self._bend = 0.0
        self._waveform_loop = (0.0, 1.0)

        self._freq_lerp = relic_synthvoice.LerpBlockInput(
            rate=0.0,
            value=0.0,
        )
        self._pitch_lerp = relic_synthvoice.LerpBlockInput(
            rate=0.0,
            value=0.0,
        )

        self._attack_time = 0.0
        self._attack_level = 1.0
        self._decay_time = 0.0
        self._sustain_level = 0.75
        self._release_time = 0.0

        self._note = synthio.Note(
            frequency=self._root,
            waveform=None,
            envelope=None,
            amplitude=synthio.Math(
                synthio.MathOperation.SUM,
                1.0,
                synthio.Math(
                    synthio.MathOperation.PRODUCT,
                    synthio.LFO(  # Tremolo LFO
                        waveform=None, rate=1.0, scale=0.0
                    ),
                    synthio.LFO(  # Tremolo Delay
                        waveform=np.array([0, 32767], dtype=np.int16),
                        rate=1 / 0.001,
                        once=True,
                    ),
                ),
                0.0,
            ),
            bend=synthio.Math(
                synthio.MathOperation.SUM,
                self._freq_lerp.block,  # Frequency Lerp
                synthio.Math(
                    synthio.MathOperation.PRODUCT,
                    synthio.LFO(  # Vibrato LFO
                        waveform=None, rate=1.0, scale=0.0, offset=0.0
                    ),
                    synthio.LFO(  # Vibrato Delay
                        waveform=np.array([0, 32767], dtype=np.int16),
                        rate=1 / 0.001,
                        once=True,
                    ),
                ),
                synthio.Math(
                    synthio.MathOperation.SUM,
                    synthio.LFO(  # Pitch Slew
                        waveform=np.array([32767, 0], dtype=np.int16),
                        rate=1 / 0.001,
                        scale=0.0,
                        offset=0.0,
                        once=True,
                    ),
                    self._pitch_lerp.block,  # Pitch Bend Lerp
                    0.0,
                ),
            ),
            panning=synthio.Math(
                synthio.MathOperation.PRODUCT,
                synthio.LFO(waveform=None, rate=1.0, scale=0.0, offset=0.0),  # Panning LFO
                synthio.LFO(  # Panning Delay
                    waveform=np.array([0, 32767], dtype=np.int16),
                    rate=1 / 0.001,
                    once=True,
                ),
            ),
        )
        self._update_envelope()

        self._filter_frequency = synthesizer.sample_rate / 2
        self._filter_resonance = 0.7071067811865475
        self._filter_envelope = relic_synthvoice.AREnvelope(
            attack_time=0.0,
            release_time=0.0,
            amount=0.0,
        )
        self._filter_frequency_block = synthio.Math(
            synthio.MathOperation.MAX,
            synthio.Math(
                synthio.MathOperation.SUM,
                self._filter_frequency,
                self._filter_envelope.block,
                synthio.Math(
                    synthio.MathOperation.PRODUCT,
                    synthio.LFO(  # Filter LFO
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
            50.0,  # Minimum allowed frequency
        )
        self.filter_mode = synthio.FilterMode.LOW_PASS  # constructs self._filter

        self._append_blocks()

    @property
    def notes(self) -> tuple[synthio.Note]:
        """Get all :class:`synthio.Note` objects attributed to this voice."""
        return tuple([self._note])

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        """Get all :class:`synthio.BlockInput` objects attributed to this voice."""
        return (
            self._filter_envelope.blocks
            + self._freq_lerp.blocks
            + self._pitch_lerp.blocks
            + (
                self._filter_frequency_block,
                self._filter_frequency_block.a,
                self._filter_frequency_block.a.c,
                self._filter_frequency_block.a.c.a,  # Filter LFO
                self._filter_frequency_block.a.c.b,  # Filter Delay
                self._note.amplitude,
                self._note.amplitude.b,
                self._note.amplitude.b.a,  # Tremolo LFO
                self._note.amplitude.b.b,  # Tremolo Delay
                self._note.bend,
                self._note.bend.b,
                self._note.bend.b.a,  # Vibrato LFO
                self._note.bend.b.b,  # Vibrato Delay
                self._note.bend.c,
                self._note.bend.c.a,  # Pitch Slew
                self._note.panning,
                self._note.panning.a,  # Panning LFO
                self._note.panning.b,  # Panning Delay
            )
        )

    def press(self, notenum: int, velocity: float | int = 1.0) -> bool:
        """Update the voice to be "pressed" with a specific MIDI note number and velocity. Returns
        whether or not a new note is received to avoid unnecessary retriggering. The envelope is
        updated with the new velocity value regardless.

        :param notenum: The MIDI note number representing the note frequency.
        :param velocity: The strength at which the note was received, between 0.0 and 1.0.
        """
        if not super().press(notenum, velocity):
            return False
        self.frequency = synthio.midi_to_hz(notenum)
        self._filter_envelope.press()
        self._filter_frequency_block.a.c.b.retrigger()
        self._note.amplitude.b.b.retrigger()  # Tremolo Delay
        self._note.bend.b.b.retrigger()  # Vibrato Delay
        self._note.bend.c.a.retrigger()  # Pitch Slew
        self._note.panning.b.retrigger()  # Panning Delay
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
        self._note.frequency = self._root * pow(2, self._coarse_tune + self._fine_tune / 12)

    @property
    def coarse_tune(self) -> float:
        """The amount of tuning from the root frequency of the oscillator (typically 440.0hz) in
        octaves. Ie: 1.0 = 880.hz, -2.0 = 110.0hz. Defaults to 0.0.
        """
        return self._coarse_tune

    @coarse_tune.setter
    def coarse_tune(self, value: float) -> None:
        self._coarse_tune = value
        self._update_root()

    @property
    def fine_tune(self) -> float:
        """The amount of tuning from the root frequency of the oscillator (typically 440.0hz) in
        semitones (1/12 of an octave). Ie: 1.0 = 466.16hz (A#4) and -1.0 = 415.30hz (G#4). Defaults
        to 0.0.
        """
        return self._fine_tune

    @fine_tune.setter
    def fine_tune(self, value: float) -> None:
        self._fine_tune = value
        self._update_root()

    @property
    def frequency(self) -> float:
        """The frequency in hertz to set the oscillator to. Updating this value will active the
        frequency lerp block to gradually change the note frequency based on the glide settings of
        this voice.
        """
        return self._freq_lerp.value

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._freq_lerp.value = math.log(value / self._root) / _LOG_2

    @property
    def glide(self) -> float:
        """The length of time it takes for the oscillator to "glide" (transition) between
        frequencies in seconds.
        """
        return self._freq_lerp.rate

    @glide.setter
    def glide(self, value: float) -> None:
        self._freq_lerp.rate = value

    def _update_pitch_bend(self):
        self._pitch_lerp.value = self._bend * self._bend_range

    @property
    def bend_range(self) -> float:
        """The maximum amount the oscillator frequency will "bend" when setting the :attr:`bend`
        property in octaves. Can be positive or negative, and the :attr:`bend` property can scale
        this range in both directions. Defaults to 0.0

        Example settings:
        - 2.0 = up two octaves
        - -1.0 = down one octave
        - 1.0/12.0 = up one semitone (chromatic note)
        """
        return self._bend_range

    @bend_range.setter
    def bend_range(self, value: float) -> None:
        self._bend_range = value
        self._update_pitch_bend()

    @property
    def bend(self) -> float:
        """The pitch bend value which changes the oscillator frequency by a relative amount.
        Positive and negative range is defined by :attr:`bend_range`. Defaults to 0.0.
        """
        return self._bend

    @bend.setter
    def bend(self, value: float) -> None:
        self._bend = value
        self._update_pitch_bend()

    @property
    def pitch_slew_time(self) -> float:
        """The amount of time in seconds it takes for the voice to reach the desired pitch after
        starting with a relative :attr:`pitch_slew` adjustment. Must be greater than 0.0s. Defaults
        to 0.001s.
        """
        return 1 / self._note.bend.c.a.rate

    @pitch_slew_time.setter
    def pitch_slew_time(self, value: float) -> None:
        self._note.bend.c.a.rate = 1 / max(value, 0.001)

    @property
    def pitch_slew(self) -> float:
        """The pitch offset in octaves at which the voice starts relative to the desired frequency
        when first pressed. Can be either positive or negative. Defaults to 0.0.
        """
        return self._note.bend.c.a.scale

    @pitch_slew.setter
    def pitch_slew(self, value: float) -> None:
        self._note.bend.c.a.scale = value

    @property
    def vibrato_rate(self) -> float:
        """The rate of the frequency LFO in hertz. Defaults to 1.0hz."""
        return self._note.bend.b.a.rate

    @vibrato_rate.setter
    def vibrato_rate(self, value: float) -> None:
        self._note.bend.b.a.rate = value

    @property
    def vibrato_depth(self) -> float:
        """The depth of the frequency LFO in octaves relative to the current note frequency and
        :attr:`bend`. Defaults to 0.0.
        """
        return self._note.bend.b.a.scale

    @vibrato_depth.setter
    def vibrato_depth(self, value: float) -> None:
        self._note.bend.b.a.scale = value

    @property
    def vibrato_delay(self) -> float:
        """The amount of time to gradually increase the depth of the frequency LFO in seconds. Must
        be greater than 0.0s. Defaults to 0.001s.
        """
        return 1 / self._note.bend.b.b.rate

    @vibrato_delay.setter
    def vibrato_delay(self, value: float) -> None:
        self._note.bend.b.b.rate = 1 / max(value, 0.001)

    @property
    def waveform(self) -> ReadableBuffer | None:
        """The waveform of the oscillator."""
        return self._note.waveform

    @waveform.setter
    def waveform(self, value: ReadableBuffer | None) -> None:
        self._note.waveform = value
        self._apply_waveform_loop()

    def _set_waveform_loop(self, value: tuple[float, float]) -> None:
        start = min(max(value[0], 0.0), 1.0)
        end = min(max(value[1], start), 1.0)
        self._waveform_loop = (start, end)
        self._apply_waveform_loop()

    def _apply_waveform_loop(self) -> None:
        if self._note.waveform is not None and len(self._note.waveform) >= 2:
            waveform_length = len(self._note.waveform)
            self._note.waveform_loop_start = min(
                max(int(self._waveform_loop[0] * waveform_length), 0), waveform_length - 2
            )
            self._note.waveform_loop_end = min(
                max(
                    int(self._waveform_loop[1] * waveform_length),
                    self._note.waveform_loop_start + 2,
                ),
                waveform_length,
            )

    @property
    def waveform_loop(self) -> tuple[float, float]:
        """The start and stop points of which to loop waveform data as a tuple of two floats from
        0.0 to 1.0. The end value must be greater than the start value. Default is (0.0, 1.0) or the
        full range of the waveform.
        """
        return self._waveform_loop

    @waveform_loop.setter
    def waveform_loop(self, value: tuple[float, float]) -> None:
        self._set_waveform_loop(value)

    @property
    def amplitude(self) -> float:
        """The relative amplitude of the oscillator from 0.0 to 1.0. An amplitude of 0 makes the
        oscillator inaudible. Defaults to 1.0.
        """
        return self._note.amplitude.a

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._note.amplitude.a = value

    @property
    def tremolo_rate(self) -> float:
        """The rate of the amplitude LFO in hertz. Defaults to 1.0hz."""
        return self._note.amplitude.b.a.rate

    @tremolo_rate.setter
    def tremolo_rate(self, value: float) -> None:
        self._note.amplitude.b.a.rate = value

    @property
    def tremolo_depth(self) -> float:
        """The depth of the amplitude LFO. This value is added to :attr:`amplitude`. Defaults to
        0.0.
        """
        return self._note.amplitude.b.a.scale

    @tremolo_depth.setter
    def tremolo_depth(self, value: float) -> None:
        self._note.amplitude.b.a.scale = value

    @property
    def tremolo_delay(self) -> float:
        """The amount of time to gradually increase the depth of the amplitude LFO in seconds. Must
        be greater than 0.0s. Defaults to 0.001s.
        """
        return 1 / self._note.amplitude.b.b.rate

    @tremolo_delay.setter
    def tremolo_delay(self, value: float) -> None:
        self._note.amplitude.b.b.rate = 1 / max(value, 0.001)

    @property
    def pan(self) -> float:
        """The distribution of the oscillator amplitude in the channel(s) output from -1.0 (left) to
        1.0 (right). Defaults to 0.0.
        """
        return self._note.panning.a.offset

    @pan.setter
    def pan(self, value: float) -> None:
        self._note.panning.a.offset = value

    @property
    def pan_rate(self) -> float:
        """The rate of the panning LFO in hertz. Defaults to 1.0."""
        return self._note.panning.a.rate

    @pan_rate.setter
    def pan_rate(self, value: float) -> None:
        self._note.panning.a.rate = value

    @property
    def pan_depth(self) -> float:
        """The depth of the panning LFO from 0.0 to 1.0. This value is added to :attr:`pan`.
        Negative values are allowed and will flip the phase of the LFO. Defaults to 0.0.
        """
        return self._note.panning.a.scale

    @pan_depth.setter
    def pan_depth(self, value: float) -> None:
        self._note.panning.a.scale = value

    @property
    def pan_delay(self) -> float:
        """The amount of time to gradually increase the depth of the panning LFO in seconds. Must be
        greater than 0.0s. Defaults to 0.001s.
        """
        return 1 / self._note.panning.b.rate

    @pan_delay.setter
    def pan_delay(self, value: float) -> None:
        self._note.panning.b.rate = 1 / max(value, 0.001)

    def _update_envelope(self):
        mod = self._get_velocity_mod()
        self._note.envelope = synthio.Envelope(
            attack_time=self._attack_time,
            attack_level=mod * self._attack_level,
            decay_time=self._decay_time,
            sustain_level=mod * self._sustain_level,
            release_time=self._release_time,
        )

    @property
    def attack_time(self) -> float:
        """The rate of attack of the amplitude envelope in seconds. Must be greater than 0.0s.
        Defaults to 0.001s.
        """
        return self._attack_time

    @attack_time.setter
    def attack_time(self, value: float) -> None:
        self._attack_time = max(value, 0.0)
        self._update_envelope()

    @property
    def attack_level(self) -> float:
        """The level that the amplitude envelope will reach after the attack time has passed as a
        relative value of :attr:`amplitude` from 0.0 to 1.0. Defaults to 1.0.
        """
        return self._attack_level

    @attack_level.setter
    def attack_level(self, value: float) -> None:
        self._attack_level = min(max(value, 0.0), 1.0)
        self._update_envelope()

    @property
    def decay_time(self) -> float:
        """The rate of decay after reaching the :attr:`attack_level` of the amplitude envelope in
        seconds. Must be greater than 0.0s. Defaults to 0.001s.
        """
        return self._decay_time

    @decay_time.setter
    def decay_time(self, value: float) -> None:
        self._decay_time = max(value, 0.0)
        self._update_envelope()

    @property
    def sustain_level(self) -> float:
        """The level that the amplitude envelope will reach after the decay time has passed as a
        relative value of :attr:`amplitude` from 0.0 to 1.0. The note will sustain with this level
        until :meth:`release` is called. Defaults to 0.75.
        """
        return self._sustain_level

    @sustain_level.setter
    def sustain_level(self, value: float) -> None:
        self._sustain_level = min(max(value, 0.0), 1.0)
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

    def _update_filter(
        self, mode: synthio.FilterMode = None, biquad: synthio.BlockBiquad = None
    ) -> None:
        if mode is None:
            mode = self.filter_mode
        if biquad is None:
            biquad = synthio.BlockBiquad(mode, self._filter_frequency_block, self._filter_resonance)
        super()._update_filter(mode, biquad)

    def _update_filter_frequency(self) -> None:
        self._filter_frequency_block.a.a = self._filter_frequency

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
        return self._filter_frequency_block.a.c.a.rate

    @filter_rate.setter
    def filter_rate(self, value: float) -> None:
        self._filter_frequency_block.a.c.a.rate = value

    @property
    def filter_depth(self) -> float:
        """The maximum level of the filter LFO to add to :attr:`filter_frequency` in hertz in both
        positive and negative directions. Defaults to 0.0hz.
        """
        return self._filter_frequency_block.a.c.a.scale

    @filter_depth.setter
    def filter_depth(self, value: float) -> None:
        self._filter_frequency_block.a.c.a.scale = value

    @property
    def filter_delay(self) -> float:
        """The amount of time to gradually increase the depth of the filter LFO in seconds. Must be
        greater than 0.0s. Defaults to 0.001s.
        """
        return 1 / self._filter_frequency_block.a.c.b.rate

    @filter_delay.setter
    def filter_delay(self, value: float) -> None:
        self._filter_frequency_block.a.c.b.rate = 1 / max(value, 0.001)
