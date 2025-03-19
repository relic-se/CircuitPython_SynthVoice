# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import synthio
import relic_waveform
import ulab.numpy as np

import synthvoice

try:
    from circuitpython_typing import ReadableBuffer
except ImportError:
    pass


class Voice(synthvoice.Voice):
    """Base single-shot "analog" drum voice used by other classes within the percussive module.
    Handles envelope times, tuning, waveforms, etc. for multiple :class:`synthio.Note` objects.

    :param count: The number of :class:`synthio.Note` objects to generate. Defaults to 3.
    :param filter_mode: The type of filter to use. Defaults to :const:`synthio.FilterMode.LOW_PASS`.
    :param filter_frequency: The exact frequency of the filter of all :class:`synthio.Note` objects
        in hertz. Defaults to 20000hz.
    :param frequencies: A list of the frequencies corresponding to each :class:`synthio.Note` object
        in hertz. Voice doesn't respond to the note frequency when pressed and instead uses these
        constant frequencies. Defaults to 440.0hz if not provided.
    :param times: A list of decay times corresponding to each :class:`synthio.Note` objects'
        amplitude envelope in seconds. Defaults to 1.0s for all notes if not provided.
    :param waveforms: A list of waveforms corresponding to each :class:`synthio.Note` object as
        :class:`numpy.int16` arrays. Defaults to a square waveform for each note.
    """

    def __init__(  # noqa: PLR0913
        self,
        synthesizer: synthio.Synthesizer,
        count: int = 3,
        filter_mode: synthio.FilterMode = synthio.FilterMode.LOW_PASS,
        filter_frequency: float = 20000.0,
        frequencies: tuple[float] = [],
        times: tuple[float] = [],
        waveforms: tuple[ReadableBuffer] | ReadableBuffer = [],
    ):
        if not frequencies:
            frequencies = tuple([440.0])
        if not times:
            times = tuple([1.0])

        self._times = times
        self._attack_level = 1.0
        self._decay_time = 0.0

        self._lfo = synthio.LFO(
            waveform=np.array([32767, -32768], dtype=np.int16),
            rate=20.0,
            scale=0.3,
            offset=0.33,
            once=True,
        )

        self._frequencies = frequencies
        self._tune = 0.0

        self._notes = tuple(
            [
                synthio.Note(frequency=frequencies[i % len(frequencies)], bend=self._lfo)
                for i in range(count)
            ]
        )
        
        super().__init__(synthesizer)

        self.times = times
        self.waveforms = waveforms

        self.filter_frequency = filter_frequency
        self.filter_mode = filter_mode

    @property
    def notes(self) -> tuple[synthio.Note]:
        """Get all :class:`synthio.Note` objects attributed to this voice."""
        return self._notes

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        """Get all :class:`synthio.BlockInput` objects attributed to this voice."""
        return tuple([self._lfo])

    def _update_frequencies(self) -> None:
        for i, note in enumerate(self.notes):
            note.frequency = self._frequencies[i % len(self._frequencies)] * pow(2, self._tune / 12)

    @property
    def frequencies(self) -> tuple[float]:
        """The base frequencies in hertz."""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value: tuple[float] | float) -> None:
        if not isinstance(value, tuple):
            value = tuple([value])
        if value:
            self._frequencies = value
            self._update_frequencies()

    @property
    def tune(self) -> float:
        """The amount of tuning form the root frequencies of the voice in semitones (1/12 of an
        octave). Defaults to 0.0.
        """

    @tune.setter
    def tune(self, value: float) -> None:
        self._tune = value
        self._update_frequencies()

    @property
    def times(self) -> tuple[float]:
        """The decay times of the amplitude envelopes."""
        return self._times

    @times.setter
    def times(self, value: tuple[float] | float) -> None:
        if not isinstance(value, tuple):
            value = tuple([value])
        if value:
            self._times = value
            self._update_envelope()

    @property
    def waveforms(self) -> tuple[ReadableBuffer]:
        """The note waveforms as :class:`ulab.numpy.ndarray` objects with the
        :class:`ulab.numpy.int16` data type.
        """
        value = []
        for note in self.notes:
            value.append(note.waveform)
        return tuple(value)

    @waveforms.setter
    def waveforms(self, value: tuple[ReadableBuffer] | ReadableBuffer) -> None:
        if not value:
            return
        if not isinstance(value, tuple):
            value = tuple([value])
        for i, note in enumerate(self.notes):
            note.waveform = value[i % len(value)]

    def press(self, velocity: float | int = 1.0) -> bool:
        """Update the voice to be "pressed". For percussive voices, this will begin the playback of
        the voice.

        :param velocity: The strength at which the note was received, between 0.0 and 1.0.
        """
        super().release()
        super().press(1, velocity)
        self._lfo.retrigger()
        return True

    def release(self) -> bool:
        """Release the voice. :class:`synthvoice.percussive.Voice` objects typically don't implement
        this operation because of their "single-shot" nature and will always return `False`.
        """
        return False

    @property
    def amplitude(self) -> float:
        """The volume of the voice from 0.0 to 1.0."""
        return self.notes[0].amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        for note in self.notes:
            note.amplitude = min(max(value, 0.0), 1.0)

    @property
    def pan(self) -> float:
        """The stereo panning of the voice from -1.0 (left) to 1.0 (right)."""
        return self.notes[0].panning

    @pan.setter
    def pan(self, value: float) -> None:
        value = min(max(value, -1.0), 1.0)
        for note in self.notes:
            note.panning = value

    def _update_envelope(self) -> None:
        mod = self._get_velocity_mod()
        for i, note in enumerate(self.notes):
            note.envelope = synthio.Envelope(
                attack_time=0.0,
                decay_time=self._times[i % len(self._times)] * pow(2, self._decay_time),
                release_time=0.0,
                attack_level=mod * self._attack_level,
                sustain_level=0.0,
            )

    @property
    def attack_level(self) -> float:
        """The level of attack of the amplitude envelope from 0.0 to 1.0."""
        return self._attack_level

    @attack_level.setter
    def attack_level(self, value: float) -> None:
        self._attack_level = min(max(value, 0.0), 1.0)
        self._update_envelope()

    @property
    def decay_time(self) -> float:
        """The amount of decay of the amplitude envelope relative to the initial decay time. 0.0 is
        the default amount of decay, 1.0 is double the decay, and -1.0 is half the decay.
        """
        return self._decay_time

    @decay_time.setter
    def decay_time(self, value: float) -> None:
        self._decay_time = value
        self._update_envelope()


class Kick(Voice):
    """A single-shot "analog" drum voice representing a low frequency sine-wave kick drum."""

    def __init__(self, synthesizer: synthio.Synthesizer):
        sine = relic_waveform.sine()
        offset_sine = relic_waveform.sine(phase=0.5)
        super().__init__(
            synthesizer,
            count=3,
            filter_frequency=2000.0,
            frequencies=(53.0, 72.0, 41.0),
            times=(0.075, 0.055, 0.095),
            waveforms=(offset_sine, sine, offset_sine),
        )


class Snare(Voice):
    """A single-shot "analog" drum voice representing a snare drum using sine and noise
    waveforms.
    """

    def __init__(self, synthesizer: synthio.Synthesizer):
        sine_noise = relic_waveform.mix(
            relic_waveform.sine(),
            (relic_waveform.noise(), 0.5),
        )
        offset_sine_noise = relic_waveform.mix(
            relic_waveform.sine(phase=0.5),
            (relic_waveform.noise(), 0.5),
        )
        super().__init__(
            synthesizer,
            count=3,
            filter_frequency=9500.0,
            frequencies=(90.0, 135.0, 165.0),
            times=(0.115, 0.095, 0.115),
            waveforms=(sine_noise, offset_sine_noise, offset_sine_noise),
        )


class Cymbal(Voice):
    """The base class to create cymbal sounds with variable timing.

    :param min_time: The minimum decay time in seconds. Must be greater than 0.0s.
    :param max_time: The maximum decay time in seconds. Must be greater than min_time.
    """

    def __init__(
        self,
        synthesizer: synthio.Synthesizer,
        time: float,
        frequency: float = 9500.0,
    ):
        super().__init__(
            synthesizer,
            count=3,
            filter_mode=synthio.FilterMode.HIGH_PASS,
            filter_frequency=frequency,
            frequencies=(90, 135, 165.0),
            waveforms=relic_waveform.noise(),
            times=(time, max(time - 0.02, 0.001), time),
        )


class ClosedHat(Cymbal):
    """A single-shot "analog" drum voice representing a closed hi-hat cymbal using noise
    waveforms.
    """

    def __init__(self, synthesizer: synthio.Synthesizer):
        super().__init__(synthesizer, 0.1125)


class OpenHat(Cymbal):
    """A single-shot "analog" drum voice representing an open hi-hat cymbal using noise
    waveforms.
    """

    def __init__(self, synthesizer: synthio.Synthesizer):
        super().__init__(synthesizer, 0.625)


class Ride(Cymbal):
    """A single-shot "analog" drum voice representing a ride cymbal using noise waveforms."""

    def __init__(self, synthesizer: synthio.Synthesizer):
        super().__init__(synthesizer, 1.25, 18000.0)


class Tom(Voice):
    """The base class to create tom drum sounds with variable timing and frequency.

    :param min_time: The minimum decay time in seconds. Must be greater than 0.0s.
    :param max_time: The maximum decay time in seconds. Must be greater than min_time.
    :param min_frequency: The minimum frequency in hertz.
    :param max_frequency: The maximum frequency in hertz.
    """

    def __init__(  # noqa: PLR0913
        self,
        synthesizer: synthio.Synthesizer,
        time: float,
        frequency: float,
    ):
        super().__init__(
            synthesizer,
            count=2,
            filter_frequency=4000.0,
            waveforms=(relic_waveform.triangle(), relic_waveform.noise(amplitude=0.25)),
            times=(time, 0.025),
            frequencies=tuple([frequency]),
        )


class HighTom(Tom):
    """A single-shot "analog" drum voice representing a high or left rack tom drum."""

    def __init__(self, synthesizer: synthio.Synthesizer):
        super().__init__(synthesizer, 0.275, 277.645)


class MidTom(Tom):
    """A single-shot "analog" drum voice representing a middle or right rack tom drum."""

    def __init__(self, synthesizer: synthio.Synthesizer):
        super().__init__(synthesizer, 0.275, 196.325)


class FloorTom(Tom):
    """A single-shot "analog" drum voice representing a low or floor tom drum."""

    def __init__(self, synthesizer: synthio.Synthesizer):
        super().__init__(synthesizer, 0.375, 131.685)
