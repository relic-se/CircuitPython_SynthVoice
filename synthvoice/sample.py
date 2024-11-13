# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import math
import time

import synthio
import synthwaveform
import ulab.numpy as np
import ulab.utils

import synthvoice.oscillator

_LOG_2 = math.log(2)


def is_pow2(value: float | int) -> bool:
    value = math.log(value) / _LOG_2
    return math.ceil(value) == math.floor(value)


def fft(data: np.ndarray, log: bool = True, length: int = 1024) -> np.ndarray:
    """Perform the Fourier Fast Transform (FFT) on data.

    :param data: The data to be processed, typically audio samples. The data type must be either
        :class:`ulab.numpy.int16` or :class:`ulab.numpy.uint16` or else a :class:`ValueError` will
        be raised.
    :param log: Use the logarithmic function on the output to convert the result to decibels.
        Defaults to `True`.
    :param length: The resulting length of the spectrogram array. A larger value will be more
        precise but require more processing and RAM usage.
    """
    if len(data) > length:
        offset = (len(data) - length) // 2
        data = data[offset : len(data) - offset]

    if data.dtype == np.uint16:
        mean = int(np.mean(data))
        data = np.array([x - mean for x in data], dtype=np.int16)
    elif data.dtype != np.int16:
        raise ValueError("Invalid data type")

    # Ensure that data length is a power of 2
    if len(data) < 2:
        return None
    if not is_pow2(len(data)):
        j = 2
        while True:
            j *= 2
            if j > len(data):
                data = data[: int(j // 2)]
                break

    data = ulab.utils.spectrogram(data)
    data = data[1 : (len(data) // 2) - 1]
    if log:
        data = np.log(data)
    return data


def fftfreq(data: np.ndarray, sample_rate: int):
    """Use the Fast Fourier Transform to determine the peak frequency of the signal.

    :param data: The data to be processed, typically audio samples. The data type must be either
        :class:`ulab.numpy.int16` or :class:`ulab.numpy.uint16` or else a :class:`ValueError` will
        be raised.
    :param sample_rate: The rate at which the data was recorded in hertz.
    """
    data = fft(data, log=False)
    freq = np.argmax(data) / len(data) * sample_rate / 4
    del data
    return freq


def normalize(data: np.ndarray) -> np.ndarray:
    """Scale the data so that it reaches the maximum peak capable of the data type (+32767 for
    :class:`ulab.numpy.int16`).

    :param data: The data to be normalized, typically audio samples. The data type must be
        :class:`ulab.numpy.int16` or else a :class:`ValueError` will be raised.
    """
    if data.dtype != np.int16:
        raise ValueError("Invalid data type")
    max_level = np.max(data)
    if max_level < 32767.0:
        data = np.array(
            np.clip(np.array(data, dtype=np.float) * 32767 / max_level, -32768, 32767),
            dtype=np.int16,
        )
    return data


class Sample(synthvoice.oscillator.Oscillator):
    """Voice which will play back an audio file. Handles pitch, looping points, and ".wav" file
    loading and inherits all properties and functionality of
    :class:`synthvoice.oscillator.Oscillator`.

    :param synthesizer: The :class:`synthio.Synthesizer` object this voice will be used with.
    :param looping: Whether or not to continuously loop the sample or play it once when the voice is
        pressed. Defaults to true.
    :param file: The path to the compatible audio file (16-bit integer ".wav"). Leave unset to
        initialize the voice without a specified sample. Defaults to `None`.
    :param max_size: The maximum number of samples to load into the waveform from the sample file.
    """

    looping: bool = True
    """Whether or not to continuously loop the sample or play it once when the voice is pressed."""

    def __init__(
        self,
        synthesizer: synthio.Synthesizer,
        looping: bool = True,
        file: str = None,
        max_size: int = 4096,
    ):
        super().__init__(synthesizer)

        self.looping = looping
        self._sample_rate = synthesizer.sample_rate
        self._sample_tune = 0.0
        self._loop_tune = 0.0
        self._start = None
        self._desired_frequency = self._root
        self._max_size = max_size

        if file:
            self.file = file
        else:
            self.file = None

    def _update_source_root(self) -> None:
        if self._note.waveform is not None:
            self._root = fftfreq(
                data=self._note.waveform,
                sample_rate=self._sample_rate,
            )
            self._cycle_duration = 1 / self._root
            self._source_duration = len(self._note.waveform) / self._sample_rate
            self._source_tune = math.log(self._cycle_duration / self._source_duration) / _LOG_2
        else:
            self._root = self._desired_frequency
            self._cycle_duration = 1.0 / self._root
            self._source_duration = 0.0
            self._source_tune = 0.0
        self._update_root()

    @property
    def sample_rate(self) -> int:
        """The recorded audio sample rate of the :attr:`waveform` data in hertz."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value
        self._update_source_root()

    @property
    def file(self) -> str | None:
        """The path to a 16-bit signed integer audio ".wav" file within the virtual file system.
        The audio sample rate and root frequency will automatically be calculated bye the file
        properties and with an FFT algorithm. An invalid file type will raise :class:`ValueError`.
        """
        return self._file

    @file.setter
    def file(self, value: str | None):
        if value is None:
            self._file = None
            self._note.waveform = None
        else:
            waveform, self.sample_rate = synthwaveform.from_wav(value, self._max_size)
            self.waveform = normalize(waveform)
        self._update_source_root()

    def press(self, notenum: int, velocity: float | int = 1.0) -> bool:
        """Update the voice to be "pressed" with a specific MIDI note number and velocity. Returns
        whether or not a new note is received to avoid unnecessary retriggering. The envelope is
        updated with the new velocity value regardless.

        :param notenum: The MIDI note number representing the note frequency.
        :param velocity: The strength at which the note was received, between 0.0 and 1.0. Defaults
            to 1.0. If an :class:`int` value is used, it will be divided by 127 assuming that it is
            a midi velocity value.
        """
        if self._note.waveform is None or not super().press(notenum, velocity):
            return False
        if not self.looping:
            self._start = time.monotonic()
        return True

    @property
    def duration(self) -> float:
        """The length of the audio sample given the current state (includes note bend
        properties).
        """
        return (
            self._source_duration
            * self._root
            / pow(2, self._note.bend.value)
            / self._desired_frequency
        )

    @property
    def waveform_loop(self) -> tuple[float, float]:
        """The start and stop points of which to loop the sample as a tuple of two floats from 0.0
        to 1.0. The end value must be greater than the start value. Default is (0.0, 1.0) or the
        full length of the sample.
        """
        return self._waveform_loop

    @waveform_loop.setter
    def waveform_loop(self, value: tuple[float, float]) -> None:
        self._set_waveform_loop(value)

        if not self._note.waveform:
            return

        length = self._note.waveform_loop_end - self._note.waveform_loop_start
        if length < 2:
            return

        sample_length = len(self._note.waveform)
        self._loop_tune = (
            math.log(sample_length / length) / _LOG_2 if length != sample_length else 0.0
        )
        self._update_root()

    def _update_root(self):
        super()._update_root()
        self._note.frequency = self._note.frequency * pow(2, self._source_tune + self._loop_tune)

    def update(self):
        """Update sample timing when :attr:`looping` is set to `False`."""
        if (
            not self.looping
            and not self._start is None
            and time.monotonic() - self._start >= self.duration
        ):
            self.release()
            self._start = None
