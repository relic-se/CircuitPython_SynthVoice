# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT
"""
`relic_synthvoice`
================================================================================

Advanced synthio voices


* Author(s): Cooper Dalrymple

Implementation Notes
--------------------

**Hardware:**

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

* CircuitPython Waveform library:
  https://github.com/relic-se/CircuitPython_Waveform
"""

# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/relic-se/CircuitPython_SynthVoice.git"

import synthio
import ulab.numpy as np


class LerpBlockInput:
    """Creates and manages a :class:`synthio.BlockInput` object to "lerp" (linear interpolation)
    between an old value and a new value. Useful for note frequency "glide" and custom envelopes.

    :param rate: The speed at which to go between values, in seconds. Must be greater than 0.0s.
        Defaults to 0.05s.
    :param value: The initial value. Defaults to 0.0.
    """

    def __init__(self, rate: float = 0.05, value: float = 0.0):
        """Constructor method"""
        self._position = synthio.LFO(
            waveform=np.linspace(-16385, 16385, num=2, dtype=np.int16),
            rate=1 / max(rate, 0.001),
            scale=1,
            offset=0.5,
            once=True,
        )
        self._lerp = synthio.Math(
            synthio.MathOperation.CONSTRAINED_LERP, value, value, self._position
        )

    @property
    def block(self) -> synthio.BlockInput:
        """Get the block input to be used with a :class:`synthio.Note` object."""
        return self._lerp

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        """Get all :class:`synthio.BlockInput` objects. In order for it to function properly, these
        blocks must be added to the primary :class:`synthio.Synthesizer` object using
        synth.blocks.append(...).
        """
        return (self._position, self._lerp)

    @property
    def value(self) -> float:
        """Get the current value of the linear interpolation output or set a new value to begin
        interpolation to from the current value state. Causes the interpolation process to
        retrigger.
        """
        return self._lerp.value

    @value.setter
    def value(self, value: float) -> None:
        self._lerp.a = self._lerp.value
        self._lerp.b = value
        self._position.retrigger()

    @property
    def rate(self) -> float:
        """The rate of change of interpolation in seconds. Must be greater than 0.001s."""
        return 1 / self._position.rate

    @rate.setter
    def rate(self, value: float) -> None:
        self._position.rate = 1 / max(value, 0.001)


class AREnvelope:
    """A simple attack, sustain and release envelope using linear interpolation. Useful for
    controlling parameters of a :class:`synthio.Note` object other than amplitude which accept
    :class:`synthio.BlockInput` values.

    :param attack: The amount of time to go from 0.0 to the specified amount in seconds when the
        envelope is pressed. Must be greater than 0.0s. Default is 0.05s.
    :param release: The amount of time to go from the specified amount back to 0.0 in seconds when
        the envelope is released. Must be greater than 0.0s. Default is 0.05s
    :param amount: The level at which to rise or fall to when the envelope is pressed. Value is
        arbitrary and can be positive or negative, but 0.0 will result in no change. Default is 1.0.
    """

    def __init__(self, attack_time: float = 0.05, release_time: float = 0.05, amount: float = 1.0):
        self._pressed = False
        self._lerp = LerpBlockInput()
        self._attack_time = attack_time
        self._release_time = release_time
        self._amount = amount

    @property
    def block(self) -> synthio.BlockInput:
        """Get the :class:`synthio.BlockInput` object to be applied to a parameter."""
        return self._lerp.block

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        """Get all :class:`synthio.BlockInput` objects. In order for it to function properly, these
        blocks must be added to the primary :class:`synthio.Synthesizer` object using
        synth.blocks.append(...).
        """
        return self._lerp.blocks

    @property
    def value(self) -> float:
        """Get the current value of the envelope."""
        return self._lerp.value

    @property
    def pressed(self) -> bool:
        """Whether or not the envelope is currently in a "pressed" state."""
        return self._pressed

    @property
    def attack_time(self) -> float:
        """The rate of attack in seconds. When changing if the envelope is currently in the attack
        state, it will update the rate immediately. Must be greater than 0.0s.
        """
        return self._attack_time

    @attack_time.setter
    def attack_time(self, value: float) -> None:
        self._attack_time = value
        if self._pressed:
            self._lerp.rate = self._attack_time

    @property
    def release_time(self) -> float:
        """The rate of release in seconds. If the envelope is currently in the release state, it
        will update the rate immediately. Must be greater than 0.0s.
        """
        return self._release_time

    @release_time.setter
    def release_time(self, value: float) -> None:
        self._release_time = value
        if not self._pressed:
            self._lerp.rate = self._release_time

    @property
    def amount(self) -> float:
        """The level at which to rise or fall to when the envelope is pressed (or sustained value).
        If the envelope is currently in the attack/press state, the targeted value will be updated
        immediately. Valu7e is arbitrary and can be positive or negative, but 0.0 will result in no
        change.
        """
        return self._amount

    @amount.setter
    def amount(self, value: float) -> None:
        self._amount = value
        if self._pressed:
            self._lerp.value = self._amount

    def press(self):
        """Active the envelope by setting it into the "pressed" state. The envelope's attack phase
        will start immediately.
        """
        self._pressed = True
        self._lerp.rate = self._attack_time
        self._lerp.value = self._amount

    def release(self):
        """Deactivate the envelope by setting it into the "released" state. The envelope's release
        phase will start immediately.
        """
        self._lerp.rate = self._release_time
        self._lerp.value = 0.0
        self._pressed = False


class Voice:
    """A "voice" to be used with a :class:`synthio.Synthesizer` object. Manages one or multiple
    :class:`synthio.Note` objects.

    The standard :class:`Voice` class is not meant to be used directly but instead inherited by one
    of the provided voice classes or within a custom class. This class helps manage note frequency,
    velocity, and filter state and provides an interface with a :class:`synthio.Synthesizer` object.

    :param synthesizer: The :class:`synthio.Synthesizer` object this voice will be used with.
    """

    def __init__(self, synthesizer: synthio.Synthesizer):
        self._synthesizer = synthesizer

        self._notenum = -1
        self._velocity = 0.0

        self._velocity_amount = 1.0

        self._update_biquad()

    def _update_biquad(
        self,
        mode: synthio.FilterMode = synthio.FilterMode.LOW_PASS,
        frequency: any = None,
        Q: float = 0.7071067811865475,
    ) -> None:
        if frequency is None:
            frequency = self._synthesizer.sample_rate / 2

        if hasattr(synthio, "BlockBiquad"):
            self._biquad = synthio.BlockBiquad(mode, frequency, Q)
        else:
            self._biquad = synthio.Biquad(mode, frequency, Q)

        for note in self.notes:
            note.filter = self._biquad

    def _append_blocks(self) -> None:
        for block in self.blocks:
            self._synthesizer.blocks.append(block)

    @property
    def notes(self) -> tuple[synthio.Note]:
        """Get all :class:`synthio.Note` objects attributed to this voice."""
        return tuple()

    @property
    def blocks(self) -> tuple[synthio.BlockInput]:
        """Get all :class:`synthio.BlockInput` objects attributed to this voice."""
        return tuple()

    def press(self, notenum: int, velocity: float | int = 1.0) -> bool:
        """Update the voice to be "pressed" with a specific MIDI note number and velocity. Returns
        whether or not a new note is received to avoid unnecessary retriggering. The envelope is
        updated with the new velocity value regardless. Updating :class:`synthio.Note` objects
        should typically occur within the child class after calling this method and checking its
        return value.

        :param notenum: The MIDI note number representing the note frequency.
        :param velocity: The strength at which the note was received, between 0.0 and 1.0. Defaults
            to 1.0. If an :class:`int` value is used, it will be divided by 127 assuming that it is
            a midi velocity value.
        """
        if type(velocity) is int:
            velocity /= 127
        self._velocity = velocity
        self._update_envelope()
        if notenum == self._notenum:
            return False
        self._notenum = notenum
        self._synthesizer.press(self.notes)
        return True

    def release(self) -> bool:
        """Release the voice if a note is currently being played. Returns `True` if a note was
        released and `False` if not.
        """
        if not self.pressed:
            return False
        self._notenum = 0
        self._synthesizer.release(self.notes)
        return True

    @property
    def pressed(self) -> bool:
        """Whether or not the voice is currently in a "pressed" state."""
        return self._notenum > 0

    @property
    def amplitude(self) -> float:
        """The volume of the voice from 0.0 to 1.0. This method should be implemented within the
        child class.
        """
        pass

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        pass

    def _get_velocity_mod(self) -> float:
        return 1.0 - (1.0 - min(max(self._velocity, 0.0), 1.0)) * self._velocity_amount

    @property
    def velocity_amount(self) -> float:
        """The amount that this voice will respond to note velocity, from 0.0 to 1.0. A value of 0.0
        represents no response to velocity. The voice will be at full level regardless of note
        velocity. Whereas a value of 1.0 represents full response to velocity.
        """
        return self._velocity_amount

    @velocity_amount.setter
    def velocity_amount(self, value: float) -> None:
        self._velocity_amount = min(max(value, 0.0), 1.0)

    def _update_envelope(self) -> None:
        pass

    @property
    def filter_mode(self) -> synthio.FilterMode:
        """The type of the filter. Defaults to :const:`synthio.FilterMode.LOW_PASS`."""
        return self._biquad.mode

    @filter_mode.setter
    def filter_mode(self, value: synthio.FilterMode) -> None:
        self._update_biquad(value, self.filter_frequency, self.filter_resonance)

    @property
    def filter_frequency(self) -> float:
        """The frequency of the filter in hertz. The maximum value allowed and default is half of
        the sample rate (the Nyquist frequency).
        """
        return self._biquad.frequency

    @filter_frequency.setter
    def filter_frequency(self, value: float) -> None:
        self._biquad.frequency = min(max(value, 1), self._synthesizer.sample_rate / 2)

    @property
    def filter_resonance(self) -> float:
        """The resonance of the filter (or Q factor) as a number starting from 0.7. Defaults to
        0.7.
        """
        return self._biquad.Q

    @filter_resonance.setter
    def filter_resonance(self, value: float) -> None:
        self._biquad.Q = max(value, 0.7071067811865475)

    def update(self) -> None:
        """Update all time-based voice logic controlled outside of synthio such as filter
        modulation.
        """
        pass
