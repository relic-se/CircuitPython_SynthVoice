# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import adafruit_midi
import audiopwmio
import board
import digitalio
import synthio
import usb_midi
from adafruit_midi.note_off import NoteOff
from adafruit_midi.note_on import NoteOn

from relic_synthvoice.oscillator import Oscillator

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

audio = audiopwmio.PWMAudioOut(board.A0)
synth = synthio.Synthesizer()
audio.play(synth)

voice = Oscillator(synth)

midi = adafruit_midi.MIDI(
    midi_in=usb_midi.ports[0], in_channel=0, midi_out=usb_midi.ports[1], out_channel=0
)

while True:
    msg = midi.receive()
    if isinstance(msg, NoteOn) and msg.velocity != 0:
        led.value = True
        voice.press(msg.note, msg.velocity)
    elif isinstance(msg, NoteOff) or (isinstance(msg, NoteOn) and msg.velocity == 0):
        led.value = False
        voice.release()
