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

from relic_synthvoice.sample import Sample

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

synth = synthio.Synthesizer(sample_rate=44100)

voice = Sample(synth, file="/test.wav")
voice.waveform_loop = (0.65, 0.96)
voice.release_time = 0.5

# Start up audio output after loading sample file to avoid file background tasks interrupting
audio = audiopwmio.PWMAudioOut(board.A0)
audio.play(synth)

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
    voice.update()
