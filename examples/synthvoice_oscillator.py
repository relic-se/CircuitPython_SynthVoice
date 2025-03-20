# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import adafruit_midi
import audiopwmio
import board
import digitalio
import relic_waveform
import synthio
import usb_midi
from adafruit_midi.note_off import NoteOff
from adafruit_midi.note_on import NoteOn

from relic_synthvoice.oscillator import Oscillator

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

audio = audiopwmio.PWMAudioOut(board.A0, board.A1)
synth = synthio.Synthesizer(sample_rate=44100, channel_count=2)
audio.play(synth)

voice = Oscillator(synth)
voice.waveform = relic_waveform.mix(
    relic_waveform.saw(),
    relic_waveform.saw(frequency=2),
)

# Frequency
voice.glide = 0.5
voice.coarse_tune = -1
voice.fine_tune = 5
voice.vibrato_depth = 1 / 12
voice.vibrato_rate = 8.0
voice.pitch_slew = -2.0
voice.pitch_slew_time = 0.2

# Envelope
voice.attack_time = 0.0
voice.attack_level = 1.0
voice.decay_time = 0.75
voice.sustain_level = 0.5
voice.release_time = 1.0

# Amplitude
voice.amplitude = 0.75
voice.tremolo_depth = 0.1
voice.tremolo_rate = 2.0

# Filter
voice.filter_frequency = 200
voice.filter_resonance = 1.75
voice.filter_attack_time = 0.25
voice.filter_amount = 1200
voice.filter_release_time = 0.75
voice.filter_rate = 0.5
voice.filter_depth = 1000

# Panning
voice.pan_depth = 1.0
voice.pan_rate = 0.25

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
