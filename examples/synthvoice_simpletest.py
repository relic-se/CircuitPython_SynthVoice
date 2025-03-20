# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import time

import audiopwmio
import board
import digitalio
import synthio

from relic_synthvoice.oscillator import Oscillator

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

audio = audiopwmio.PWMAudioOut(board.A0)
synth = synthio.Synthesizer()
audio.play(synth)

voice = Oscillator(synth)

while True:
    led.value = True
    voice.press(60)
    time.sleep(0.5)
    led.value = False
    voice.release()
    time.sleep(0.5)
