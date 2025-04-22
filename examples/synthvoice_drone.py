# SPDX-FileCopyrightText: Copyright (c) 2025 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

import audiopwmio
import board
import synthio

from relic_synthvoice.drone import Drone

audio = audiopwmio.PWMAudioOut(board.A0)
synth = synthio.Synthesizer()
audio.play(synth)

voice = Drone(synth)
voice.press()
