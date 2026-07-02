"""Runtime patches for mlx-audio 0.4.4 bugs. Import before synthesis.

Bug: kokoro/istftnet.py SineGen -- _f02sine round-trips the f0 track through
interpolate(scale_factor=1/300) then interpolate(scale_factor=300). The
helper computes output size as ceil(L * scale_factor); float error can push
L/300 just above an integer (e.g. 37800 * (1/300) == 126.00000000000001),
so the sine track comes back one 300-sample frame longer than f0 and
broadcasting against the uv mask fails:
  ValueError: [broadcast_shapes] Shapes (1,37800,1) and (1,38100,9) ...

Fix: trim/align the sine track to f0's length in SineGen.__call__.
"""

import mlx.core as mx

from mlx_audio.tts.models.kokoro import istftnet


def _sinegen_call(self, f0):
    fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
    sine_waves = self._f02sine(fn) * self.sine_amp

    # align to f0 length (upstream off-by-one-frame interpolation bug)
    length = f0.shape[1]
    if sine_waves.shape[1] > length:
        sine_waves = sine_waves[:, :length, :]
    elif sine_waves.shape[1] < length:
        pad = length - sine_waves.shape[1]
        sine_waves = mx.pad(sine_waves, [(0, 0), (0, pad), (0, 0)])

    uv = self._f02uv(f0)
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    noise = noise_amp * mx.random.normal(sine_waves.shape)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


def apply():
    istftnet.SineGen.__call__ = _sinegen_call


apply()
