import unittest

import numpy as np

from nmfsep.stft import istft, stft, stft_consistency_error


class STFTTests(unittest.TestCase):
    def test_roundtrip_random(self):
        rng = np.random.default_rng(0)
        for n_fft, hop in [(1024, 256), (512, 128), (2048, 512), (1024, 300)]:
            x = rng.standard_normal(9001)
            err = stft_consistency_error(x, stft(x, n_fft, hop), n_fft, hop)
            self.assertLess(err, 1e-10, (n_fft, hop, err))

    def test_roundtrip_sine_and_silence(self):
        sr = 22050
        t = np.arange(sr) / sr
        for x in (0.5 * np.sin(2 * np.pi * 440 * t), np.zeros(sr)):
            y = istft(stft(x, 1024, 256), 1024, 256, length=len(x))
            np.testing.assert_allclose(y, x, atol=1e-10)

    def test_shapes(self):
        x = np.zeros(10000)
        S = stft(x, 1024, 256)
        self.assertEqual(S.shape[0], 513)
        # centered padding adds n_fft, and the last frame is padded to the end
        self.assertEqual(S.shape[1], 1 + int(np.ceil(10000 / 256)))

    def test_rejects_bad_hop(self):
        with self.assertRaises(ValueError):
            stft(np.zeros(1000), 1024, 0)
        with self.assertRaises(ValueError):
            stft(np.zeros(1000), 1024, 2048)

    def test_length_handling(self):
        x = np.zeros(1234)
        y = istft(stft(x, 1024, 256), 1024, 256, length=1234)
        self.assertEqual(len(y), 1234)


if __name__ == "__main__":
    unittest.main()
