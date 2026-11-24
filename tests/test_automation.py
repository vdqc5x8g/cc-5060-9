import unittest

import numpy as np

from nmfsep.automation import (
    AmbiguousClaimError,
    AutomationTargetError,
    apply_automation,
    make_automation,
    two_voice_automations,
    write_gate,
)


def separated_fixture():
    rng = np.random.default_rng(0)
    W = np.zeros((20, 3))
    W[0:6, 0] = 1.0
    W[7:13, 1] = 1.0
    W[14:20, 2] = 1.0
    W += 0.02 * rng.uniform(size=W.shape)
    W /= W.sum(axis=0, keepdims=True)
    H = rng.uniform(0.2, 1.0, (3, 50))
    return W, H


class AutomationIdentityTests(unittest.TestCase):
    def test_two_voices_get_different_curves(self):
        W, H = separated_fixture()
        curves = two_voice_automations(W, H.shape[1])
        self.assertEqual(len(curves), 2)
        self.assertNotEqual(curves[0].preset, curves[1].preset)
        self.assertFalse(np.allclose(curves[0].gain, curves[1].gain))
        self.assertEqual(curves[0].gain.shape, (H.shape[1],))

    def test_automation_follows_voice_across_column_swap(self):
        W, H = separated_fixture()
        curve = make_automation("fade", W, 0, H.shape[1], "fade_out_exponential")
        # A backend switch returning permuted columns with identical product:
        perm = [2, 1, 0]
        W2, H2 = W[:, perm], H[perm, :]
        np.testing.assert_allclose(W @ H, W2 @ H2)
        out = apply_automation(curve, W2, H2)
        # identity-keyed claim routes the fade to the voice's new index (2),
        # whereas the stale index 0 would have faded a different instrument.
        self.assertEqual(out["index"], 2)
        np.testing.assert_allclose(out["H"][2], H2[2] * out["env"])
        self.assertFalse(np.allclose(H2[0] * out["env"], H[0] * out["env"]))

    def test_naive_index_route_would_target_wrong_instrument(self):
        # Documents the bug explicitly: index-based automation mis-fires after
        # a column swap; identity-based automation does not.
        W, H = separated_fixture()
        curve = make_automation("fade", W, 0, H.shape[1], "fade_out_linear")
        W2, H2 = W[:, [2, 1, 0]], H[[2, 1, 0], :]
        env = curve.gain
        naive = H2[0] * env  # wrong: column 0 is now the old voice 2
        correct = apply_automation(curve, W2, H2)
        self.assertEqual(correct["index"], 2)
        self.assertFalse(np.allclose(naive, correct["H"][2]))

    def test_ambiguous_voices_block_until_confirmed(self):
        W, H = separated_fixture()
        W[:, 1] = W[:, 0] * 1.0001  # magnitude-indistinguishable twins
        W /= W.sum(axis=0, keepdims=True)
        curve = make_automation("fade", W, 0, H.shape[1], "fade_out_cosine")
        with self.assertRaises(AmbiguousClaimError):
            apply_automation(curve, W, H)
        # gate stays closed without an operator confirmation
        gate = write_gate([curve], W)
        self.assertFalse(gate["writable"])
        self.assertEqual(gate["curves"][0]["status"], "needs_confirmation")
        # explicit confirmation opens the gate and records the decision
        gate_ok = write_gate([curve], W, confirmations={"fade": 0})
        self.assertTrue(gate_ok["writable"])
        curve.confirmed_index = 0
        applied = apply_automation(curve, W, H, confirm=True)
        self.assertTrue(applied["claim"]["confirmed"])

    def test_missing_target_blocks_write(self):
        W, H = separated_fixture()
        curve = make_automation("fade", W, 0, H.shape[1], "swell")
        rng = np.random.default_rng(5)
        W_other = rng.uniform(size=W.shape)
        W_other /= W_other.sum(axis=0, keepdims=True)
        with self.assertRaises(AutomationTargetError):
            apply_automation(curve, W_other, H)
        gate = write_gate([curve], W_other)
        self.assertFalse(gate["writable"])
        self.assertEqual(gate["curves"][0]["status"], "missing")

    def test_presets_are_valid_envelopes(self):
        for preset in ("fade_out_linear", "fade_out_exponential",
                       "fade_out_cosine", "swell"):
            c = make_automation("p", separated_fixture()[0], 0, 40, preset)
            self.assertTrue(np.all(c.gain >= 0.0) and np.all(c.gain <= 1.0001))
        with self.assertRaises(ValueError):
            make_automation("p", separated_fixture()[0], 0, 40, "nonsense")


if __name__ == "__main__":
    unittest.main()
