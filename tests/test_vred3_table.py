"""The vred3 rows of the arbitration table (R1 on the approach, first-answer yellow decision, commit) on scripted answer sequences: no CARLA server.

    python -m unittest tests.test_vred3_table -v          (needs the Bench2Drive paths of envs/carla: runs on the box, skipped elsewhere)
"""
import os
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
for d in (REPO / "lib", REPO / "scripts"):
    sys.path.insert(0, str(d))
DATA = os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")
B2D = Path(DATA) / "third_party/Bench2Drive"
for d in (B2D / "leaderboard", B2D / "scenario_runner", Path(DATA) / "third_party/carla/CARLA_0.9.15/PythonAPI/carla"):
    sys.path.insert(0, str(d))
try:
    import vlm_arb_agent as V
    OK = True
except Exception:  # noqa: BLE001 - no leaderboard / carla on this machine
    OK = False

GREEN, RED, NONE = "green_for_ego", "red_or_yellow_for_ego", "no_light"
R2B = 3.8394


def agent(rows=("R1", "R2", "R5")):
    A = dict(cruise=8.0, amax=2.0, idm_b=2.0, idm_s0=2.0, idm_T=1.0)
    ev = []
    a = SimpleNamespace(arb=A, rows=set(rows), answer=None, h_light=deque(maxlen=2), h_sign=deque(maxlen=2), h_block=deque(maxlen=2), r5=False, r2_hold=False, r3_hold=False,
                        r3_since=None, r3_done=set(), r4_on=False, r4_seen=False, stop_since=None, release_until=-1e9, owned_stop=False, r2_target="stopline", v3=True,
                        n_ans=0, _seen_ans=0, last_green_tq=None, go_commit=False, go_t=None, go_line_s=None, r2_tentative=False, tent_n=0, table_state={}, events=ev)
    a._ylog = lambda **kw: ev.append(kw)
    a._ctx = lambda: {}
    return a


def arrive(a, t, ans, tq=None):
    """An answer for the frame of time tq arrives at time t."""
    tq = t - 0.4 if tq is None else tq
    a.answer = dict(t_eff=t, t_q=tq)
    a.h_light.append(ans)
    a.h_sign.append("na")
    a.h_block.append(("na", "na"))
    a.n_ans += 1
    if ans == GREEN:
        a.last_green_tq = tq


def step(a, t, v, stop_dist, ego_s=50.0, gap=4.0):
    """One table call; `stop_dist` = front bumper to the stop line, the junction entrance is `gap` m beyond the line."""
    junc_dist = stop_dist + gap + R2B
    return V.VlmArbAgent._table(a, v, t, junc_dist, 7, stop_dist, ego_s)


@unittest.skipUnless(OK, "needs the Bench2Drive paths (box)")
class Table(unittest.TestCase):
    def test_r1_on_the_approach_only(self):
        a = agent()
        out, cap, rel, act = step(a, 1.0, 6.0, 12.0)
        self.assertIn("R1", act)
        self.assertEqual(cap, 4.5)
        out, cap, rel, act = step(a, 2.0, 6.0, -0.5)               # the front bumper is past the stop line
        self.assertNotIn("R1", act)
        self.assertEqual(cap, 8.0)
        out, cap, rel, act = step(a, 3.0, 6.0, 40.0)               # far from the junction: no cap
        self.assertNotIn("R1", act)

    def test_stop_decision_is_tentative_and_confirmed(self):
        a = agent()
        arrive(a, 1.0, GREEN, 0.4)
        step(a, 1.0, 3.0, 12.0)
        arrive(a, 1.5, RED, 1.1)
        out, cap, rel, act = step(a, 1.5, 3.0, 11.5)               # first non-green answer: yellow decision, comfortable stop (3 m/s needs 3.0 m)
        self.assertTrue(a.r2_hold)
        self.assertTrue(a.r2_tentative)
        self.assertIn("R2", act)
        self.assertEqual([e["decision"] for e in a.events if e["ev"] == "yellow"], ["stop"])
        arrive(a, 2.0, RED, 1.6)
        step(a, 2.0, 2.5, 10.5)
        self.assertTrue(a.r2_hold)
        self.assertFalse(a.r2_tentative)
        self.assertIn("tentative_confirm", [e["ev"] for e in a.events])

    def test_tentative_cancelled_by_a_green_second_answer(self):
        a = agent()
        arrive(a, 1.0, GREEN, 0.4)
        step(a, 1.0, 3.0, 12.0)
        arrive(a, 1.5, RED, 1.1)
        step(a, 1.5, 3.0, 11.5)
        self.assertTrue(a.r2_hold)
        arrive(a, 2.0, GREEN, 1.6)
        out, cap, rel, act = step(a, 2.0, 2.8, 10.5)
        self.assertFalse(a.r2_hold)
        self.assertNotIn("R2", act)
        self.assertIn("tentative_cancel", [e["ev"] for e in a.events])

    def test_go_suppresses_r2_and_lifts_the_cap_until_the_tail_has_cleared(self):
        a = agent()
        arrive(a, 1.0, GREEN, 0.6)
        step(a, 1.0, 8.0, 3.0, ego_s=40.0)
        arrive(a, 1.5, RED, 1.1)
        out, cap, rel, act = step(a, 1.5, 8.0, 2.0, ego_s=41.0)   # 8 m/s, 2 m before the stop line: no comfortable stop, clears in time
        self.assertTrue(a.go_commit)
        self.assertEqual([e["decision"] for e in a.events if e["ev"] == "yellow"], ["go"])
        self.assertFalse(a.r2_hold)
        self.assertEqual(cap, 8.0)
        arrive(a, 2.0, RED, 1.6)
        out, cap, rel, act = step(a, 2.0, 8.0, 0.5, ego_s=45.0)   # a second red answer does not start a stop during the go
        self.assertFalse(a.r2_hold)
        self.assertNotIn("R2", act)
        self.assertNotIn("R1", act)
        step(a, 2.6, 8.0, -8.0, ego_s=60.0)                        # the front bumper is 5.9 m past the entrance: committed go ends
        self.assertFalse(a.go_commit)
        self.assertIn("go_end", [e["ev"] for e in a.events])

    def test_no_r2_start_past_the_stop_line(self):
        a = agent()
        arrive(a, 1.0, RED, 0.5)
        arrive(a, 1.5, RED, 1.0)
        out, cap, rel, act = step(a, 1.5, 0.5, -0.2)               # creeping, front bumper 0.2 m past the line: the old rule would still start
        self.assertFalse(a.r2_hold)
        self.assertNotIn("R1", act)
        a2 = agent()
        arrive(a2, 1.0, RED, 0.5)
        arrive(a2, 1.5, RED, 1.0)
        step(a2, 1.5, 0.5, 2.0)                                    # the same state before the line starts the stop (K = 2 path)
        self.assertTrue(a2.r2_hold)
        self.assertFalse(a2.r2_tentative)

    def test_old_k2_path_for_a_red_seen_from_far(self):
        a = agent()
        arrive(a, 1.0, NONE, 0.5)
        arrive(a, 1.5, RED, 1.0)
        step(a, 1.5, 5.0, 30.0)
        self.assertFalse(a.r2_hold)                                # first non-green answer without a green before: wait for the second
        arrive(a, 2.0, RED, 1.5)
        step(a, 2.0, 5.0, 28.0)
        self.assertTrue(a.r2_hold)

    def test_old_green_answer_is_treated_as_red(self):
        a = agent()
        arrive(a, 1.0, GREEN, 0.0)
        step(a, 1.0, 6.0, 12.0)
        arrive(a, 4.0, RED, 3.6)                                   # last green frame 3.6 s ago: the yellow's age is unknown
        step(a, 4.0, 6.0, 12.0)
        ev = [e for e in a.events if e["ev"] == "yellow"][0]
        self.assertLess(ev["remaining"], 0)
        self.assertEqual(ev["decision"], "stop")                   # comfortable stop is still preferred


if __name__ == "__main__":
    unittest.main()
