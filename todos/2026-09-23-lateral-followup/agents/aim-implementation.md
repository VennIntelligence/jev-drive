# Guarded Hermite aim implementation and G1

Source commit `18d6790` changes only `scripts/b2d_controller.py` and the new `scripts/test_b2d_controller_aim.py`. `aim_interpolation="linear"` remains the default; `"hermite"` requires pursuit. The original projection, arc/time arrays, speed calculation, PI, lookahead, and actuator limits remain unchanged. Default and explicit-linear controls match the frozen 432-control fixture exactly; existing 1,800-control vendor regression also passes in the full suite.

The opt-in evaluates a chord-length cubic Hermite aim using locally weighted vector secants. Aim-only duplicate knots are removed; two or fewer distinct knots fall back to linear. Local reversals, nonfinite/degenerate local geometry, or chordwise overshoot fall back with a reason. Query station is clamped; stationary tails do not disable interpolation earlier on the moving path. This does not claim global spline curvature bounds or smoothness across changing plans or projection-segment changes.

Diagnostics expose `aim_interpolation` (requested), `aim_interpolation_used`, and `aim_interpolation_fallback`. No evaluated aim has `aim_xy=null`, used=null and fallback=null; evaluated Hermite fallback has used=linear and a nonempty reason. Successful Hermite has used=hermite and fallback=null.

Seven new tests cover exact default controls, configuration rejection, rigid-transform/reflection equivariance, duplicate points and stationary tails, short-horizon/cusp/local-invalid fallback, unchanged longitudinal outputs for identical input histories, and ideal circular phase ripple/analytic error. The entire controller test discovery passed **119/119**. After that run only a misleading test method name was corrected; the archived test source keeps the original name.

The paired existing synthetic plant suite passed **18/18 linear and 18/18 Hermite** at max lookahead 0.5 s and PI 0.5/0.25: both circle signs at 6/8 m/s with 0/0.3 s trajectory delay, four lateral/heading offsets, S with both delays, stop, and speed steps with 0/0.1/0.2 s actuator delay. Maximum reported circle lateral RMS across these fixtures is 0.016803 m linear versus 0.004355 m Hermite; maximum S RMS is 0.009982 versus 0.009964 m. Both stop runs have identical −0.028329 m final error and zero displacement/speed during the 5 s hold. These are synthetic tire/pedal dynamics with measured controller geometry, not CARLA or neural-policy evidence.

Raw traces and source copies (about 24.7 MB) stay at `/data/runs/b2d/controller/lateral-followup/aim-g1-v1`. [Small summary and hash index](../results/aim-g1-v1/index.json) identify every raw file. Current controller SHA `cc7a7f782ab5327bfc8730d2752bbc8af457bcc5fc17772122ed5610e0ba2004` is identical to the G1 source snapshot. Reproduce into a fresh directory with:

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/aim_closedplant.py --out /data/runs/b2d/controller/lateral-followup/aim-g1-reproduction
```

Negative evidence remains material: the independent prototype shadow on historical baseline states increased raw steer-rate P95 in all four real-route windows (26966: 0.8001→0.9806/s; 24240: 0.2184→0.2517/s; S1: 2.1967→2.3496/s; S2: 2.6783→2.8013/s), despite improving ideal circle sampling. Its artifact is `/data/runs/b2d/controller/lateral-followup/interpolation-shadow-v1`. This prototype lacked the production guards and held physical states fixed; neither its losses nor synthetic G1 wins establish closed-loop performance. The frozen six-case G2 must evaluate emitted steering, tracking, speed and comfort together. There is no real-TCP lateral integration claim.
