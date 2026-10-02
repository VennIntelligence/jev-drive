"""VLM arbitration protocol: the four questions, their options, input variants and answer parsing (Python 3.8).

  Q_light  no_light / red_or_yellow_for_ego / green_for_ego / light_for_other_lane
  Q_sign   stop sign controlling the ego ahead: yes / no (the server returns P(yes) as "noul")
  Q_block  clear / moving_lead / static_block
  Q_side   left_free / right_free / none_free (read only when Q_block == static_block)

A variant changes what the Q_light request sees (which cameras, an extra crop, the wording); it never changes the
answer options. Variants exist for the Q-light diagnosis (experiments/vlm_arb/scripts/vlm_arb_qlight.py); "base" is
what Phase A measured.
"""
from typing import Any, Dict

QUESTIONS_SCHEMA = {
    "Q_light": {
        "type": "choice",
        "instructions": "Traffic light status controlling the ego vehicle lane ahead.",
        "criteria": {
            "no_light": "no traffic light controlling ego lane ahead",
            "red_or_yellow_for_ego": "red or yellow traffic light controlling ego lane ahead",
            "green_for_ego": "green traffic light controlling ego lane ahead",
            "light_for_other_lane": "traffic light visible but controlling another lane, not ego lane"
        }
    },
    "Q_sign": {
        "type": "noul",
        "instructions": "Is there a stop sign controlling the ego vehicle ahead at the intersection?"
    },
    "Q_block": {
        "type": "choice",
        "instructions": "Obstacle status in the ego vehicle travel lane ahead.",
        "criteria": {
            "clear": "clear path ahead in ego lane",
            "moving_lead": "moving vehicle or dynamic object moving ahead in ego lane",
            "static_block": "stationary obstacle, construction cones, or stopped broken vehicle blocking ego lane"
        }
    },
    "Q_side": {
        "type": "choice",
        "instructions": "If there is a blockage ahead, which adjacent lane is clear and safe to bypass?",
        "criteria": {
            "left_free": "left adjacent lane is clear and available for bypass",
            "right_free": "right adjacent lane is clear and available for bypass",
            "none_free": "neither adjacent lane is clear, or no adjacent lane exists"
        }
    }
}

# Q_light wording that reads the lit lamp by its position in the housing instead of its hue (the red lamp of a
# CARLA light saturates to amber in the openpilot camera rig, see the Q-light diagnosis in the plan).
LIGHT_POS = {
    "type": "choice",
    "instructions": ("Look only at the traffic light that faces the camera straight ahead, above or beside the ego "
                     "lane at the next stop line. Ignore lights seen from the side that serve cross traffic. Decide "
                     "by which lamp of the vertical housing is lit, not by its colour: top lamp lit means stop, "
                     "middle lamp lit means stop, bottom lamp lit means go."),
    "criteria": {
        "no_light": "no traffic light faces the ego lane ahead",
        "red_or_yellow_for_ego": "the facing light has its top or middle lamp lit (stop)",
        "green_for_ego": "the facing light has its bottom lamp lit (go)",
        "light_for_other_lane": "only lights for other directions are visible"
    }
}

# name -> {"cams": camera order, "crop": add an upper-centre crop of the wide camera, "light": Q_light schema}
VARIANTS = {
    "base": {"cams": ["wide", "road"], "crop": False, "light": QUESTIONS_SCHEMA["Q_light"]},
    "road": {"cams": ["road"], "crop": False, "light": QUESTIONS_SCHEMA["Q_light"]},
    "pos": {"cams": ["wide", "road"], "crop": False, "light": LIGHT_POS},
    "crop": {"cams": ["wide", "road"], "crop": True, "light": QUESTIONS_SCHEMA["Q_light"]},
}
CROP_BOX = (0.25, 0.10, 0.75, 0.55)   # x0, y0, x1, y1 as fractions of the wide frame: where a facing light sits

LIGHT_CHOICES = ["no_light", "red_or_yellow_for_ego", "green_for_ego", "light_for_other_lane"]
BLOCK_CHOICES = ["clear", "moving_lead", "static_block"]
SIDE_CHOICES = ["left_free", "right_free", "none_free"]


def parse_vlm_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalized answers of the questions present in a raw System One response (absent questions are left out)."""
    answers = resp.get("answers", {})
    out = {}
    for key, choices, default in (("Q_light", LIGHT_CHOICES, "no_light"), ("Q_block", BLOCK_CHOICES, "clear"),
                                  ("Q_side", SIDE_CHOICES, "none_free")):
        if key in answers:
            a = answers[key]
            c = a.get("choice", default)
            out[key] = c if c in choices else default
            out[key + "_conf"] = float(a.get("confidence", 0.0))
            out[key + "_p"] = {k: round(float(v), 4) for k, v in (a.get("probabilities") or {}).items()}
    if "Q_sign" in answers:
        v = float(answers["Q_sign"].get("noul", 0.0))                # P(yes)
        out.update(Q_sign="yes" if v >= 0.5 else "no", Q_sign_val=v)
    return out
