"""VLM Arbitration Protocol: Question schemas, labels, and answer parsing.

Four discrete decision questions:
  1. Q_light: traffic light state for ego lane
     options: ['no_light', 'red_or_yellow_for_ego', 'green_for_ego', 'light_for_other_lane']
  2. Q_sign: stop sign controlling ego vehicle
     options: yes / no (noul)
  3. Q_block: obstacle state in ego lane ahead
     options: ['clear', 'moving_lead', 'static_block']
  4. Q_side: which adjacent lane is free to bypass (only when Q_block == 'static_block')
     options: ['left_free', 'right_free', 'none_free']
"""
from typing import Dict, Any, Optional

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
            "static_block": "static blockage, stopped vehicle, construction cone, or obstacle blocking ego lane"
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

LIGHT_CHOICES = ["no_light", "red_or_yellow_for_ego", "green_for_ego", "light_for_other_lane"]
BLOCK_CHOICES = ["clear", "moving_lead", "static_block"]
SIDE_CHOICES = ["left_free", "right_free", "none_free"]


def parse_vlm_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Parse raw System One / VLM response into normalized answers."""
    answers = resp.get("answers", {})
    
    # Q_light
    ql = answers.get("Q_light", {})
    q_light = ql.get("choice", "no_light")
    if q_light not in LIGHT_CHOICES:
        q_light = "no_light"
    q_light_conf = float(ql.get("confidence", 0.0))

    # Q_sign
    qs = answers.get("Q_sign", {})
    # noul is probability of yes (1.0 = yes, 0.0 = no)
    q_sign_val = float(qs.get("noul", 0.0))
    q_sign = "yes" if q_sign_val >= 0.5 else "no"
    q_sign_conf = float(qs.get("confidence", 0.0))

    # Q_block
    qb = answers.get("Q_block", {})
    q_block = qb.get("choice", "clear")
    if q_block not in BLOCK_CHOICES:
        q_block = "clear"
    q_block_conf = float(qb.get("confidence", 0.0))

    # Q_side
    qside = answers.get("Q_side", {})
    q_side = qside.get("choice", "none_free")
    if q_side not in SIDE_CHOICES:
        q_side = "none_free"
    q_side_conf = float(qside.get("confidence", 0.0))

    return {
        "Q_light": q_light,
        "Q_light_conf": q_light_conf,
        "Q_sign": q_sign,
        "Q_sign_val": q_sign_val,
        "Q_sign_conf": q_sign_conf,
        "Q_block": q_block,
        "Q_block_conf": q_block_conf,
        "Q_side": q_side,
        "Q_side_conf": q_side_conf,
    }
