"""TFv6 controller search arms; only combine already computed model/author controls."""
import numpy as np


SEARCH_ARMS = 'GHJLM'


def compose_search_control(arm, controls):
    if arm not in SEARCH_ARMS or not all(k in controls for k in 'AB'):
        raise ValueError('Unknown search arm or missing A/B controls')
    a, b = controls['A'], controls['B']
    if arm == 'G':
        steer, longitudinal = b['steer'], a
    elif arm == 'H':
        steer, longitudinal = .5*a['steer'] + .5*b['steer'], a
    elif arm == 'J':
        steer, longitudinal = .8*a['steer'] + .2*b['steer'], a
    elif arm == 'L':
        steer, longitudinal = a['steer'], b
    else:  # M: modest waypoint-PID gain, same B longitudinal law.
        steer, longitudinal = 1.1*b['steer'], b
    return {'steer': float(np.clip(steer,-1.,1.)),
            'throttle': float(longitudinal['throttle']),
            'brake': float(longitudinal['brake'])}
