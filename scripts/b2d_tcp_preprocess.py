"""Parallelize independent official TCP JPEG operations without changing pixels."""
import cv2
import os
import numpy as np
import torch

CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')


def combine_rgb(futures):
    return model_rgb({}, None, futures)


def jpeg_camera(bgra):
    # Direct four-channel conversion avoids materializing a strided three-channel view.
    rgb = (cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
           if os.environ.get('B2D_TCP_FAST_COLOR', '0') == '1'
           else cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB))
    ok, encoded = cv2.imencode('.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
    if not ok:
        raise RuntimeError('TCP JPEG encoding failed')
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def model_rgb(input_data, pool, futures=None):
    names = CAMERAS
    if futures is not None:
        left, front, right = [futures[k].result() for k in names]
    else:
        left, front, right = pool.map(jpeg_camera, [input_data[k][1] for k in names])
    rgb = np.concatenate((left[:, :1400], front[:, 200:1400], right[:, 200:]), axis=1)
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    rgb = torch.nn.functional.interpolate(rgb, size=(256, 900), mode='bilinear', align_corners=False)
    return rgb.squeeze(0).permute(1, 2, 0).byte().numpy()
