import os
import tempfile
import unittest

import numpy as np
from drive_runtime.preview import PreviewReader, PreviewWriter, LivePreview, shared_path
from b2d_viewer import project_waypoints, anchored_waypoints, chase_waypoints


class PreviewTests(unittest.TestCase):
    def test_session_marks_end(self):
        with tempfile.TemporaryDirectory() as root:
            session = LivePreview(root)
            reader = PreviewReader(root)
            try:
                session.publish({'frame': 7, 'running': True}, {'arbitrary': np.zeros((2, 3, 3), np.uint8)})
                self.assertTrue(reader.read()['state']['running'])
                session.close()
                ended = reader.read()['state']
                self.assertEqual(ended['frame'], 7)
                self.assertFalse(ended['running'])
            finally:
                reader.close()
                os.unlink(shared_path(root))

    def test_camera_overlay_uses_matching_historical_prediction(self):
        old = dict(frame=9, pred_wp=[[10, 1]], camera_inverse=np.eye(4).tolist())
        current = dict(frame=10, chase_frame=9, pred_wp=[[10, -4]], chase_state=old)
        self.assertEqual(chase_waypoints(current, 640, 360), project_waypoints(old, 640, 360))
        current['chase_frame'] = 8
        self.assertEqual(chase_waypoints(current, 640, 360), [])
    def test_transport_latest_noncontiguous_and_restart(self):
        with tempfile.TemporaryDirectory() as root:
            writer = PreviewWriter(root)
            reader = PreviewReader(root)
            try:
                self.assertIsNone(reader.read())
                image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, :, ::-1]
                writer.publish({'frame': 1}, {'input': image})
                writer.publish({'frame': 2}, {'input': image})
                packet = reader.read()
                self.assertEqual(packet['state']['frame'], 2)
                self.assertEqual(packet['pixels'][:72], image.tobytes())
                self.assertIsNone(reader.read())
                writer.close()
                writer = PreviewWriter(root)
                writer.publish({'frame': 3}, {'input': image})
                self.assertEqual(reader.read()['state']['frame'], 3)
                with self.assertRaises(ValueError):
                    writer.publish({'text': 'x' * 70000}, {})
            finally:
                reader.close()
                writer.close()
                os.unlink(shared_path(root))

    def test_projection_axis(self):
        state = dict(pred_wp=[[10, 0], [10, 2]], camera_inverse=np.eye(4).tolist(), camera_fov=90,
                     prediction_origin=[0, 0])
        points = project_waypoints(state, 640, 360)
        self.assertEqual(points[0][0], 320)
        self.assertGreater(points[1][0], points[0][0])
        self.assertLess(points[0][1], 180)

    def test_origin_is_fixed_without_recentring_prediction(self):
        wp = [[1, -0.3], [4, 0.4]]
        self.assertEqual(anchored_waypoints({'pred_wp': wp}), [[0, 0]] + wp)
        self.assertEqual(wp, [[1, -0.3], [4, 0.4]])


if __name__ == '__main__':
    unittest.main()
