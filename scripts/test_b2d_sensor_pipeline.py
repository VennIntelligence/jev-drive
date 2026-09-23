import os
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np
from drive_runtime.sensors import TimedSensorQueue, collect_frame
from b2d_tcp_preprocess import jpeg_camera, model_rgb, combine_rgb, CAMERAS


class PipelineTests(unittest.TestCase):
    def test_only_requested_frame_is_submitted(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            queue = TimedSensorQueue(pool, transforms={tag: jpeg_camera for tag in CAMERAS})
            queue.begin(10)
            data = np.zeros((10, 10, 4), dtype=np.uint8)
            queue.put(('CAM_FRONT', 9, data))
            queue.get()
            self.assertEqual(queue.futures, {})
            queue.put(('CAM_FRONT', 11, data))
            queue.get()
            self.assertEqual(queue.futures, {})
            queue.put(('CAM_FRONT', 10, data))
            queue.get()
            np.testing.assert_array_equal(queue.futures['CAM_FRONT'].result(), jpeg_camera(data))
            self.assertIn('CAM_FRONT', queue.trace)
            queue.begin(11)
            self.assertEqual(queue.futures, {})

    def test_color_and_pipeline_pixel_equivalence(self):
        rng = np.random.RandomState(55)
        images = {tag: (20, rng.randint(0, 256, (900, 1600, 4), dtype=np.uint8)) for tag in CAMERAS}
        original = {tag: data[1].copy() for tag, data in images.items()}
        with ThreadPoolExecutor(max_workers=3) as pool:
            with patch.dict(os.environ, B2D_TCP_FAST_COLOR='0'):
                expected = model_rgb(images, pool)
            with patch.dict(os.environ, B2D_TCP_FAST_COLOR='1'):
                queue = TimedSensorQueue(pool, transforms={tag: jpeg_camera for tag in CAMERAS}, combine=combine_rgb)
                queue.begin(20)
                for tag in reversed(CAMERAS):
                    queue.put((tag, 20, images[tag][1]))
                    queue.get()
                actual = model_rgb(images, pool, queue.futures)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(queue.combined_future.result(timeout=5), expected)
                # Every view/worker operation is read-only with respect to policy inputs.
                for tag in CAMERAS:
                    self.assertEqual(images[tag][0], 20)
                    np.testing.assert_array_equal(images[tag][1], original[tag])

    def test_worker_failure_is_not_hidden(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            with patch('b2d_tcp_preprocess.jpeg_camera', side_effect=RuntimeError('JPEG test error')) as broken:
                queue = TimedSensorQueue(pool, transforms={tag: broken for tag in CAMERAS}, combine=combine_rgb)
                queue.begin(2)
                for tag in CAMERAS:
                    queue.put((tag, 2, None))
                    queue.get()
                with self.assertRaisesRegex(RuntimeError, 'JPEG test error'):
                    queue.combined_future.result(timeout=5)

    def test_generic_non_image_data_and_optional_display(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            queue = TimedSensorQueue(pool, transforms={'measurement': lambda x: x * 2},
                                     combine=lambda fs: fs['measurement'].result() + 1,
                                     optional_tags=('display',))
            queue.begin(10)
            queue.put(('measurement', 9, 999))
            queue.put(('measurement', 10, 3))
            # No display frame: model barrier must still return.
            result = collect_frame(queue, ['measurement'], 10, timeout=0.1)
            self.assertEqual(result['measurement'], (10, 3))
            self.assertEqual(queue.combined_future.result(timeout=1), 7)
            self.assertIsNone(queue.optional_frame('display', 10))
            queue.put(('display', 11, 'future'))
            queue.put(('display', 9, 'older'))
            self.assertEqual(queue.optional_frame('display', 10), (9, 'older'))
            self.assertTrue(queue.empty())
            with self.assertRaises(ValueError):
                collect_frame(queue, ['display'], 10, timeout=0.1)


if __name__ == '__main__':
    unittest.main()
