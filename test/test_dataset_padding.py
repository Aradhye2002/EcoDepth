import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import DepthDataset


class DepthDatasetPaddingTest(unittest.TestCase):
    def make_dataset(self, crop_h, crop_w):
        dataset = DepthDataset.__new__(DepthDataset)
        dataset.args = SimpleNamespace(
            do_random_crop=True,
            crop_h=crop_h,
            crop_w=crop_w,
            use_cut_depth=False,
        )
        dataset.train_transforms = []
        return dataset

    def test_pads_short_height_and_width_on_their_own_axes(self):
        dataset = self.make_dataset(crop_h=4, crop_w=5)
        image = np.full((2, 3, 3), 7, dtype=np.uint8)
        depth = np.full((2, 3), 1.25, dtype=np.float32)

        padded_image, padded_depth = dataset.apply_train_transforms(image, depth)

        self.assertEqual(padded_image.shape, (4, 5, 3))
        self.assertEqual(padded_depth.shape, (4, 5))
        self.assertEqual(padded_depth.dtype, np.float32)
        np.testing.assert_array_equal(padded_image[:2, :3], image)
        np.testing.assert_array_equal(padded_depth[:2, :3], depth)
        self.assertTrue(np.all(padded_image[2:, :, :] == 0))
        self.assertTrue(np.all(padded_image[:, 3:, :] == 0))
        self.assertTrue(np.all(padded_depth[2:, :] == 0))
        self.assertTrue(np.all(padded_depth[:, 3:] == 0))

    def test_short_width_padding_preserves_taller_images(self):
        dataset = self.make_dataset(crop_h=4, crop_w=5)
        image = np.full((6, 3, 3), 9, dtype=np.uint8)
        depth = np.full((6, 3), 2.5, dtype=np.float32)

        padded_image, padded_depth = dataset.apply_train_transforms(image, depth)

        self.assertEqual(padded_image.shape, (6, 5, 3))
        self.assertEqual(padded_depth.shape, (6, 5))
        self.assertEqual(padded_depth.dtype, np.float32)
        np.testing.assert_array_equal(padded_image[:, :3], image)
        np.testing.assert_array_equal(padded_depth[:, :3], depth)
        self.assertTrue(np.all(padded_image[:, 3:, :] == 0))
        self.assertTrue(np.all(padded_depth[:, 3:] == 0))


if __name__ == "__main__":
    unittest.main()
