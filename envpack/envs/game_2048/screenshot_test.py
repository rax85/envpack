"""Screenshot diff tests for the Gym2048Env environment."""

import os
import unittest
import numpy as np
from PIL import Image, ImageChops

from envpack.envs.game_2048.env import Gym2048Env


class TestGym2048Screenshot(unittest.TestCase):
    """Screenshot diff tests for the Gym2048Env environment."""

    def setUp(self):
        self.env = Gym2048Env()
        self.env.reset()
        self.baseline_dir = os.path.join(os.path.dirname(__file__), "baselines")
        os.makedirs(self.baseline_dir, exist_ok=True)

    def _get_screenshot_path(self, name, is_baseline=True):
        """Helper to get the path for a screenshot."""
        if is_baseline:
            return os.path.join(self.baseline_dir, f"{name}.png")
        return os.path.join("/tmp", f"current_{name}.png")

    def _save_screenshot(self, env, name, is_baseline=False):
        """Helper to save a screenshot of the environment."""
        env._render()
        rgb_data = env.render()
        image = Image.fromarray(rgb_data)
        image.save(self._get_screenshot_path(name, is_baseline))

    def _compare_screenshots(self, name, threshold=10):
        """Compares a generated screenshot to its baseline."""
        current_path = self._get_screenshot_path(name, is_baseline=False)
        baseline_path = self._get_screenshot_path(name, is_baseline=True)

        if not os.path.exists(baseline_path):
            self.fail(
                f"Baseline image not found: {baseline_path}. "
                "Please generate it by running the test with UPDATE_BASELINES=1."
            )

        current_image = Image.open(current_path).convert("RGB")
        baseline_image = Image.open(baseline_path).convert("RGB")

        diff = ImageChops.difference(current_image, baseline_image)
        diff_array = np.array(diff)
        # Calculate the sum of absolute differences across all color channels
        total_diff = np.sum(np.abs(diff_array))

        # The threshold is applied to the total difference.
        # A common approach is to normalize this by the image size * 255 * 3 (for RGB)
        # to get a percentage difference, but for simplicity, we'll use a raw sum threshold.
        # This threshold might need tuning based on image size and acceptable variance.
        self.assertLessEqual(
            total_diff,
            threshold,
            f"Screenshot '{name}' differs from baseline. "
            f"Total difference: {total_diff}, Threshold: {threshold}",
        )

    def test_initial_render_matches_baseline(self):
        """Test that the initial render state matches a baseline screenshot."""
        # Set a known initial state (e.g., after reset)
        self.env.reset()
        self.env._grid = np.asarray(
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]]
        )  # Example grid for a consistent baseline

        # Save the current screenshot
        self._save_screenshot(self.env, "initial_state_render", is_baseline=False)

        # If an environment variable is set, update the baseline
        if os.environ.get("UPDATE_BASELINES") == "1":
            self._save_screenshot(self.env, "initial_state_render", is_baseline=True)
            print(
                f"Updated baseline for 'initial_state_render' at {self._get_screenshot_path('initial_state_render', True)}"
            )
        else:
            # Compare to baseline
            self._compare_screenshots("initial_state_render")

    def test_game_over_render_matches_baseline(self):
        """Test that the game over render state matches a baseline screenshot."""
        self.env.reset()
        # Set a deadlock state (Game Over)
        # 2 4 2 4
        # 4 2 4 2
        # 2 4 2 4
        # 4 2 4 2
        self.env._grid = np.asarray([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ], dtype=np.int32)
        self.env._score = 1000  # Set a fixed score for consistency

        # Save the current screenshot
        self._save_screenshot(self.env, "game_over_render", is_baseline=False)

        # If an environment variable is set, update the baseline
        if os.environ.get("UPDATE_BASELINES") == "1":
            self._save_screenshot(self.env, "game_over_render", is_baseline=True)
            print(
                f"Updated baseline for 'game_over_render' at {self._get_screenshot_path('game_over_render', True)}"
            )
        else:
            # Compare to baseline
            self._compare_screenshots("game_over_render")


if __name__ == "__main__":
    unittest.main()
