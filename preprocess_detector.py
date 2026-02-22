import imagehash
from PIL import Image
import numpy as np
import cv2
import sys
import time
from statistics import HashingReportModule

class ImageHashDetector:
    def __init__(self, nsfw_detector, config, stats=None):
        self.nsfw_detector = nsfw_detector
        self.enabled = config.get('enabled', True)
        self.algorithm_name = config.get('algorithm', 'average_hash')
        self.hash_threshold = config.get('hash_threshold', 11)
        self.max_skip_frames = config.get('max_skip_frames', 5)
        self.debug_overlay = config.get('debug_overlay', False)

        self.last_known_hash = None
        self.last_known_detections = []
        self.current_frame_skip_count = 0

        self.hash_function = self._get_hash_function(self.algorithm_name)

        if stats and stats.enabled:
            stats.register_module(HashingReportModule(), priority=20)

    def _get_hash_function(self, name):
        if name == 'average_hash':
            return imagehash.average_hash
        elif name == 'phash':
            return imagehash.phash
        elif name == 'dhash':
            return imagehash.dhash
        elif name == 'whash':
            return imagehash.whash
        else:
            print(f"Warning: Unknown hash algorithm '{name}'. Falling back to average_hash.", file=sys.stderr)
            return imagehash.average_hash

    def _calculate_hash(self, image_np):
        image_pil = Image.fromarray(image_np)
        return self.hash_function(image_pil)

    def _draw_debug_overlay(self, image_np, color):
        if not self.debug_overlay:
            return image_np
        # Draw a circle on the upper left corner
        cv2.circle(image_np, (10, 10), 5, color, -1) # 10px radius, filled circle
        return image_np

    def detect(self, image_np, stats=None):
        if not self.enabled:
            # If disabled, just pass through to the underlying detector
            return self.nsfw_detector.detect(image_np, stats=stats)

        if stats:
            with stats.timer('hashing'):
                current_hash = self._calculate_hash(image_np)
        else:
            current_hash = self._calculate_hash(image_np)

        # Check if it's the first frame or if max skip frames reached
        if self.last_known_hash is None or self.current_frame_skip_count >= self.max_skip_frames:
            if stats:
                with stats.timer('detection'):
                    detections = self.nsfw_detector.detect(image_np, stats=stats)
            else:
                detections = self.nsfw_detector.detect(image_np, stats=stats)

            self.last_known_hash = current_hash
            self.last_known_detections = detections
            self.current_frame_skip_count = 0

            # Blue circle for forced detection run
            self._draw_debug_overlay(image_np, (0, 0, 255))
            return detections
        else:
            # Compare hashes
            hash_diff = current_hash - self.last_known_hash
            if hash_diff < self.hash_threshold:
                # Hashes are similar, reuse last detections
                self.current_frame_skip_count += 1

                if stats:
                    stats.increment_counter('skip_count')

                # Green circle for skipped detection
                self._draw_debug_overlay(image_np, (0, 255, 0))
                return self.last_known_detections
            else:
                # Hashes are different, run new detection
                if stats:
                    with stats.timer('detection'):
                        detections = self.nsfw_detector.detect(image_np, stats=stats)
                else:
                    detections = self.nsfw_detector.detect(image_np, stats=stats)

                self.last_known_hash = current_hash
                self.last_known_detections = detections
                self.current_frame_skip_count = 0

                # Red circle for actual detection run
                self._draw_debug_overlay(image_np, (255, 0, 0))
                return detections
