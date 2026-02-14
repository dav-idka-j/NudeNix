import cv2
import numpy as np
import os
import importlib.util
import sys

from config_loader import ConfigLoader

class CensorProcessor:
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader

        # Determine the base path for finding resources (e.g., censors directory)
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # Add the 'censors' directory to sys.path if it exists within base_path
        # This allows dynamic importlib.import_module to find censor files
        self.censors_dir = os.path.join(self.base_path, 'censors')
        if os.path.isdir(self.censors_dir) and self.censors_dir not in sys.path:
            sys.path.insert(0, self.censors_dir) # Insert at the beginning to prioritize local censors
            print(f"Added {self.censors_dir} to sys.path for dynamic censor loading.")

        self.censor_classes = self._load_censor_modules()
        print(f"CensorProcessor initialized with config.")

    def _load_censor_modules(self):
        if not os.path.isdir(self.censors_dir):
            print(f"Warning: Censors directory not found at {self.censors_dir}", file=sys.stderr)
            return {}

        loaded_censors = {}
        for filename in os.listdir(self.censors_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3] # Remove .py extension
                try:
                    # Dynamically import the module. Since censors_dir is on sys.path,
                    # we can directly import the module by its name.
                    censor_module = importlib.import_module(module_name)

                    # Convert e.g. 'box_censor' to 'BoxCensor'
                    censor_class_name = "".join([s.capitalize() for s in module_name.split('_')])
                    if hasattr(censor_module, censor_class_name):
                        censor_class = getattr(censor_module, censor_class_name)
                        # Store by its 'type' name (e.g., 'box')
                        loaded_censors[module_name.replace('_censor', '')] = censor_class
                        print(f"Loaded censor type: {module_name.replace('_censor', '')} from {filename}")
                    else:
                        print(f"Warning: Censor class '{censor_class_name}' not found in {filename}", file=sys.stderr)
                except Exception as e:
                    print(f"Error loading censor module {filename}: {e}", file=sys.stderr)
        return loaded_censors

    def apply_censorship(self, frame_ndarray: np.ndarray, detections: list, frame_time=0):
        """
        Applies censorship effects to the given frame based on detections and configured settings.

        Args:
            frame_ndarray (np.ndarray): The video frame as a NumPy array (RGB), modified in-place.
            detections (list): A list of detection dictionaries from NudeDetector.
            frame_time (float): The timestamp of the current frame (for debugging/logging).

        Returns:
            np.ndarray: The modified frame_ndarray.
        """
        for detection in detections:
            label = detection['class']
            score = detection['score']

            # Get effective settings for this label (merging all_labels defaults with overrides)
            effective_settings = self.config_loader.get_label_settings(label)

            # Apply general detection filters
            if not effective_settings.get('enabled', False):
                continue
            if score < effective_settings.get('min_score', 0.0):
                continue

            # Apply configured censors
            censors_to_apply = effective_settings.get('censors', [])
            for censor_config in censors_to_apply:
                censor_type = censor_config.get('type')
                censor_settings = censor_config.get('settings', {})

                if censor_type in self.censor_classes:
                    censor_class = self.censor_classes[censor_type]
                    censor_instance = censor_class(censor_settings)
                    censor_instance.apply(frame_ndarray, detection)
                else:
                    print(f"Warning: Unknown censor type '{censor_type}' for label '{label}'", file=sys.stderr)

        return frame_ndarray
