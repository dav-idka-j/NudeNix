import yaml
import os
import sys

class ConfigLoader:
    def __init__(self, config_path=None, config_filename='censor_settings.yaml'):
        if config_path:
            self.config_path = config_path
        else:
            # Determine the base path for finding resources
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(base_path, config_filename)

        self.settings = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"Error: Configuration file not found at {self.config_path}", file=sys.stderr)
            print("Please ensure 'censor_settings.yaml' exists in the same directory as the script.", file=sys.stderr)
            print("You can copy and customize 'censor_settings.example.yaml' to 'censor_settings.yaml'.", file=sys.stderr)
            sys.exit(1)

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading config: {e}", file=sys.stderr)
            sys.exit(1)

    def get_settings(self):
        return self.settings

    def get_nudenet_settings(self):
        """
        Retrieves Nudenet-specific configuration settings.
        """
        return self.settings.get('nudenet_settings', {})

    def get_clustering_settings(self):
        """
        Retrieves clustering-specific configuration settings.
        """
        return self.settings.get('clustering_settings', {})

    def get_image_hash_settings(self):
        """
        Retrieves image hashing specific configuration settings.
        """
        return self.settings.get('image_hash_settings', {})

    def get_label_settings(self, label):
        """
        Retrieves effective settings for a given label, merging all_labels defaults with label-specific overrides.
        """
        all_labels_settings = self.settings.get('all_labels', {})
        label_specific_settings = self.settings.get('labels', {}).get(label, {})

        # Start with a deep copy of all_labels settings to ensure independent modifications
        effective_settings = {**all_labels_settings}

        # Merge top-level settings (min_score, enabled)
        effective_settings.update(label_specific_settings)

        # If label_specific_settings defines 'censors', it overrides all_labels 'censors'.
        if 'censors' in label_specific_settings:
            effective_settings['censors'] = label_specific_settings['censors']
        elif 'censors' not in effective_settings: # Ensure 'censors' key exists even if all_labels didn't have it
            effective_settings['censors'] = []

        return effective_settings
