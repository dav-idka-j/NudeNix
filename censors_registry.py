import os
import sys
import importlib
from typing import Dict, Type, Any

# No longer importing BaseCensor

class CensorsRegistry:
    """
    A registry for managing and providing censor implementations dynamically.
    It discovers censor classes in the 'censors' directory and registers them.
    """
    def __init__(self):
        self._censors: Dict[str, Type[Any]] = {} # Type hints updated
        self.base_path = self._get_base_path()
        self.censors_dir = os.path.join(self.base_path, 'censors')
        self._load_censor_modules()
        self.cache: Dict[Tuple[str, str], obj] = {}

    def _get_base_path(self) -> str:
        """
        Determines the base path for finding resources (e.g., censors directory).
        Handles frozen applications (e.g., PyInstaller).
        """
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def _load_censor_modules(self):
        """
        Dynamically loads censor modules from the 'censors' directory
        and registers any discovered censor classes that follow the convention
        (i.e., have an 'apply' method).
        """
        if os.path.isdir(self.censors_dir) and self.censors_dir not in sys.path:
            sys.path.insert(0, self.censors_dir) # Insert at the beginning to prioritize local censors
            print(f"Added {self.censors_dir} to sys.path for dynamic censor loading.")

        if not os.path.isdir(self.censors_dir):
            print(f"Warning: Censors directory not found at {self.censors_dir}.", file=sys.stderr)
            return

        for filename in os.listdir(self.censors_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3] # Remove .py extension
                try:
                    module = importlib.import_module(module_name)
                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)
                        if (isinstance(attribute, type) and
                            hasattr(attribute, 'apply') and # Check for 'apply' method
                            callable(getattr(attribute, 'apply'))): # Ensure it's callable

                            # Register the censor using its module name (filename)
                            # Assuming class name == module_name or similar convention
                            # For example, box_censor.py -> BoxCensor -> 'box'
                            censor_type = module_name.replace('_censor', '')
                            self.register_censor(censor_type, attribute)
                            print(f"Dynamically registered censor: {censor_type} from {filename}")
                except Exception as e:
                    print(f"Error loading censor module {filename}: {e}", file=sys.stderr)

    def register_censor(self, censor_type: str, censor_class: Type[Any]):
        """
        Registers a censor implementation with a given type string.

        Args:
            censor_type: The string identifier for the censor (e.g., "box", "pixelation").
            censor_class: The class (not an instance) of the censor implementation.
                          Must follow the convention of having an 'apply' method.
        """
        # Runtime check for 'apply' method as we are relying on convention
        if not (hasattr(censor_class, 'apply') and callable(getattr(censor_class, 'apply'))):
            raise ValueError(f"Censor class {censor_class.__name__} must have an 'apply' method.")
        self._censors[censor_type] = censor_class

    def get_censor_instance(self, label: str, censor_type: str, settings: Dict[str, Any]) -> Any:
        """
        Retrieves an instance of the censor for the given type, initialized with settings.

        Args:
            censor_type: The string identifier for the censor.
            settings: A dictionary of settings specific to the censor type.

        Returns:
            An instance of the requested censor.

        Raises:
            ValueError: If no censor is registered for the given type.
        """
        cache_key = (label, censor_type)
        instance = self.cache.get(cache_key);
        if instance is not None:
            return instance

        censor_class = self._censors.get(censor_type)
        if censor_class is None:
            raise ValueError(f"No censor registered for type: {censor_type}")
        instance = censor_class(settings)
        self.cache[cache_key] = instance

        return instance
