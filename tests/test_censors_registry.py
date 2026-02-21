import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add censors directory to path to allow importing mock censors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'censors')))

from censors_registry import CensorsRegistry

# Mock Censor Classes
class MockCensorWithApply:
    def __init__(self, settings):
        self.settings = settings
    def apply(self, image_np, detection):
        pass

class MockCensorWithoutApply:
    def __init__(self, settings):
        self.settings = settings

@pytest.fixture
def registry():
    """Fixture to provide a CensorsRegistry instance."""
    # Patching _load_censor_modules to prevent it from running automatically
    with patch.object(CensorsRegistry, '_load_censor_modules') as mock_load:
        registry = CensorsRegistry()
    return registry

def test_registry_initialization(registry):
    """Test that the registry initializes with an empty dictionary of censors."""
    assert registry._censors == {}
    assert registry.cache == {}

def test_register_censor_success(registry):
    """Test successful registration of a valid censor class."""
    registry.register_censor("mock_censor", MockCensorWithApply)
    assert "mock_censor" in registry._censors
    assert registry._censors["mock_censor"] == MockCensorWithApply

def test_register_censor_no_apply_method_raises_error(registry):
    """Test that registering a class without an 'apply' method raises a ValueError."""
    with pytest.raises(ValueError, match="Censor class MockCensorWithoutApply must have an 'apply' method."):
        registry.register_censor("invalid_censor", MockCensorWithoutApply)

def test_get_censor_instance_success(registry):
    """Test retrieving a valid censor instance."""
    registry.register_censor("mock_censor", MockCensorWithApply)
    settings = {"param": "value"}
    instance = registry.get_censor_instance("TEST_LABEL", "mock_censor", settings)
    assert isinstance(instance, MockCensorWithApply)
    assert instance.settings == settings

def test_get_censor_instance_unknown_type_raises_error(registry):
    """Test that retrieving an unknown censor type raises a ValueError."""
    with pytest.raises(ValueError, match="No censor registered for type: unknown"):
        registry.get_censor_instance("TEST_LABEL", "unknown", {})

def test_get_censor_instance_caching(registry):
    """Test that the censor instances are cached."""
    registry.register_censor("mock_censor", MockCensorWithApply)
    settings = {"param": "value"}
    instance1 = registry.get_censor_instance("TEST_LABEL", "mock_censor", settings)
    instance2 = registry.get_censor_instance("TEST_LABEL", "mock_censor", settings)
    assert instance1 is instance2

def test_dynamic_loading_of_censors():
    """Test that the registry dynamically loads censors from the 'censors' directory."""
    # This test relies on the actual file structure
    registry = CensorsRegistry()
    assert "box" in registry._censors
    assert "pixelation" in registry._censors
    # Check if the class names are correct (optional, but good for verification)
    assert registry._censors["box"].__name__ == "BoxCensor"
    assert registry._censors["pixelation"].__name__ == "PixelationCensor"
