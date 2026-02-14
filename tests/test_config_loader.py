import pytest
import yaml
from unittest.mock import patch
from config_loader import ConfigLoader

@pytest.fixture
def create_test_config(tmp_path):
    """Creates a temporary config file for testing."""
    config_content = {
        'nudenet_settings': {
            'model_path': '/path/to/model',
            'inference_resolution': 640
        },
        'clustering_settings': {
            'enabled': True,
            'max_distance': 100
        },
        'all_labels': {
            'enabled': True,
            'min_score': 0.6,
            'censors': [{'type': 'blur'}]
        },
        'labels': {
            'LABEL_A': {
                'enabled': False,
                'min_score': 0.9,
                'censors': [{'type': 'pixelate'}]
            }
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    return str(config_file)

def test_config_loader_loads_and_merges(create_test_config):
    """Test that the config loader correctly loads and merges settings."""
    with patch('config_loader.os.path.join') as mock_path_join:
        mock_path_join.return_value = create_test_config
        loader = ConfigLoader()

        # Test nudenet settings
        nudenet_settings = loader.get_nudenet_settings()
        assert nudenet_settings['model_path'] == '/path/to/model'
        assert nudenet_settings['inference_resolution'] == 640

        # Test clustering settings
        clustering_settings = loader.get_clustering_settings()
        assert clustering_settings['enabled'] is True
        assert clustering_settings['max_distance'] == 100

        # Test all_labels settings
        all_labels_settings = loader.settings.get('all_labels')
        assert all_labels_settings['min_score'] == 0.6

        # Test specific label override by using get_label_settings
        label_a_settings = loader.get_label_settings('LABEL_A')
        assert label_a_settings['enabled'] is False
        assert label_a_settings['min_score'] == 0.9
        assert label_a_settings['censors'][0]['type'] == 'pixelate'

def test_config_loader_exits_if_config_not_found():
    """Test that the config loader exits if the config file is not found."""
    with patch('config_loader.os.path.exists') as mock_exists:
        mock_exists.return_value = False # No config file
        with pytest.raises(SystemExit):
            ConfigLoader()
