import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from censor import CensorProcessor
from config_loader import ConfigLoader

@pytest.fixture
def mock_config_loader():
    """Fixture for a mock ConfigLoader configured for TEST_LABEL with two censors."""
    mock_loader = MagicMock(spec=ConfigLoader)
    label_settings = {
        'enabled': True,
        'min_score': 0.8,
        'censors': [
            {
                'type': 'mock_censor_1',
                'settings': {'param': 'value1'}
            },
            {
                'type': 'mock_censor_2',
                'settings': {'param': 'value2'}
            }
        ]
    }
    # Configure get_label_settings to return a mutable copy
    mock_loader.get_label_settings.side_effect = lambda label: dict(label_settings) if label == 'TEST_LABEL' else {}
    return mock_loader

def test_censor_processor_initialization(mock_config_loader):
    """Test that CensorProcessor initializes correctly and loads configured censors."""
    mock_cc = MagicMock()
    with patch('censor.CensorProcessor._load_censor_modules', return_value={'mock_censor_1': mock_cc, 'mock_censor_2': mock_cc}):
        censor_processor = CensorProcessor(mock_config_loader)

        # Check that the censor classes were loaded correctly
        assert 'mock_censor_1' in censor_processor.censor_classes
        assert 'mock_censor_2' in censor_processor.censor_classes

def test_apply_censorship_no_detections(mock_config_loader):
    """Test that no censorship is applied when there are no detections."""
    censor_processor = CensorProcessor(mock_config_loader)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = []

    processed_frame = censor_processor.apply_censorship(frame.copy(), detections)

    # The frame should be unchanged
    assert np.array_equal(frame, processed_frame)

def test_apply_censorship_disabled_label(mock_config_loader):
    """Test that no censorship is applied for a disabled label."""
    # Temporarily modify the return value for this test
    mock_config_loader.get_label_settings.return_value['enabled'] = False

    censor_processor = CensorProcessor(mock_config_loader)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 20, 20]}]

    processed_frame = censor_processor.apply_censorship(frame.copy(), detections)

    assert np.array_equal(frame, processed_frame)
    # Reset mock for other tests in case of side_effect
    mock_config_loader.get_label_settings.reset_mock()
    mock_config_loader.get_label_settings.side_effect = lambda label: {'enabled': True, 'min_score': 0.8, 'censors': [{'type': 'mock_censor_1', 'settings': {'param': 'value1'}}, {'type': 'mock_censor_2', 'settings': {'param': 'value2'}}]}


def test_apply_censorship_low_score(mock_config_loader):
    """Test that no censorship is applied for a detection with a low score."""
    # Temporarily modify the return value for this test
    mock_config_loader.get_label_settings.return_value['min_score'] = 0.95

    censor_processor = CensorProcessor(mock_config_loader)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 20, 20]}] # score < 0.95

    processed_frame = censor_processor.apply_censorship(frame.copy(), detections)

    assert np.array_equal(frame, processed_frame)
    # Reset mock for other tests in case of side_effect
    mock_config_loader.get_label_settings.reset_mock()
    mock_config_loader.get_label_settings.side_effect = lambda label: {'enabled': True, 'min_score': 0.8, 'censors': [{'type': 'mock_censor_1', 'settings': {'param': 'value1'}}, {'type': 'mock_censor_2', 'settings': {'param': 'value2'}}]}


def test_apply_censorship_single_censor_single_detection(mock_config_loader):
    """Test that a single censor is applied for a valid single detection."""
    mock_censor_class_1 = MagicMock(name='MockCensor1Class')
    mock_censor_instance_1 = MagicMock(name='MockCensor1Instance')
    mock_censor_class_1.return_value = mock_censor_instance_1

    # Configure mock_config_loader to only have mock_censor_1
    mock_config_loader.get_label_settings.side_effect = lambda label: {
        'enabled': True,
        'min_score': 0.8,
        'censors': [{'type': 'mock_censor_1', 'settings': {'param': 'value1'}}]
    }

    with patch('censor.CensorProcessor._load_censor_modules',
               return_value={'mock_censor_1': mock_censor_class_1}):
        censor_processor = CensorProcessor(mock_config_loader)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = {'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 20, 20]}

        censor_processor.apply_censorship(frame.copy(), [detection])

        # Verify that constructor was called once with correct settings
        mock_censor_class_1.assert_called_once_with({'param': 'value1'})
        # Verify that apply method was called once with correct arguments
        mock_censor_instance_1.apply.assert_called_once_with(ANY, detection)


def test_apply_censorship_multiple_censors_multiple_detections(mock_config_loader):
    """
    Test that censorship is applied for multiple valid detections
    and that multiple censors are applied in order.
    """
    mock_censor_1_instances = [MagicMock(name=f'MockCensor1Instance_det{i}') for i in range(2)]
    mock_censor_2_instances = [MagicMock(name=f'MockCensor2Instance_det{i}') for i in range(2)]

    # Use side_effect to return distinct mock instances each time the class is instantiated
    mock_censor_class_1 = MagicMock(name='MockCensor1Class', side_effect=mock_censor_1_instances)
    mock_censor_class_2 = MagicMock(name='MockCensor2Class', side_effect=mock_censor_2_instances)

    with patch('censor.CensorProcessor._load_censor_modules',
               return_value={'mock_censor_1': mock_censor_class_1,
                             'mock_censor_2': mock_censor_class_2}):
        censor_processor = CensorProcessor(mock_config_loader)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 20, 20]},
            {'class': 'TEST_LABEL', 'score': 0.9, 'box': [50, 50, 20, 20]}
        ]

        censor_processor.apply_censorship(frame.copy(), detections)

        # Verify that constructors were called for each detection and each censor
        assert mock_censor_class_1.call_count == 2
        assert mock_censor_class_2.call_count == 2

        # Verify apply calls for each instance
        mock_censor_1_instances[0].apply.assert_called_once_with(ANY, detections[0])
        mock_censor_2_instances[0].apply.assert_called_once_with(ANY, detections[0])
        mock_censor_1_instances[1].apply.assert_called_once_with(ANY, detections[1])
        mock_censor_2_instances[1].apply.assert_called_once_with(ANY, detections[1])
