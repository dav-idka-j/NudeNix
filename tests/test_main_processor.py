import pytest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os
import numpy as np
import av
import argparse

from main_processor import MainProcessor, main, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_OUTPUT_FORMAT
from reader import VideoReader
from detector import NsfwDetector
from censor import CensorProcessor
from output import VideoOutputWriter
from config_loader import ConfigLoader
from postprocessor import BoxClusterer
from preprocess_detector import ImageHashDetector

@pytest.fixture
def mock_dependencies():
    """Fixture to provide mocked instances of all MainProcessor's dependencies."""
    mock_reader = MagicMock(spec=VideoReader)
    mock_detector = MagicMock(spec=NsfwDetector)
    mock_censor_processor = MagicMock(spec=CensorProcessor)
    mock_writer = MagicMock(spec=VideoOutputWriter)
    mock_config_loader = MagicMock(spec=ConfigLoader)
    mock_box_clusterer = MagicMock(spec=BoxClusterer)

    # Configure common mock behaviors
    mock_reader.open.return_value = True
    mock_reader.video_in = MagicMock(spec=av.VideoStream)
    mock_reader.audio_in = MagicMock(spec=av.AudioStream)
    mock_reader.read_packets.return_value = iter([]) # Default to no packets

    mock_writer.open.return_value = True

    mock_config_loader.get_nudenet_settings.return_value = {}
    mock_config_loader.get_clustering_settings.return_value = {'enabled': False}
    mock_config_loader.get_image_hash_settings.return_value = {'enabled': True} # Default for image hash

    return {
        'reader': mock_reader,
        'detector': mock_detector,
        'censor_processor': mock_censor_processor,
        'writer': mock_writer,
        'config_loader': mock_config_loader,
        'box_clusterer': mock_box_clusterer
    }

@pytest.fixture
def create_main_processor(mock_dependencies):
    """Fixture to create a MainProcessor instance with mocked dependencies."""
    return MainProcessor(
        reader=mock_dependencies['reader'],
        detector=mock_dependencies['detector'],
        censor_processor=mock_dependencies['censor_processor'],
        writer=mock_dependencies['writer'],
        config_loader=mock_dependencies['config_loader'],
        box_clusterer=mock_dependencies['box_clusterer']
    )

def test_main_processor_initialization(mock_dependencies):
    """Test MainProcessor initializes correctly with all dependencies."""
    processor = MainProcessor(**mock_dependencies)
    assert processor.reader == mock_dependencies['reader']
    assert processor.detector == mock_dependencies['detector']
    assert processor.censor_processor == mock_dependencies['censor_processor']
    assert processor.writer == mock_dependencies['writer']
    assert processor.config_loader == mock_dependencies['config_loader']
    assert processor.box_clusterer == mock_dependencies['box_clusterer']

def test_main_processor_run_empty_video(create_main_processor, mock_dependencies):
    """Test run method with a video that yields no packets."""
    processor = create_main_processor
    processor.run()

    mock_dependencies['reader'].open.assert_called_once()
    mock_dependencies['writer'].open.assert_called_once()
    mock_dependencies['reader'].read_packets.assert_called_once()
    mock_dependencies['writer'].flush_encoders.assert_called_once()
    mock_dependencies['writer'].close.assert_called_once()
    mock_dependencies['reader'].close.assert_called_once()
    mock_dependencies['detector'].detect.assert_not_called()
    mock_dependencies['censor_processor'].apply_censorship.assert_not_called()

def test_main_processor_run_reader_open_fails(create_main_processor, mock_dependencies):
    """Test run method exits if reader.open() fails."""
    mock_dependencies['reader'].open.return_value = False
    processor = create_main_processor

    with pytest.raises(SystemExit):
        processor.run()

    mock_dependencies['reader'].open.assert_called_once()
    mock_dependencies['reader'].close.assert_not_called()  # Reader's open failed, so its close should not be called

def test_main_processor_run_writer_open_fails(create_main_processor, mock_dependencies):
    """Test run method exits if writer.open() fails."""
    mock_dependencies['writer'].open.return_value = False
    processor = create_main_processor

    with pytest.raises(SystemExit):
        processor.run()

    mock_dependencies['reader'].open.assert_called_once()
    mock_dependencies['writer'].open.assert_called_once()
    mock_dependencies['reader'].close.assert_called_once() # Reader's open succeeded, so it should be closed
    mock_dependencies['writer'].close.assert_not_called() # Writer's open failed, so its close should not be called
    mock_dependencies['reader'].read_packets.assert_not_called()

def test_main_processor_run_with_video_frames(mock_dependencies):
    """Test run method processes video frames correctly."""
    mock_dependencies['box_clusterer'] = None
    processor = MainProcessor(**mock_dependencies)

    mock_frame = MagicMock(spec=av.VideoFrame)
    mock_frame.to_ndarray.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_packet = MagicMock(spec=av.Packet)
    mock_packet.stream = mock_dependencies['reader'].video_in
    mock_packet.decode.return_value = iter([mock_frame])
    mock_dependencies['reader'].read_packets.return_value = iter([mock_packet])

    mock_dependencies['detector'].detect.return_value = [{'box': [0,0,1,1], 'class': 'test', 'score': 0.9}]
    mock_dependencies['censor_processor'].apply_censorship.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    processor.run()

    mock_dependencies['detector'].detect.assert_called_once_with(mock_frame.to_ndarray.return_value)
    mock_dependencies['censor_processor'].apply_censorship.assert_called_once_with(
        ANY, # The actual frame (NumPy array) passed to apply_censorship
        mock_dependencies['detector'].detect.return_value
    )
    mock_dependencies['writer'].write_video_frame.assert_called_once()
    mock_dependencies['writer'].flush_encoders.assert_called_once()

@patch('main_processor.os.path.exists', return_value=True) # Mock os.path.exists for input file
@patch('main_processor.VideoReader')
@patch('main_processor.NsfwDetector')
@patch('main_processor.CensorProcessor')
@patch('main_processor.VideoOutputWriter')
@patch('main_processor.ConfigLoader')
@patch('main_processor.BoxClusterer')
@patch('main_processor.ImageHashDetector') # Patch ImageHashDetector
@patch('argparse.ArgumentParser') # Patch ArgumentParser
def test_main_function_default_flow_with_argparse(
    MockArgumentParser,
    MockImageHashDetector, MockBoxClusterer, MockConfigLoader, MockVideoOutputWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoReader,
    mock_exists # This is the mocked os.path.exists
):
    """Test the main function with default flow and arguments using argparse."""
    # Configure mock args
    mock_args = MagicMock()
    mock_args.input = 'input.mp4'
    mock_args.output = 'output.mp4'
    mock_args.output_type = 'file'
    mock_args.format = 'mp4'
    mock_args.config = None
    mock_args.disable_image_hash = False
    mock_args.enable_image_hash_debug = False
    mock_args.nudenet_model_path = None
    mock_args.nudenet_resolution = None
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_reader_instance = MockVideoReader.return_value
    mock_reader_instance.open.return_value = True
    mock_reader_instance.video_in = MagicMock(spec=av.VideoStream)
    mock_reader_instance.audio_in = MagicMock(spec=av.AudioStream)

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_nudenet_settings.return_value = {}
    mock_config_loader_instance.get_clustering_settings.return_value = {'enabled': False}
    # Mock image hash settings to be enabled by default from config
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True, 'debug_overlay': False}

    mock_writer_instance = MockVideoOutputWriter.return_value
    mock_writer_instance.open.return_value = True

    with patch('main_processor.MainProcessor') as MockMainProcessor:
        mock_main_processor_instance = MockMainProcessor.return_value
        main()

        # Verify ArgumentParser usage
        MockArgumentParser.assert_called_once()
        MockArgumentParser.return_value.parse_args.assert_called_once()

        # Verify component instantiation
        mock_exists.assert_called_once_with('input.mp4')
        MockVideoReader.assert_called_once_with('input.mp4')
        MockConfigLoader.assert_called_once_with(config_path=None) # No custom config path
        MockNsfwDetector.assert_called_once_with(model_path=None, inference_resolution=320)

        # ImageHashDetector should be called with the NsfwDetector instance
        MockImageHashDetector.assert_called_once_with(MockNsfwDetector.return_value, {'enabled': True, 'debug_overlay': False})
        MockCensorProcessor.assert_called_once_with(mock_config_loader_instance)
        MockBoxClusterer.assert_not_called() # Clustering disabled

        # Verify reader.open() is called before writer is instantiated
        mock_reader_instance.open.assert_called_once()

        MockVideoOutputWriter.assert_called_once_with(
            'output.mp4',
            mock_reader_instance, # Pass the reader_instance
            output_format='mp4'
        )

        MockMainProcessor.assert_called_once_with(
            mock_reader_instance,
            MockImageHashDetector.return_value, # Expect ImageHashDetector if enabled
            MockCensorProcessor.return_value,
            mock_writer_instance,
            mock_config_loader_instance,
            None # BoxClusterer is None
        )
        mock_main_processor_instance.run.assert_called_once()

@patch('main_processor.os.path.exists', return_value=False) # Input file not found
@patch('main_processor.VideoReader')
@patch('main_processor.NsfwDetector')
@patch('main_processor.CensorProcessor')
@patch('main_processor.VideoOutputWriter')
@patch('main_processor.ConfigLoader')
@patch('main_processor.BoxClusterer')
@patch('argparse.ArgumentParser') # Patch ArgumentParser
def test_main_function_input_file_not_found_with_argparse(
    MockArgumentParser,
    MockBoxClusterer, MockConfigLoader, MockVideoOutputWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoReader,
    mock_exists # This is the mocked os.path.exists
):
    """Test main function exits if input file is not found using argparse."""
    # Configure mock args for non-existent input
    mock_args = MagicMock()
    mock_args.input = 'non_existent.mp4'
    mock_args.output_type = 'file' # Set a default for output type
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    with pytest.raises(SystemExit):
        main()
    mock_exists.assert_called_once_with('non_existent.mp4')
    MockVideoReader.assert_not_called()
    MockConfigLoader.assert_not_called()

@patch('main_processor.os.path.exists', return_value=True)
@patch('main_processor.ConfigLoader')
@patch('argparse.ArgumentParser')
def test_main_function_output_required_for_file_type(MockArgumentParser, MockConfigLoader, mock_exists):
    """Test parser error when output is file type but --output is missing."""
    mock_args = MagicMock()
    mock_args.input = 'input.mp4'
    mock_args.output = None # Missing output for file type
    mock_args.output_type = 'file'
    mock_args.format = 'mp4' # Required for coverage
    mock_args.config = None
    mock_args.disable_image_hash = False
    mock_args.enable_image_hash_debug = False
    mock_args.nudenet_model_path = None
    mock_args.nudenet_resolution = None
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    # Mock parser.error to check if it's called
    MockArgumentParser.return_value.error = MagicMock(side_effect=SystemExit)

    with pytest.raises(SystemExit):
        main()

    MockArgumentParser.return_value.error.assert_called_once_with(
        "--output is required when --output-type is 'file'."
    )

@patch('main_processor.os.path.exists', return_value=True)
@patch('main_processor.VideoReader')
@patch('main_processor.NsfwDetector')
@patch('main_processor.CensorProcessor')
@patch('main_processor.VideoOutputWriter')
@patch('main_processor.ConfigLoader')
@patch('main_processor.ImageHashDetector')
@patch('argparse.ArgumentParser')
def test_main_function_cli_overrides_nudenet_settings(
    MockArgumentParser, MockImageHashDetector, MockConfigLoader, MockVideoOutputWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoReader, mock_exists
):
    """Test CLI arguments override nudenet settings from config."""
    mock_args = MagicMock()
    mock_args.input = 'input.mp4'
    mock_args.output = 'output.mp4'
    mock_args.output_type = 'file'
    mock_args.format = 'mp4'
    mock_args.config = None
    mock_args.disable_image_hash = False
    mock_args.enable_image_hash_debug = False
    mock_args.nudenet_model_path = '/cli/model.onnx'
    mock_args.nudenet_resolution = 1024
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_nudenet_settings.return_value = {
        'model_path': '/config/model.onnx',
        'inference_resolution': 320
    }
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True}

    with patch('main_processor.MainProcessor'):
        main()

        MockNsfwDetector.assert_called_once_with(
            model_path='/cli/model.onnx',
            inference_resolution=1024
        )

@patch('main_processor.os.path.exists', return_value=True)
@patch('main_processor.VideoReader')
@patch('main_processor.NsfwDetector')
@patch('main_processor.CensorProcessor')
@patch('main_processor.VideoOutputWriter')
@patch('main_processor.ConfigLoader')
@patch('main_processor.ImageHashDetector')
@patch('argparse.ArgumentParser')
def test_main_function_cli_overrides_image_hash_settings(
    MockArgumentParser, MockImageHashDetector, MockConfigLoader, MockVideoOutputWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoReader, mock_exists
):
    """Test CLI arguments override image hash settings from config."""
    mock_args = MagicMock()
    mock_args.input = 'input.mp4'
    mock_args.output = 'output.mp4'
    mock_args.output_type = 'file'
    mock_args.format = 'mp4'
    mock_args.config = None
    mock_args.disable_image_hash = True # CLI disables
    mock_args.enable_image_hash_debug = True # CLI enables debug
    mock_args.nudenet_model_path = None
    mock_args.nudenet_resolution = None
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_config_loader_instance = MockConfigLoader.return_value
    # Config has image hash enabled, but CLI overrides it to disable
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True, 'debug_overlay': False}

    with patch('main_processor.MainProcessor'):
        main()

        MockImageHashDetector.assert_called_once()
        actual_config = MockImageHashDetector.call_args[0][1]
        assert actual_config['enabled'] is False # CLI override
        assert actual_config['debug_overlay'] is True # CLI override

@patch('main_processor.os.path.exists', return_value=True)
@patch('main_processor.VideoReader')
@patch('main_processor.NsfwDetector')
@patch('main_processor.CensorProcessor')
@patch('main_processor.VideoOutputWriter')
@patch('main_processor.ConfigLoader')
@patch('main_processor.ImageHashDetector')
@patch('argparse.ArgumentParser')
def test_main_function_custom_config_path(
    MockArgumentParser, MockImageHashDetector, MockConfigLoader, MockVideoOutputWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoReader, mock_exists
):
    """Test main function passes custom config path to ConfigLoader."""
    mock_args = MagicMock()
    mock_args.input = 'input.mp4'
    mock_args.output = 'output.mp4'
    mock_args.output_type = 'file'
    mock_args.format = 'mp4'
    mock_args.config = 'custom_config.yaml' # Custom config path
    mock_args.disable_image_hash = False
    mock_args.enable_image_hash_debug = False
    mock_args.nudenet_model_path = None
    mock_args.nudenet_resolution = None
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_nudenet_settings.return_value = {}
    mock_config_loader_instance.get_clustering_settings.return_value = {'enabled': False}
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True}

    with patch('main_processor.MainProcessor'):
        main()

        MockConfigLoader.assert_called_once_with(config_path='custom_config.yaml')
