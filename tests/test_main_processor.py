import pytest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os
import numpy as np
import av
import argparse

from main import MainProcessor, main, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_OUTPUT_FORMAT
from video_file_reader import VideoFileReader
from detector import NsfwDetector
from censor import CensorProcessor
from video_file_writer import VideoFileWriter
from config_loader import ConfigLoader
from postprocessor import BoxClusterer
from preprocess_detector import ImageHashDetector
from censor_drawing_contexts.in_place_drawing_context import InPlaceDrawingContext


@pytest.fixture
def mock_dependencies():
    """Fixture to provide mocked instances of all MainProcessor's dependencies."""
    mock_reader = MagicMock(spec=VideoFileReader)
    mock_detector = MagicMock(spec=NsfwDetector)
    mock_drawing_context = MagicMock(spec=InPlaceDrawingContext)
    mock_censor_processor = MagicMock(spec=CensorProcessor)
    mock_writer = MagicMock(spec=VideoFileWriter)
    mock_config_loader = MagicMock(spec=ConfigLoader)
    mock_box_clusterer = MagicMock(spec=BoxClusterer)

    mock_reader.open.return_value = True
    mock_reader.video_in = MagicMock(spec=av.VideoStream)
    mock_reader.audio_in = MagicMock(spec=av.AudioStream)
    mock_reader.get_fps.return_value = 25.0
    mock_reader.get_frame_size.return_value = (640, 480)
    mock_reader.read_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), None)

    mock_writer.open.return_value = True
    mock_writer.write_video_frame.return_value = None
    mock_writer.write_audio_packet.return_value = None

    mock_config_loader.get_nudenet_settings.return_value = {}
    mock_config_loader.get_clustering_settings.return_value = {'enabled': False}
    mock_config_loader.get_image_hash_settings.return_value = {'enabled': True}
    mock_config_loader.get_label_settings.return_value = {'enabled': True, 'min_score': 0.5, 'censors': [{'type': 'box', 'settings': {}}]}

    return {
        'reader': mock_reader,
        'detector': mock_detector,
        'censor_processor': mock_censor_processor,
        'writer': mock_writer,
        'config_loader': mock_config_loader,
        'box_clusterer': mock_box_clusterer,
        'drawing_context': mock_drawing_context
    }

@pytest.fixture
def create_main_processor(mock_dependencies):
    """Fixture to create a MainProcessor instance with mocked dependencies."""
    # Ensure censor_processor is the MagicMock directly
    mock_dependencies['censor_processor'].apply_censorship.return_value = ANY # Default return for apply_censorship

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
    # For this test, we need a real CensorProcessor, but not necessarily its full mock behavior
    real_censor_processor = CensorProcessor(
        mock_dependencies['config_loader'],
        mock_dependencies['drawing_context']
    )

    processor = MainProcessor(
        reader=mock_dependencies['reader'],
        detector=mock_dependencies['detector'],
        censor_processor=real_censor_processor, # Use real CensorProcessor
        writer=mock_dependencies['writer'],
        config_loader=mock_dependencies['config_loader'],
        box_clusterer=mock_dependencies['box_clusterer']
    )
    assert processor.reader == mock_dependencies['reader']
    assert processor.detector == mock_dependencies['detector']
    assert processor.censor_processor == real_censor_processor
    assert processor.writer == mock_dependencies['writer']
    assert processor.config_loader == mock_dependencies['config_loader']
    assert processor.box_clusterer == mock_dependencies['box_clusterer']

def test_main_processor_run_empty_video(create_main_processor):
    """Test run method with a video that yields no frames."""
    processor = create_main_processor
    processor.reader.read_frame.side_effect = [(None, None)]
    processor.run()

    processor.reader.open.assert_called_once()
    processor.writer.open.assert_called_once()
    processor.reader.read_frame.assert_called_once()
    processor.writer.flush_encoders.assert_called_once()
    processor.writer.close.assert_called_once()
    processor.reader.close.assert_called_once()
    processor.detector.detect.assert_not_called()
    processor.censor_processor.apply_censorship.assert_not_called()

def test_main_processor_run_reader_open_fails(create_main_processor):
    """Test run method exits if reader.open() fails."""
    processor = create_main_processor
    processor.reader.open.return_value = False

    with pytest.raises(SystemExit):
        processor.run()

    processor.reader.open.assert_called_once()
    processor.reader.close.assert_not_called()

def test_main_processor_run_writer_open_fails(create_main_processor):
    """Test run method exits if writer.open() fails."""
    processor = create_main_processor
    processor.writer.open.return_value = False

    with pytest.raises(SystemExit):
        processor.run()

    processor.reader.open.assert_called_once()
    processor.writer.open.assert_called_once()
    processor.reader.close.assert_called_once()
    processor.writer.close.assert_not_called()
    processor.reader.read_frame.assert_not_called()

def test_main_processor_run_with_video_frames(create_main_processor):
    """Test run method processes video frames correctly."""
    processor = create_main_processor

    test_frame_np = np.zeros((480, 640, 3), dtype=np.uint8)
    processor.reader.read_frame.side_effect = [(test_frame_np, None), (None, None)]

    detections = [{'box': [0,0,1,1], 'class': 'test', 'score': 0.9}]
    processor.detector.detect.return_value = detections
    # Ensure box_clusterer returns something if it's used
    if processor.box_clusterer:
        processor.box_clusterer.cluster_boxes.return_value = detections

    processor.censor_processor.apply_censorship.return_value = test_frame_np

    processor.run()

    processor.reader.open.assert_called_once()
    processor.writer.open.assert_called_once()
    assert processor.reader.read_frame.call_count == 2
    processor.detector.detect.assert_called_once_with(test_frame_np)

    expected_detections_for_censor = detections
    if processor.box_clusterer:
        processor.box_clusterer.cluster_boxes.assert_called_once_with(detections)
        expected_detections_for_censor = processor.box_clusterer.cluster_boxes.return_value

    processor.censor_processor.apply_censorship.assert_called_once_with(
        test_frame_np,
        expected_detections_for_censor
    )
    processor.writer.write_video_frame.assert_called_once()
    processor.writer.flush_encoders.assert_called_once()
    processor.writer.close.assert_called_once()
    processor.reader.close.assert_called_once()

# Corrected order of arguments in test function signature AND explicit mock_args attributes
@patch('main.sys.exit', side_effect=SystemExit) # Add side_effect
@patch('main.os.path.exists', return_value=True)
@patch('main.VideoFileReader')
@patch('main.NsfwDetector')
@patch('main.CensorProcessor')
@patch('main.VideoFileWriter')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.ImageHashDetector')
@patch('argparse.ArgumentParser')
@patch('main.InPlaceDrawingContext') # Patch InPlaceDrawingContext in main module
def test_main_function_default_flow_with_argparse(
    MockInPlaceDrawingContext, MockArgumentParser, # Order corrected
    MockImageHashDetector, MockBoxClusterer, MockConfigLoader, MockVideoFileWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoFileReader,
    mock_exists,
    mock_sys_exit
):
    """Test the main function with default flow and arguments using argparse."""
    # Explicitly define mock_args attributes to return concrete values
    mock_args = MagicMock(
        input='input.mp4',
        output='output.mp4',
        output_type='file',
        format='mp4',
        config=None,
        disable_image_hash=False,
        enable_image_hash_debug=False,
        nudenet_model_path=None,
        nudenet_resolution=None
    )
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_reader_instance = MockVideoFileReader.return_value
    mock_reader_instance.open.return_value = True
    mock_reader_instance.video_in = MagicMock(spec=av.VideoStream)
    mock_reader_instance.audio_in = MagicMock(spec=av.AudioStream)

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_nudenet_settings.return_value = {}
    mock_config_loader_instance.get_clustering_settings.return_value = {'enabled': False}
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True, 'debug_overlay': False}

    mock_writer_instance = MockVideoFileWriter.return_value
    mock_writer_instance.open.return_value = True

    # Removed the patch('main.MainProcessor') block
    main()

    MockArgumentParser.assert_called_once()
    MockArgumentParser.return_value.parse_args.assert_called_once()

    mock_exists.assert_called_once_with('input.mp4')
    MockVideoFileReader.assert_called_once_with('input.mp4')
    MockConfigLoader.assert_called_once_with(config_path=None)
    MockNsfwDetector.assert_called_once_with(model_path=None, inference_resolution=320)

    MockImageHashDetector.assert_called_once_with(MockNsfwDetector.return_value, {'enabled': True, 'debug_overlay': False})

    MockInPlaceDrawingContext.assert_called_once()
    MockCensorProcessor.assert_called_once_with(mock_config_loader_instance, MockInPlaceDrawingContext.return_value)
    MockBoxClusterer.assert_not_called()

    mock_reader_instance.open.assert_called_once()
    mock_writer_instance.open.assert_called_once() # Add assertion for writer.open()

    MockVideoFileWriter.assert_called_once_with(
        'output.mp4',
        mock_reader_instance,
        output_format='mp4'
    )

    mock_sys_exit.assert_not_called()


@patch('main.sys.exit', side_effect=SystemExit) # Add side_effect
@patch('main.os.path.exists', return_value=False)
@patch('main.VideoFileReader')
@patch('main.NsfwDetector')
@patch('main.CensorProcessor')
@patch('main.VideoFileWriter')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.ImageHashDetector')
@patch('argparse.ArgumentParser')
@patch('main.InPlaceDrawingContext') # Patch InPlaceDrawingContext in main module
def test_main_function_input_file_not_found_with_argparse(
    MockInPlaceDrawingContext, MockArgumentParser, # Order corrected
    MockImageHashDetector, MockBoxClusterer, MockConfigLoader, MockVideoFileWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoFileReader,
    mock_exists,
    mock_sys_exit
):
    """Test main function exits if input file is not found using argparse."""
    # Explicitly define mock_args attributes to return concrete values
    mock_args = MagicMock(
        input='non_existent.mp4',
        output='output.mp4',
        output_type='file',
        format='mp4',
        config=None,
        disable_image_hash=False,
        enable_image_hash_debug=False,
        nudenet_model_path=None,
        nudenet_resolution=None
    )
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    with pytest.raises(SystemExit): # Now this should correctly catch the SystemExit
        main()

    mock_exists.assert_called_once_with('non_existent.mp4')
    MockVideoFileReader.assert_not_called()
    MockConfigLoader.assert_not_called()
    MockInPlaceDrawingContext.assert_not_called()
    mock_sys_exit.assert_called_once_with(1) # assert that sys.exit(1) was called

# Corrected arguments order in test function signature AND explicit mock_args attributes
@patch('main.sys.exit', side_effect=SystemExit) # Add side_effect
@patch('main.os.path.exists', return_value=True)
@patch('main.VideoFileReader')
@patch('main.NsfwDetector')
@patch('main.CensorProcessor')
@patch('main.VideoFileWriter')
@patch('main.ConfigLoader')
@patch('argparse.ArgumentParser')
def test_main_function_output_required_for_file_type(
    MockArgumentParser, MockConfigLoader,
    MockVideoFileWriter, MockCensorProcessor, MockNsfwDetector, MockVideoFileReader, # Order corrected
    mock_exists, mock_sys_exit
):
    """Test parser error when output is file type but --output is missing."""
    # Explicitly define mock_args attributes to return concrete values
    mock_args = MagicMock(
        input='input.mp4',
        output=None, # Missing output for file type
        output_type='file',
        format='mp4',
        config=None,
        disable_image_hash=False,
        enable_image_hash_debug=False,
        nudenet_model_path=None,
        nudenet_resolution=None
    )
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    MockArgumentParser.return_value.error = MagicMock(side_effect=SystemExit)

    with pytest.raises(SystemExit): # Wrap main call with pytest.raises
        main()

    MockArgumentParser.return_value.error.assert_called_once_with(
        "--output is required when --output-type is 'file'."
    )
    mock_sys_exit.assert_not_called() # sys.exit is not called directly by main() in this path


# Corrected arguments order in test function signature AND explicit mock_args attributes
@patch('main.sys.exit', side_effect=SystemExit) # Add side_effect
@patch('main.os.path.exists', return_value=True)
@patch('main.VideoFileReader')
@patch('main.NsfwDetector')
@patch('main.CensorProcessor')
@patch('main.VideoFileWriter')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.ImageHashDetector')
@patch('argparse.ArgumentParser')
@patch('main.InPlaceDrawingContext') # Patch InPlaceDrawingContext in main module
def test_main_function_cli_overrides_nudenet_settings(
    MockInPlaceDrawingContext, MockArgumentParser, # Order corrected
    MockImageHashDetector, MockBoxClusterer, MockConfigLoader, MockVideoFileWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoFileReader, mock_exists,
    mock_sys_exit
):
    """Test CLI arguments override nudenet settings from config."""
    # Explicitly define mock_args attributes to return concrete values
    mock_args = MagicMock(
        input='input.mp4',
        output='output.mp4',
        output_type='file',
        format='mp4',
        config=None,
        disable_image_hash=False,
        enable_image_hash_debug=False,
        nudenet_model_path='/cli/model.onnx',
        nudenet_resolution=1024
    )
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_nudenet_settings.return_value = {
        'model_path': '/config/model.onnx',
        'inference_resolution': 320
    }
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True}

    # Removed the patch('main.MainProcessor') block
    main()

    MockNsfwDetector.assert_called_once_with(
        model_path='/cli/model.onnx',
        inference_resolution=1024
    )
    MockInPlaceDrawingContext.assert_called_once()
    mock_sys_exit.assert_not_called()


# Corrected arguments order in test function signature AND explicit mock_args attributes
@patch('main.sys.exit', side_effect=SystemExit) # Add side_effect
@patch('main.os.path.exists', return_value=True)
@patch('main.VideoFileReader')
@patch('main.NsfwDetector')
@patch('main.CensorProcessor')
@patch('main.VideoFileWriter')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.ImageHashDetector')
@patch('argparse.ArgumentParser')
@patch('main.InPlaceDrawingContext') # Patch InPlaceDrawingContext in main module
def test_main_function_cli_overrides_image_hash_settings(
    MockInPlaceDrawingContext, MockArgumentParser, # Order corrected
    MockImageHashDetector, MockBoxClusterer, MockConfigLoader, MockVideoFileWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoFileReader, mock_exists,
    mock_sys_exit
):
    """Test CLI arguments override image hash settings from config."""
    # Explicitly define mock_args attributes to return concrete values
    mock_args = MagicMock(
        input='input.mp4',
        output='output.mp4',
        output_type='file',
        format='mp4',
        config=None,
        disable_image_hash=True,
        enable_image_hash_debug=True,
        nudenet_model_path=None,
        nudenet_resolution=None
    )

    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True, 'debug_overlay': False}

    # Removed the patch('main.MainProcessor') block
    main()

    MockImageHashDetector.assert_called_once()
    actual_config = MockImageHashDetector.call_args[0][1]
    assert actual_config['enabled'] is False
    assert actual_config['debug_overlay'] is True
    MockInPlaceDrawingContext.assert_called_once()
    mock_sys_exit.assert_not_called()


# Corrected arguments order in test function signature AND explicit mock_args attributes
@patch('main.sys.exit', side_effect=SystemExit) # Add side_effect
@patch('main.os.path.exists', return_value=True)
@patch('main.VideoFileReader')
@patch('main.NsfwDetector')
@patch('main.CensorProcessor')
@patch('main.VideoFileWriter')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.ImageHashDetector')
@patch('argparse.ArgumentParser')
@patch('main.InPlaceDrawingContext') # Patch InPlaceDrawingContext in main module
def test_main_function_custom_config_path(
    MockInPlaceDrawingContext, MockArgumentParser, # Order corrected
    MockImageHashDetector, MockBoxClusterer, MockConfigLoader, MockVideoFileWriter,
    MockCensorProcessor, MockNsfwDetector, MockVideoFileReader, mock_exists,
    mock_sys_exit
):
    """Test main function passes custom config path to ConfigLoader."""
    # Explicitly define mock_args attributes to return concrete values
    mock_args = MagicMock(
        input='input.mp4',
        output='output.mp4',
        output_type='file',
        format='mp4',
        config='custom_config.yaml',
        disable_image_hash=False,
        enable_image_hash_debug=False,
        nudenet_model_path=None,
        nudenet_resolution=None
    )
    MockArgumentParser.return_value.parse_args.return_value = mock_args

    mock_config_loader_instance = MockConfigLoader.return_value
    mock_config_loader_instance.get_nudenet_settings.return_value = {}
    mock_config_loader_instance.get_clustering_settings.return_value = {'enabled': False}
    mock_config_loader_instance.get_image_hash_settings.return_value = {'enabled': True}

    # Removed the patch('main.MainProcessor') block
    main()

    MockConfigLoader.assert_called_once_with(config_path='custom_config.yaml')
    MockInPlaceDrawingContext.assert_called_once()
    mock_sys_exit.assert_not_called()
