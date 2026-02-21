from fractions import Fraction
import pytest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os
import numpy as np
import av
import argparse

from main import MainProcessor, main, parse_args
from input.video_file_reader import VideoFileReader
from detector import NsfwDetector
from censor import CensorProcessor
from video_file_writer import VideoFileWriter
from config_loader import ConfigLoader
from postprocessor import BoxClusterer
from preprocess_detector import ImageHashDetector
from censor_drawing_contexts.in_place_drawing_context import InPlaceDrawingContext
from input.screen_reader import ScreenCaptureReader
from output.screen_writer import ScreenWriter


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
    mock_reader.read_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), None, 0)

    mock_writer.open.return_value = True
    mock_writer.get_fps.return_value = 25.0
    mock_writer.get_video_time_base.return_value = Fraction(1, 25)
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

def test_main_processor_run_empty_video(create_main_processor):
    """Test run method with a video that yields no frames."""
    processor = create_main_processor
    processor.reader.read_frame.side_effect = [(None, None, None)]
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
    processor.reader.read_frame.side_effect = [(test_frame_np, None, 100), (None, None, None)]

    detections = [{'box': [0,0,1,1], 'class': 'test', 'score': 0.9}]
    processor.detector.detect.return_value = detections

    processor.censor_processor.apply_censorship.return_value = test_frame_np

    processor.run()

    processor.reader.open.assert_called_once()
    processor.writer.open.assert_called_once()
    assert processor.reader.read_frame.call_count == 2
    processor.detector.detect.assert_called_once_with(test_frame_np)

    expected_detections_for_censor = detections
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


@patch('main.os.path.exists', return_value=True)
@patch('main.VideoFileReader')
@patch('main.VideoFileWriter')
@patch('main.ScreenCaptureReader')
@patch('main.ScreenWriter')
@patch('main.CensorProcessor')
@patch('main.ImageHashDetector')
@patch('main.NsfwDetector')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.InPlaceDrawingContext')
def test_main_flow_video_to_file(
    MockInPlaceDrawingContext,
    MockBoxClusterer,
    MockConfigLoader,
    MockNsfwDetector,
    MockImageHashDetector,
    MockCensorProcessor,
    MockScreenWriter,
    MockScreenCaptureReader,
    MockVideoFileWriter,
    MockVideoFileReader,
    mock_os_exists):
    """Tests the main execution flow for a file-to-file operation."""
    mock_reader = MockVideoFileReader.return_value
    mock_reader.open.return_value = True
    mock_reader.read_frame.side_effect = [(np.zeros((100, 100, 3), np.uint8), None, 0), (None, None, None)]

    mock_writer = MockVideoFileWriter.return_value
    mock_writer.open.return_value = True
    mock_writer.get_fps.return_value = 30.0
    mock_writer.get_video_time_base.return_value = Fraction(1, 30)

    mock_censor_processor = MockCensorProcessor.return_value
    mock_censor_processor.apply_censorship.return_value = np.zeros((100, 100, 3), np.uint8)

    MockConfigLoader.return_value.get_clustering_settings.return_value = {'enabled': False}
    MockConfigLoader.return_value.get_image_hash_settings.return_value = {'enabled': True}

    input_path = "/test/input.mp4"
    output_path = "/test/output.mp4"

    cli_args = ["-i", f"file://{input_path}", "-o", f"file://{output_path}"]

    args = parse_args(cli_args)
    main(args)

    mock_os_exists.assert_called_with(input_path)
    MockVideoFileReader.assert_called_once_with(input_path)
    MockVideoFileWriter.assert_called_once_with(output_path, mock_reader, output_format='mp4')
    MockScreenCaptureReader.assert_not_called()
    MockScreenWriter.assert_not_called()
    mock_reader.open.assert_called_once()
    mock_writer.open.assert_called_once()
    MockCensorProcessor.assert_called_once()


@patch('main.ScreenCaptureReader')
@patch('main.ScreenWriter')
@patch('main.VideoFileReader')
@patch('main.VideoFileWriter')
@patch('main.CensorProcessor')
@patch('main.ImageHashDetector')
@patch('main.NsfwDetector')
@patch('main.ConfigLoader')
@patch('main.BoxClusterer')
@patch('main.InPlaceDrawingContext')
def test_main_flow_screen_to_overlay(
    MockInPlaceDrawingContext,
    MockBoxClusterer,
    MockConfigLoader,
    MockNsfwDetector,
    MockImageHashDetector,
    MockCensorProcessor,
    MockVideoFileWriter,
    MockVideoFileReader,
    MockScreenWriter,
    MockScreenCaptureReader):
    """Tests the main execution flow for a screen-to-overlay operation."""
    mock_reader = MockScreenCaptureReader.return_value
    mock_reader.open.return_value = True
    mock_reader.read_frame.side_effect = [(np.zeros((100, 100, 3), np.uint8), None, 0), (None, None, None)]
    mock_reader.get_fps.return_value = 30.0
    mock_reader.get_frame_size.return_value = (1920, 1080)
    mock_reader.get_audio_layout.return_value = None
    mock_reader.get_audio_rate.return_value = None

    mock_writer = MockScreenWriter.return_value
    mock_writer.open.return_value = True
    mock_writer.get_fps.return_value = 30.0
    mock_writer.get_video_time_base.return_value = Fraction(1, 30)

    mock_censor_processor = MockCensorProcessor.return_value
    mock_censor_processor.apply_censorship.return_value = np.zeros((100, 100, 3), np.uint8)

    MockConfigLoader.return_value.get_clustering_settings.return_value = {'enabled': False}
    MockConfigLoader.return_value.get_image_hash_settings.return_value = {'enabled': True}

    cli_args = ["-i", "screen://2", "-o", "overlay://"]
    args = parse_args(cli_args)
    main(args)

    MockScreenCaptureReader.assert_called_once_with(monitor=2)
    MockScreenWriter.assert_called_once_with(
        reader=mock_reader,
        overlay_context=ANY
    )
    MockVideoFileReader.assert_not_called()
    MockVideoFileWriter.assert_not_called()
    mock_reader.open.assert_called_once()
    mock_writer.open.assert_called_once()
    MockCensorProcessor.assert_called_once()
