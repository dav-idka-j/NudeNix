import pytest
from unittest.mock import MagicMock, patch, call
import av
import numpy as np
from video_file_reader import VideoFileReader

@pytest.fixture
def mock_av_container():
    """Fixture for a mock av.container.InputContainer."""
    mock_container = MagicMock(spec=av.container.InputContainer)

    # Mock video stream
    mock_video_stream = MagicMock(spec=av.VideoStream)
    mock_video_stream.name = 'video_stream_0'
    mock_video_stream.index = 0
    mock_video_stream.average_rate = 25.0
    mock_video_stream.width = 640
    mock_video_stream.height = 480

    # Mock audio stream
    mock_audio_stream = MagicMock(spec=av.AudioStream)
    mock_audio_stream.name = 'audio_stream_0'
    mock_audio_stream.index = 1
    mock_audio_stream.rate = 44100
    mock_audio_stream.sample_rate = 44100 # Add sample_rate
    mock_audio_stream.channels = 2
    mock_audio_stream.layout = av.AudioLayout('stereo') # Use actual Av.AudioLayout type

    # Mock container streams property
    mock_container.streams.video = [mock_video_stream]
    mock_container.streams.audio = [mock_audio_stream]

    # Mock decode method to yield actual Av.VideoFrame and Av.AudioFrame instances
    mock_video_frame_instance = av.VideoFrame(width=640, height=480, format="bgr24")
    mock_audio_frame_instance = av.AudioFrame(format='s16', layout='stereo', samples=1024)

    # Mock demux method to yield actual Av.Packet instances
    mock_packet_video = MagicMock(spec=av.Packet)
    mock_packet_video.stream = mock_video_stream
    mock_packet_video.stream.type = 'video'
    mock_packet_video.decode.return_value = iter([mock_video_frame_instance])

    mock_packet_audio = MagicMock(spec=av.Packet)
    mock_packet_audio.stream = mock_audio_stream
    mock_packet_audio.stream.type = 'audio'
    mock_packet_audio.decode.return_value = iter([mock_audio_frame_instance]) # Should yield actual Av.AudioFrame

    # The iterator for demux should yield these AvPacket objects
    mock_container.demux.return_value = iter([mock_packet_video, mock_packet_audio])

    # Ensure mock_container.decode is not called
    mock_container.decode.return_value = iter([])
    mock_container.decode.assert_not_called = MagicMock()


    return mock_container

def test_video_file_reader_init_with_container(mock_av_container):
    """Test that VideoFileReader uses the provided av.Container instance."""
    reader = VideoFileReader("dummy_path.mp4", input_container_instance=mock_av_container)
    assert reader.input_container == mock_av_container
    assert reader.input_path == "dummy_path.mp4"

def test_video_file_reader_init_creates_container():
    """Test that VideoFileReader initializes input_container to None if not provided."""
    reader = VideoFileReader("dummy_path.mp4")
    assert reader.input_container is None

@patch('video_file_reader.av.open')
def test_open_success_with_av_open(mock_av_open, mock_av_container):
    """Test successful opening when av.open is called internally."""
    mock_av_open.return_value = mock_av_container
    reader = VideoFileReader("path/to/video.mp4")

    assert reader.open() is True
    mock_av_open.assert_called_once_with("path/to/video.mp4")
    assert reader.video_in == mock_av_container.streams.video[0]
    assert reader.audio_in == mock_av_container.streams.audio[0]
    mock_av_container.demux.assert_called_once_with(mock_av_container.streams.video[0], mock_av_container.streams.audio[0])


@patch('video_file_reader.av.open')
def test_open_failure_invalid_path(mock_av_open):
    """Test open failure with an invalid file path."""
    mock_av_open.side_effect = Exception("File not found")
    reader = VideoFileReader("invalid/path.mp4")

    assert reader.open() is False
    mock_av_open.assert_called_once_with("invalid/path.mp4")
    assert reader.video_in is None
    assert reader.audio_in is None

def test_open_success_with_injected_container(mock_av_container):
    """Test successful opening when container is injected."""
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)

    assert reader.open() is True
    # av.open should not be called if container is injected
    with patch('video_file_reader.av.open') as mock_av_open_inner:
        mock_av_open_inner.assert_not_called()

    assert reader.video_in == mock_av_container.streams.video[0]
    assert reader.audio_in == mock_av_container.streams.audio[0]
    mock_av_container.demux.assert_called_once_with(mock_av_container.streams.video[0], mock_av_container.streams.audio[0])


def test_read_frame_yields_video_and_audio(mock_av_container):
    """Test that read_frame correctly yields video frames (as numpy) and audio frames."""
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open()

    # Create mock packets and frames that will be yielded by reader.read_packets()
    mock_video_packet = MagicMock(spec=av.Packet)
    mock_video_packet.stream.type = 'video'
    mock_video_frame = MagicMock(spec=av.VideoFrame)
    mock_video_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_video_packet.decode.return_value = iter([mock_video_frame, None]) # Yield one video frame, then stop for this packet

    mock_audio_packet = MagicMock(spec=av.Packet)
    mock_audio_packet.stream.type = 'audio'
    mock_audio_frame = MagicMock(spec=av.AudioFrame)
    mock_audio_packet.decode.return_value = iter([mock_audio_frame]) # Yield one audio frame, then stop for this packet

    # Configure reader.read_packets to yield our mock packets
    # Note: reader.open() already calls iter(reader.read_packets()), so we need to mock it
    # AFTER reader.open() or ensure the mock is set on the 'instance'.
    # For simplicity, let's patch the instance's 'packets' iterator directly.
    reader.packets = iter([mock_video_packet, mock_audio_packet])

    # First call: Should get video frame
    video_frame_np, audio_frame_av = reader.read_frame()
    assert video_frame_np is not None
    assert isinstance(video_frame_np, np.ndarray)
    assert audio_frame_av is None

    # Second call: Should get audio packet
    video_frame_np, audio_frame_av = reader.read_frame()
    assert video_frame_np is None
    assert isinstance(audio_frame_av, av.Packet) # Changed from av.AudioFrame

    # Third call: Should get None, None (StopIteration from reader.packets)
    video_frame_np, audio_frame_av = reader.read_frame()
    assert video_frame_np is None
    assert audio_frame_av is None


def test_read_frame_not_opened_raises_error():
    """Test that read_frame raises RuntimeError if input container is not opened."""
    reader = VideoFileReader("path/to/video.mp4")
    with pytest.raises(RuntimeError, match="Input container not opened"):
        reader.read_frame()

def test_get_frame_size(mock_av_container):
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open()
    width, height = reader.get_frame_size()
    assert width == 640
    assert height == 480

def test_get_fps(mock_av_container):
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open()
    fps = reader.get_fps()
    assert fps == 25.0

def test_get_duration():
    reader = VideoFileReader("dummy.mp4")
    assert reader.get_duration() == 0.0 # Default value if not open or unknown

def test_get_audio_layout(mock_av_container):
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open()
    layout = reader.get_audio_layout()
    assert layout == av.AudioLayout('stereo')

def test_get_audio_rate(mock_av_container):
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open()
    rate = reader.get_audio_rate()
    assert rate == 44100


def test_close_container(mock_av_container):
    """Test that close method calls close on the container."""
    reader = VideoFileReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open() # Open to ensure container is set

    reader.close()
    mock_av_container.close.assert_called_once()
    assert reader.input_container is None # Ensure container is dereferenced
    assert reader.video_in is None
    assert reader.audio_in is None
    assert reader.packets is None
    assert reader.video_frames is None

def test_close_no_open_does_nothing():
    """Test that close does nothing if container was never opened."""
    reader = VideoFileReader("path/to/video.mp4")
    reader.close()
    # Ensure no error and no attempt to close a non-existent container
    assert reader.input_container is None
