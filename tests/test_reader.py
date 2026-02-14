import pytest
from unittest.mock import MagicMock, patch
import av
from reader import VideoReader

@pytest.fixture
def mock_av_container():
    """Fixture for a mock av.Container."""
    mock_container = MagicMock(spec=av.container.InputContainer)

    # Mock video stream
    mock_video_stream = MagicMock(spec=av.VideoStream)
    mock_video_stream.name = 'video_stream_0'

    # Mock audio stream
    mock_audio_stream = MagicMock(spec=av.AudioStream)
    mock_audio_stream.name = 'audio_stream_0'

    # Mock container streams property
    mock_container.streams.video = [mock_video_stream]
    mock_container.streams.audio = [mock_audio_stream]

    # Mock demux method to yield packets
    mock_packet_video = MagicMock(spec=av.Packet)
    mock_packet_video.stream = mock_video_stream
    mock_packet_audio = MagicMock(spec=av.Packet)
    mock_packet_audio.stream = mock_audio_stream

    # Configure demux to return a sequence of packets
    mock_container.demux.return_value = iter([mock_packet_video, mock_packet_audio])

    return mock_container

def test_video_reader_init_with_container(mock_av_container):
    """Test that VideoReader uses the provided av.Container instance."""
    reader = VideoReader("dummy_path.mp4", input_container_instance=mock_av_container)
    assert reader.input_container == mock_av_container
    assert reader.input_path == "dummy_path.mp4"

def test_video_reader_init_creates_container():
    """Test that VideoReader initializes input_container to None if not provided."""
    reader = VideoReader("dummy_path.mp4")
    assert reader.input_container is None

@patch('reader.av.open')
def test_open_success_with_av_open(mock_av_open, mock_av_container):
    """Test successful opening when av.open is called internally."""
    mock_av_open.return_value = mock_av_container
    reader = VideoReader("path/to/video.mp4")

    assert reader.open() is True
    mock_av_open.assert_called_once_with("path/to/video.mp4")
    assert reader.video_in == mock_av_container.streams.video[0]
    assert reader.audio_in == mock_av_container.streams.audio[0]

@patch('reader.av.open')
def test_open_failure_invalid_path(mock_av_open):
    """Test open failure with an invalid file path."""
    mock_av_open.side_effect = Exception("File not found")
    reader = VideoReader("invalid/path.mp4")

    assert reader.open() is False
    mock_av_open.assert_called_once_with("invalid/path.mp4")
    assert reader.video_in is None
    assert reader.audio_in is None

def test_open_success_with_injected_container(mock_av_container):
    """Test successful opening when container is injected."""
    reader = VideoReader("path/to/video.mp4", input_container_instance=mock_av_container)

    assert reader.open() is True
    # av.open should not be called if container is injected
    with patch('reader.av.open') as mock_av_open_inner:
        mock_av_open_inner.assert_not_called()

    assert reader.video_in == mock_av_container.streams.video[0]
    assert reader.audio_in == mock_av_container.streams.audio[0]

def test_read_packets_yields_all_packets(mock_av_container):
    """Test that read_packets yields all packets from the container."""
    reader = VideoReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open() # Need to call open to set video_in, audio_in

    packets = list(reader.read_packets())
    assert len(packets) == 2
    assert packets[0].stream == reader.video_in
    assert packets[1].stream == reader.audio_in
    mock_av_container.demux.assert_called_once()

def test_read_packets_not_opened_raises_error():
    """Test that read_packets raises RuntimeError if container is not opened."""
    reader = VideoReader("path/to/video.mp4")
    with pytest.raises(RuntimeError, match="Input container not opened"):
        list(reader.read_packets())

def test_close_container(mock_av_container):
    """Test that close method calls close on the container."""
    reader = VideoReader("path/to/video.mp4", input_container_instance=mock_av_container)
    reader.open() # Open to ensure container is set

    reader.close()
    mock_av_container.close.assert_called_once()

def test_close_no_open_does_nothing():
    """Test that close does nothing if container was never opened."""
    reader = VideoReader("path/to/video.mp4")
    reader.close()
    # Ensure no error and no attempt to close a non-existent container
    assert reader.input_container is None
