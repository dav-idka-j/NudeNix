import pytest
from unittest.mock import MagicMock, patch, call
from fractions import Fraction
import av
import numpy as np
from video_file_writer import VideoFileWriter
from video_file_reader import VideoFileReader # Import VideoFileReader for mocking


@pytest.fixture
def mock_video_reader():
    """Fixture for a mock VideoFileReader instance."""
    mock_reader = MagicMock(spec=VideoFileReader)
    mock_video_in = MagicMock(spec=av.VideoStream)
    mock_video_in.average_rate = 25
    mock_video_in.width = 640
    mock_video_in.height = 480

    mock_audio_in = MagicMock(spec=av.AudioStream)

    mock_audio_in.rate = 44100
    mock_audio_in.channels = 2
    mock_audio_in.layout = av.AudioLayout('stereo')

    mock_reader.get_fps.return_value = 25.0
    mock_reader.get_frame_size.return_value = (640, 480)
    mock_reader.get_video_stream.return_value = mock_video_in
    mock_reader.get_audio_stream.return_value = mock_audio_in
    mock_reader.get_audio_rate.return_value = 44100
    mock_reader.get_audio_layout.return_value = av.AudioLayout('stereo')

    return mock_reader

@pytest.fixture
def mock_output_container():
    """Fixture for a mock av.container.OutputContainer."""
    mock_container = MagicMock(spec=av.container.OutputContainer)
    return mock_container

def test_output_writer_init_with_injected_instances(mock_video_reader, mock_output_container):
    """Test that VideoFileWriter uses provided injected instances."""
    mock_output_video_stream = MagicMock(spec=av.VideoStream)
    mock_output_audio_stream = MagicMock(spec=av.AudioStream)

    writer = VideoFileWriter(
        "output.mp4",
        mock_video_reader,
        output_container_instance=mock_output_container,
        output_video_stream_instance=mock_output_video_stream,
        output_audio_stream_instance=mock_output_audio_stream
    )

    assert writer.output_container == mock_output_container
    assert writer.video_out == mock_output_video_stream
    assert writer.audio_out == mock_output_audio_stream
    assert writer.reader == mock_video_reader

@patch('video_file_writer.av.open')
def test_open_creates_container_and_streams_mp4(mock_av_open, mock_video_reader):
    """Test open method creates container and streams for MP4 output."""
    mock_container = MagicMock(spec=av.container.OutputContainer)
    mock_video_out = MagicMock(spec=av.VideoStream)
    mock_video_out.name = 'video_out_0' # Added name attribute
    mock_audio_out = MagicMock(spec=av.AudioStream)
    mock_audio_out.name = 'audio_out_0' # Added name attribute

    mock_av_open.return_value = mock_container
    mock_container.add_stream.side_effect = [mock_video_out] # Only video stream is added by this call
    mock_container.add_stream_from_template.return_value = mock_audio_out # Audio stream is added via template


    writer = VideoFileWriter("output.mp4", mock_video_reader, output_format='mp4')
    assert writer.open() is True

    mock_av_open.assert_called_once_with("output.mp4", 'w', format='mp4')
    # Check video stream setup
    from fractions import Fraction
    expected_rate = Fraction(mock_video_reader.get_fps.return_value).limit_denominator()
    mock_container.add_stream.assert_called_once_with('h264', rate=expected_rate)
    assert mock_video_out.width == mock_video_reader.get_frame_size.return_value[0]
    assert mock_video_out.height == mock_video_reader.get_frame_size.return_value[1]
    assert mock_video_out.pix_fmt == 'yuv420p'
    # Check audio stream setup
    mock_container.add_stream_from_template.assert_called_once_with(mock_video_reader.get_audio_stream.return_value)


@patch('video_file_writer.av.open')
def test_open_creates_container_and_streams_webm(mock_av_open, mock_video_reader):
    """Test open method creates container and streams for WebM output."""
    mock_container = MagicMock(spec=av.container.OutputContainer)
    mock_video_out = MagicMock(spec=av.VideoStream)
    mock_video_out.name = 'video_out_0' # Added name attribute
    mock_audio_out = MagicMock(spec=av.AudioStream)
    mock_audio_out.name = 'audio_out_0' # Added name attribute

    mock_av_open.return_value = mock_container
    # Ensure correct side_effect for multiple add_stream calls (video then audio)
    mock_container.add_stream.side_effect = [mock_video_out, mock_audio_out]


    writer = VideoFileWriter("output.webm", mock_video_reader, output_format='webm')
    assert writer.open() is True

    mock_av_open.assert_called_once_with("output.webm", 'w', format='webm')
    # Check video stream setup
    from fractions import Fraction
    expected_rate = Fraction(mock_video_reader.get_fps.return_value).limit_denominator()
    assert mock_container.add_stream.call_args_list[0] == call('libvpx', rate=expected_rate)
    assert mock_video_out.width == mock_video_reader.get_frame_size.return_value[0]
    assert mock_video_out.height == mock_video_reader.get_frame_size.return_value[1]
    assert mock_video_out.pix_fmt == 'yuv420p'
    # Check audio stream setup
    assert mock_container.add_stream.call_args_list[1] == call('opus')
    assert mock_audio_out.codec_context.rate == mock_video_reader.get_audio_rate.return_value
    assert mock_audio_out.codec_context.channels == mock_video_reader.get_audio_stream.return_value.channels
    assert mock_audio_out.codec_context.layout == mock_video_reader.get_audio_layout.return_value # Now retrieves layout directly

def test_open_uses_injected_container_and_streams(mock_video_reader, mock_output_container):
    """Test open method uses injected container and streams without re-creating."""
    mock_video_out = MagicMock(spec=av.VideoStream)
    mock_audio_out = MagicMock(spec=av.AudioStream)

    writer = VideoFileWriter(
        "output.mp4",
        mock_video_reader,
        output_container_instance=mock_output_container,
        output_video_stream_instance=mock_video_out,
        output_audio_stream_instance=mock_audio_out
    )

    with patch('video_file_writer.av.open') as mock_av_open:
        assert writer.open() is True
        mock_av_open.assert_not_called() # av.open should not be called

    # assert add_stream not called if streams are injected
    mock_output_container.add_stream.assert_not_called()
    mock_output_container.add_stream_from_template.assert_not_called()

    assert writer.video_out == mock_video_out
    assert writer.audio_out == mock_audio_out

def test_write_video_frame(mock_video_reader, mock_output_container):
    """Test write_video_frame encodes and muxes a video frame."""
    mock_video_out = MagicMock(spec=av.VideoStream)
    mock_video_out.pix_fmt = 'yuv420p'
    mock_output_container.mux.reset_mock() # Clear calls from setup

    writer = VideoFileWriter(
        "output.mp4",
        mock_video_reader,
        output_container_instance=mock_output_container,
        output_video_stream_instance=mock_video_out
    )
    writer.open() # Ensure video_out is properly set up with width/height/pix_fmt

    mock_av_frame = MagicMock(spec=av.VideoFrame)
    # The reformat call is no longer in VideoFileWriter's write_video_frame
    # It's assumed the incoming frame is already correctly formatted.
    # mock_av_frame.reformat.return_value = mock_av_frame
    mock_video_out.encode.return_value = [MagicMock(spec=av.Packet)] # Mock an encoded packet

    writer.write_video_frame(mock_av_frame)

    # mock_av_frame.reformat.assert_called_once_with no longer relevant
    # assert False, "Reformat assert removed, confirm new behavior if needed."

    mock_video_out.encode.assert_called_once_with(mock_av_frame)
    mock_output_container.mux.assert_called_once()

def test_write_audio_packet_mp4(mock_video_reader, mock_output_container):
    """Test write_audio_packet remuxes for MP4 output."""
    mock_audio_out = MagicMock(spec=av.AudioStream)
    mock_output_container.mux.reset_mock()

    writer = VideoFileWriter(
        "output.mp4",
        mock_video_reader,
        output_container_instance=mock_output_container,
        output_audio_stream_instance=mock_audio_out,
        output_format='mp4'
    )
    writer.open()

    mock_av_packet = MagicMock()
    mock_av_packet.pts = 100
    mock_av_packet.dts = 100
    mock_av_packet.duration = 10
    mock_av_packet.time_base = Fraction(1, 44100)
    mock_audio_out.time_base = Fraction(1, 44100)

    writer.write_audio_packet(mock_av_packet)

    assert mock_av_packet.stream == mock_audio_out # Stream should be updated
    mock_output_container.mux.assert_called_once_with(mock_av_packet)

def test_write_audio_packet_webm(mock_video_reader, mock_output_container):
    """Test write_audio_packet transcodes for WebM output."""
    mock_audio_out = MagicMock(spec=av.AudioStream)
    mock_output_container.mux.reset_mock()

    writer = VideoFileWriter(
        "output.webm",
        mock_video_reader,
        output_container_instance=mock_output_container,
        output_audio_stream_instance=mock_audio_out,
        output_format='webm'
    )
    writer.open()

    mock_audio_frame = MagicMock(spec=av.AudioFrame)
    mock_in_packet = MagicMock(spec=av.Packet)
    mock_in_packet.decode.return_value = [mock_audio_frame] # Mock an audio frame from input packet
    mock_audio_out.encode.return_value = [MagicMock(spec=av.Packet)] # Mock an encoded packet

    writer.write_audio_packet(mock_in_packet)

    mock_in_packet.decode.assert_called_once()
    mock_audio_out.encode.assert_called_once_with(mock_audio_frame)
    mock_output_container.mux.assert_called_once()

def test_flush_encoders(mock_video_reader, mock_output_container):
    """Test flush_encoders flushes video and audio encoders."""
    mock_video_out = MagicMock(spec=av.VideoStream)
    mock_audio_out = MagicMock(spec=av.AudioStream) # WebM case
    mock_output_container.mux.reset_mock()

    writer = VideoFileWriter(
        "output.webm",
        mock_video_reader,
        output_container_instance=mock_output_container,
        output_video_stream_instance=mock_video_out,
        output_audio_stream_instance=mock_audio_out,
        output_format='webm'
    )
    writer.open()

    mock_video_out.encode.return_value = [MagicMock(spec=av.Packet)]
    mock_audio_out.encode.return_value = [MagicMock(spec=av.Packet)]

    writer.flush_encoders()

    mock_video_out.encode.assert_called_once_with(None) # Flush video
    mock_audio_out.encode.assert_called_once_with(None) # Flush audio (for webm)
    assert mock_output_container.mux.call_count == 2 # One for video, one for audio

def test_close_container(mock_output_container, mock_video_reader):
    """Test that close method calls close on the output container."""
    writer = VideoFileWriter(
        "output.mp4",
        mock_video_reader, # Dummy
        output_container_instance=mock_output_container
    )
    writer.open() # Open to ensure container is set

    writer.close()
    mock_output_container.close.assert_called_once()

def test_close_no_open_does_nothing(mock_video_reader):
    """Test that close does nothing if container was never opened."""
    writer = VideoFileWriter("output.mp4", mock_video_reader)
    writer.close()
    # Ensure no error and no attempt to close a non-existent container
    assert writer.output_container is None
