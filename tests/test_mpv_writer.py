import pytest
from unittest.mock import MagicMock, patch
import os
import av
from output.mpv_writer import MpvWriter
from input.video_file_reader import VideoFileReader

@pytest.fixture
def mock_video_reader():
    mock_reader = MagicMock(spec=VideoFileReader)
    mock_reader.get_fps.return_value = 25.0
    mock_reader.get_frame_size.return_value = (640, 480)
    return mock_reader

@patch('output.mpv_writer.mpv')
@patch('output.mpv_writer.VideoFileWriter')
def test_mpv_writer_init(mock_vfw, mock_mpv, mock_video_reader):
    writer = MpvWriter(mock_video_reader)
    assert writer.temp_file_path.endswith('.mkv')
    assert os.path.exists(writer.temp_file_path)
    mock_vfw.assert_called_once_with(writer.temp_file_path, mock_video_reader, output_format='matroska')
    writer.close()
    assert not os.path.exists(writer.temp_file_path)

@patch('output.mpv_writer.mpv')
@patch('output.mpv_writer.VideoFileWriter')
def test_mpv_writer_open(mock_vfw, mock_mpv, mock_video_reader):
    mock_vfw_instance = mock_vfw.return_value
    mock_vfw_instance.open.return_value = True

    writer = MpvWriter(mock_video_reader)
    assert writer.open() is True
    # MPV is no longer initialized in open()
    mock_mpv.MPV.assert_not_called()
    writer.close()

@patch('output.mpv_writer.os.path.getsize')
@patch('output.mpv_writer.mpv')
@patch('output.mpv_writer.VideoFileWriter')
def test_mpv_writer_write_frame_starts_playback(mock_vfw, mock_mpv, mock_os_getsize, mock_video_reader):
    mock_vfw_instance = mock_vfw.return_value
    mock_vfw_instance.open.return_value = True
    mock_vfw_instance.get_fps.return_value = 25.0
    mock_os_getsize.return_value = 1000 # File is not empty
    mock_player = mock_mpv.MPV.return_value

    writer = MpvWriter(mock_video_reader)
    writer.open()

    frame = MagicMock(spec=av.VideoFrame)

    # Write frames until threshold
    for _ in range(24):
        writer.write_video_frame(frame)
        assert writer.playback_started is False
        mock_player.play.assert_not_called()

    writer.write_video_frame(frame)
    assert writer.playback_started is True
    mock_player.play.assert_called_once_with(writer.temp_file_path)

    writer.close()

@patch('output.mpv_writer.os.path.getsize')
@patch('output.mpv_writer.mpv')
@patch('output.mpv_writer.VideoFileWriter')
def test_mpv_writer_close_waits_for_playback(mock_vfw, mock_mpv, mock_os_getsize, mock_video_reader):
    mock_vfw_instance = mock_vfw.return_value
    mock_vfw_instance.open.return_value = True
    mock_vfw_instance.get_fps.return_value = 10.0
    mock_os_getsize.return_value = 1000
    mock_player = mock_mpv.MPV.return_value

    writer = MpvWriter(mock_video_reader)
    writer.open()

    # Trigger playback
    frame = MagicMock(spec=av.VideoFrame)
    for _ in range(10):
        writer.write_video_frame(frame)

    writer.close()

    mock_vfw_instance.close.assert_called_once()
    mock_player.wait_for_playback.assert_called_once()
    mock_player.terminate.assert_called_once()
    assert not os.path.exists(writer.temp_file_path)
