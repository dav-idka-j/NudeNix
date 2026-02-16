import pytest
import sys
import os
from unittest.mock import patch

from main import parse_args

@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test_video.mp4"
    file_path.touch()
    return str(file_path)

def test_parse_args_file_to_file_success(temp_file):
    """Tests successful parsing for a file input and file output."""
    args_list = ["-i", f"file://{temp_file}", "-o", "file:///tmp/output.mp4"]
    parsed_args = parse_args(args_list)
    assert parsed_args.input == f"file://{temp_file}"
    assert parsed_args.output == "file:///tmp/output.mp4"
    assert parsed_args.format == "mp4"

def test_parse_args_screen_to_overlay_success():
    """Tests successful parsing for screen input and overlay output."""
    args_list = ["-i", "screen://1", "-o", "overlay://"]
    parsed_args = parse_args(args_list)
    assert parsed_args.input == "screen://1"
    assert parsed_args.output == "overlay://"
    assert parsed_args.format is None

def test_parse_args_missing_input_file_error(tmp_path):
    """Tests that a non-existent input file URI raises an error."""
    non_existent_file = tmp_path / "non_existent.mp4"
    args_list = ["-i", f"file://{non_existent_file}", "-o", "file:///tmp/output.mp4"]
    with pytest.raises(SystemExit):
        with patch('sys.stderr'): # Suppress error message print
            parse_args(args_list)

def test_parse_args_invalid_input_scheme_error():
    """Tests that an unsupported input scheme raises an error."""
    args_list = ["-i", "http://example.com/video.mp4", "-o", "file:///tmp/output.mp4"]
    with pytest.raises(SystemExit):
        with patch('sys.stderr'):
            parse_args(args_list)

def test_parse_args_invalid_output_scheme_error(temp_file):
    """Tests that an unsupported output scheme raises an error."""
    args_list = ["-i", f"file://{temp_file}", "-o", "ftp://server/output.mp4"]
    with pytest.raises(SystemExit):
        with patch('sys.stderr'):
            parse_args(args_list)

def test_parse_args_invalid_screen_monitor_error():
    """Tests that an invalid monitor specifier for screen input raises an error."""
    args_list = ["-i", "screen://abc", "-o", "overlay://"]
    with pytest.raises(SystemExit):
        with patch('sys.stderr'):
            parse_args(args_list)

def test_parse_args_overwrite_confirmation_yes(temp_file):
    """Tests the overwrite confirmation prompt with a 'y' response."""
    args_list = ["-i", f"file://{temp_file}", "-o", f"file://{temp_file}"]
    with patch('builtins.input', return_value='y') as mock_input:
        parse_args(args_list)
        mock_input.assert_called_once()

def test_parse_args_overwrite_confirmation_no(temp_file):
    """Tests that the overwrite confirmation with a 'n' response exits."""
    args_list = ["-i", f"file://{temp_file}", "-o", f"file://{temp_file}"]
    with pytest.raises(SystemExit):
        with patch('builtins.input', return_value='n'):
            parse_args(args_list)

def test_parse_args_force_overwrite(temp_file):
    """Tests that --force-overwrite skips the confirmation prompt."""
    args_list = ["-i", f"file://{temp_file}", "-o", f"file://{temp_file}", "-y"]
    with patch('builtins.input') as mock_input:
        parse_args(args_list)
        mock_input.assert_not_called()

def test_parse_args_format_inference():
    """Tests that the output format is correctly inferred from the file extension."""
    args_list = ["-i", "screen://1", "-o", "file:///tmp/video.webm"]
    parsed_args = parse_args(args_list)
    assert parsed_args.format == "webm"

    args_list_no_ext = ["-i", "screen://1", "-o", "file:///tmp/video"]
    parsed_args_no_ext = parse_args(args_list_no_ext)
    assert parsed_args_no_ext.format == "mp4" # Default case
