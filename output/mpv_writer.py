import tempfile
import os
try:
    import mpv
except OSError:
    mpv = None
import av
import logging
from typing import Optional, Tuple
from fractions import Fraction
from interfaces.base_frame_writer import BaseFrameWriter
from interfaces.base_frame_reader import BaseFrameReader
from video_file_writer import VideoFileWriter

logger = logging.getLogger(__name__)

class MpvWriter(BaseFrameWriter):
    def __init__(self, reader_instance: BaseFrameReader):
        self.reader = reader_instance
        # Use a temporary file with .mkv extension for MPV to read while it's being written
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.mkv', delete=False)
        self.temp_file_path = self.temp_file.name
        self.temp_file.close() # Close it so VideoFileWriter can open it

        self.file_writer = VideoFileWriter(
            self.temp_file_path,
            self.reader,
            output_format='matroska'
        )
        self.player = None
        self.frame_count = 0
        self.playback_started = False

    def open(self) -> bool:
        if mpv is None:
            logger.error("MPV library (libmpv) not found. Please install mpv.")
            return False

        if not self.file_writer.open():
            return False

        # We'll initialize the player later when we actually have data to show
        return True

    def write_video_frame(self, frame: av.VideoFrame) -> None:
        self.file_writer.write_video_frame(frame)
        self.frame_count += 1

        # Start playback after a small buffer is written
        if not self.playback_started:
            fps = self.get_fps()
            start_threshold = max(10, int(fps))
            if self.frame_count >= start_threshold:
                file_size = os.path.getsize(self.temp_file_path)
                if file_size > 0:
                    logger.info(f"Starting MPV playback of {self.temp_file_path} (Size: {file_size} bytes)")
                    try:
                        self.player = mpv.MPV(
                            input_default_bindings=True,
                            input_vo_keyboard=True,
                            osc=True,
                            title="NudeNix - MPV Output",
                            loglevel="warn",
                            demuxer_lavf_format='matroska',
                            ytdl=False,
                            keep_open='yes', # Don't close when reaching "end" of file
                            cache='yes',
                            demuxer_max_bytes=1024*1024*50, # 50MB buffer
                            demuxer_max_back_bytes=1024*1024*50
                        )
                        self.player.play(self.temp_file_path)
                        self.playback_started = True
                    except Exception as e:
                        print(f"CRITICAL ERROR: Failed to start MPV playback: {e}")
                        logger.error(f"Failed to start MPV playback: {e}")
                else:
                    # If file is still 0, we'll try again on the next frame
                    if self.frame_count % 10 == 0:
                        logger.debug(f"Frame {self.frame_count} reached, but file is still empty. Waiting for data...")

    def write_audio_packet(self, packet: av.Packet) -> None:
        self.file_writer.write_audio_packet(packet)

    def flush_encoders(self) -> None:
        self.file_writer.flush_encoders()

    def close(self) -> None:
        # First close the file writer to ensure all data is flushed to disk
        self.file_writer.close()

        if self.player:
            # If playback started, wait for the window to be closed
            if self.playback_started:
                print("Processing finished. Waiting for MPV window to be closed...")
                try:
                    self.player.wait_for_playback()
                except KeyboardInterrupt:
                    pass
            self.player.terminate()

        # Clean up the temporary file
        if os.path.exists(self.temp_file_path):
            try:
                os.remove(self.temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {self.temp_file_path}: {e}")

    def get_fps(self) -> float:
        return self.file_writer.get_fps()

    def get_frame_size(self) -> Tuple[int, int]:
        return self.file_writer.get_frame_size()

    def get_audio_layout(self) -> Optional[av.AudioLayout]:
        return self.file_writer.get_audio_layout()

    def get_audio_rate(self) -> Optional[int]:
        return self.file_writer.get_audio_rate()

    def get_video_time_base(self) -> Optional[Fraction]:
        return self.file_writer.get_video_time_base()
