import av
import sys
from typing import Optional, Union, Tuple
from fractions import Fraction
from interfaces.base_frame_writer import BaseFrameWriter
from interfaces.base_frame_reader import BaseFrameReader

class VideoFileWriter(BaseFrameWriter):
    def __init__(
        self,
        output_destination: str,
        reader_instance: BaseFrameReader,
        output_format: str = 'mp4',
        output_container_instance: Optional[av.container.OutputContainer] = None,
        output_video_stream_instance: Optional[av.VideoStream] = None,
        output_audio_stream_instance: Optional[av.AudioStream] = None
    ):
        self.output_destination = output_destination
        self.output_format = output_format
        self.reader = reader_instance

        self.output_container = output_container_instance
        self.video_out = output_video_stream_instance
        self.audio_out = output_audio_stream_instance

    def open(self) -> bool:
        try:
            if self.output_container is None:
                self.output_container = av.open(self.output_destination, 'w', format=self.output_format)

            video_in = self.reader.get_video_stream()
            audio_in = self.reader.get_audio_stream()

            if self.video_out is None:
                fps = self.reader.get_fps()
                # Use Fraction for rate to avoid floating point issues and satisfy PyAV
                rate = Fraction(fps).limit_denominator()

                if self.output_format in ['mp4', 'matroska', 'mkv']:
                    # Use libx264 with low latency settings for streaming
                    options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
                    try:
                        self.video_out = self.output_container.add_stream('libx264', rate=rate, options=options)
                    except Exception:
                        # Fallback to 'h264' if 'libx264' name is not recognized by the build
                        self.video_out = self.output_container.add_stream('h264', rate=rate, options=options)
                elif self.output_format == 'webm':
                    self.video_out = self.output_container.add_stream('libvpx', rate=rate)
                else:
                    raise ValueError(f"Unsupported video output format: {self.output_format}")

                width, height = self.reader.get_frame_size()
                self.video_out.width = width
                self.video_out.height = height
                self.video_out.pix_fmt = 'yuv420p'

            if audio_in and self.audio_out is None:
                if self.output_format in ['mp4', 'matroska', 'mkv']:
                    # For MP4 and MKV, remux audio if template is available
                    self.audio_out = self.output_container.add_stream_from_template(audio_in)
                elif self.output_format == 'webm':
                    # For WebM, transcode to Opus
                    self.audio_out = self.output_container.add_stream('opus')
                    self.audio_out.codec_context.rate = self.reader.get_audio_rate()
                    self.audio_out.codec_context.channels = audio_in.channels
                    self.audio_out.codec_context.layout = self.reader.get_audio_layout()
                else:
                    raise ValueError(f"Unsupported audio output format: {self.output_format}")

            return True
        except Exception as e:
            print(f"Error setting up output: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False

    def write_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Writes a single video frame to the output.
        Expects an av.VideoFrame that is already properly formatted (e.g., pix_fmt, dimensions).
        """
        if not self.video_out:
            return

        for packet in self.video_out.encode(frame):
            self.output_container.mux(packet)

    def write_audio_packet(self, packet: av.Packet) -> None:
        if not self.audio_out:
            return

        if self.output_format in ['mp4', 'matroska', 'mkv']:
            # Remux audio
            if packet.dts is None:
                return
            packet.stream = self.audio_out

            # Manually rescale timestamps to output stream timebase
            # Ensure both timebases are available before rescaling
            if packet.time_base is not None and self.audio_out.time_base is not None:
                if packet.pts is not None:
                    packet.pts = int(packet.pts * packet.time_base / self.audio_out.time_base)
                if packet.dts is not None:
                    packet.dts = int(packet.dts * packet.time_base / self.audio_out.time_base)
                if packet.duration is not None:
                    packet.duration = int(packet.duration * packet.time_base / self.audio_out.time_base)

            self.output_container.mux(packet)
        elif self.output_format == 'webm':
            # Transcode audio
            for frame in packet.decode():
                for out_packet in self.audio_out.encode(frame):
                    self.output_container.mux(out_packet)

    def flush_encoders(self) -> None:
        if self.video_out:
            for packet in self.video_out.encode(None):
                self.output_container.mux(packet)
        if self.audio_out and self.output_format == 'webm': # Only flush audio if transcoded
            for packet in self.audio_out.encode(None):
                self.output_container.mux(packet)

    def close(self) -> None:
        if self.output_container:
            try:
                self.output_container.close()
            except Exception as e:
                print(f"Error closing output container: {e}", file=sys.stderr)
        self.output_container = None
        self.video_out = None
        self.audio_out = None

    def get_fps(self) -> float:
        """Returns the FPS from the reader."""
        return self.reader.get_fps()

    def get_frame_size(self) -> Tuple[int, int]:
        """Returns the frame size from the reader."""
        return self.reader.get_frame_size()

    def get_audio_layout(self) -> Optional[av.AudioLayout]:
        """Returns the audio layout from the reader."""
        return self.reader.get_audio_layout()

    def get_audio_rate(self) -> Optional[int]:
        """Returns the audio rate from the reader."""
        return self.reader.get_audio_rate()

    def get_video_time_base(self) -> Optional[Fraction]:
        """Returns the time base of the output video stream."""
        if self.video_out:
            return self.video_out.time_base
        return None
