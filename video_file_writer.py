from interfaces.base_frame_writer import BaseFrameWriter
import av
import sys
from typing import Optional, Union

class VideoFileWriter(BaseFrameWriter):
    def __init__(self, output_destination: str, reader_instance, output_format: str = 'mp4',
                 output_container_instance: Optional[av.container.OutputContainer] = None, 
                 output_video_stream_instance: Optional[av.VideoStream] = None, 
                 output_audio_stream_instance: Optional[av.AudioStream] = None):
        self.output_destination = output_destination
        self.output_format = output_format
        self.reader = reader_instance # Store the VideoFileReader instance (formerly VideoReader)

        self.output_container = output_container_instance
        self.video_out = output_video_stream_instance
        self.audio_out = output_audio_stream_instance

    def open(self) -> bool:
        try:
            if self.output_container is None:
                self.output_container = av.open(self.output_destination, 'w', format=self.output_format)

            # --- Setup Output Video Stream ---
            # Ensure reader_instance has video_in and output_video_stream_instance is not pre-injected
            if self.reader.video_in and self.video_out is None:
                if self.output_format == 'mp4':
                    self.video_out = self.output_container.add_stream('h264', rate=self.reader.video_in.average_rate)
                elif self.output_format == 'webm':
                    self.video_out = self.output_container.add_stream('libvpx', rate=self.reader.video_in.average_rate)
                else:
                    raise ValueError(f"Unsupported video output format: {self.output_format}")

                self.video_out.width = self.reader.video_in.width
                self.video_out.height = self.reader.video_in.height
                self.video_out.pix_fmt = 'yuv420p' # Common for h264/libvpx
                # print(f"Output video stream created ({self.output_format}): {self.video_out.name}") # Commented for cleaner output
            elif self.reader.video_in and self.video_out is not None:
                # print("Using injected output video stream.") # Commented for cleaner output
                pass


            # --- Setup Output Audio Stream ---
            # Ensure reader_instance has audio_in and output_audio_stream_instance is not pre-injected
            if self.reader.audio_in and self.audio_out is None:
                if self.output_format == 'mp4':
                    # For MP4, remux AAC audio directly
                    self.audio_out = self.output_container.add_stream_from_template(self.reader.audio_in)
                    # print(f"Output audio stream (copied for MP4): {self.audio_out.name}") # Commented for cleaner output
                elif self.output_format == 'webm':
                    # For WebM, transcode to Opus
                    self.audio_out = self.output_container.add_stream('opus')
                    # Make sure to set codec context parameters for the new stream
                    self.audio_out.codec_context.rate = self.reader.get_audio_rate() # Use reader's get_audio_rate
                    self.audio_out.codec_context.channels = self.reader.audio_in.channels
                    self.audio_out.codec_context.layout = self.reader.get_audio_layout() # Use reader's get_audio_layout
                    # print(f"Output audio stream (Opus for WebM): {self.audio_out.name}") # Commented for cleaner output
                else:
                    raise ValueError(f"Unsupported audio output format: {self.output_format}")
            elif self.reader.audio_in and self.audio_out is not None:
                # print("Using injected output audio stream.") # Commented for cleaner output
                pass
            return True
        except Exception as e:
            print(f"Error setting up output: {e}", file=sys.stderr)
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

        if self.output_format == 'mp4':
            # Remux audio
            if packet.dts is None:
                return
            packet.stream = self.audio_out
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
        # print("Encoders flushed.") # Commented for cleaner output

    def close(self) -> None:
        if self.output_container:
            self.output_container.close()
            # print("Output container closed.") # Commented for cleaner output
        self.output_container = None
        self.video_out = None
        self.audio_out = None
