import av
import sys

class VideoOutputWriter:
    def __init__(self, output_destination, reader_instance, output_format='mp4',
                 output_container_instance=None, output_video_stream_instance=None, output_audio_stream_instance=None):
        self.output_destination = output_destination
        self.output_format = output_format
        self.reader = reader_instance # Store the VideoReader instance

        self.output_container = output_container_instance
        self.video_out = output_video_stream_instance
        self.audio_out = output_audio_stream_instance

    def open(self):
        try:
            if self.output_container is None:
                self.output_container = av.open(self.output_destination, 'w', format=self.output_format)

            # --- Setup Output Video Stream ---
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
                print(f"Output video stream created ({self.output_format}): {self.video_out.name}")
            elif self.reader.video_in and self.video_out is not None:
                print("Using injected output video stream.")

            # --- Setup Output Audio Stream ---
            if self.reader.audio_in and self.audio_out is None:
                if self.output_format == 'mp4':
                    # For MP4, remux AAC audio directly
                    self.audio_out = self.output_container.add_stream_from_template(self.reader.audio_in)
                    print(f"Output audio stream (copied for MP4): {self.audio_out.name}")
                elif self.output_format == 'webm':
                    # For WebM, transcode to Opus
                    self.audio_out = self.output_container.add_stream('opus')
                    self.audio_out.codec_context.rate = self.reader.audio_in.rate
                    self.audio_out.codec_context.channels = self.reader.audio_in.channels
                    self.audio_out.codec_context.layout = self.reader.audio_in.layout
                    print(f"Output audio stream (Opus for WebM): {self.audio_out.name}")
                else:
                    raise ValueError(f"Unsupported audio output format: {self.output_format}")
            elif self.reader.audio_in and self.audio_out is not None:
                print("Using injected output audio stream.")
            return True
        except Exception as e:
            print(f"Error setting up output: {e}", file=sys.stderr)
            return False

    def write_video_frame(self, frame): # Removed frame_time parameter
        if not self.video_out:
            return

        # Ensure frame is in correct pixel format for encoding
        processed_frame = frame.reformat(
            width=self.video_out.width,
            height=self.video_out.height,
            format=self.video_out.pix_fmt
        )

        for packet in self.video_out.encode(processed_frame):
            self.output_container.mux(packet)

    def write_audio_packet(self, packet):
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

    def flush_encoders(self):
        if self.video_out:
            for packet in self.video_out.encode(None):
                self.output_container.mux(packet)
        if self.audio_out and self.output_format == 'webm': # Only flush audio if transcoded
            for packet in self.audio_out.encode(None):
                self.output_container.mux(packet)
        print("Encoders flushed.")

    def close(self):
        if self.output_container:
            self.output_container.close()
            print("Output container closed.")
