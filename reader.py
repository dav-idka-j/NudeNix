import av
import sys

class VideoReader:
    def __init__(self, input_path, input_container_instance=None):
        self.input_path = input_path
        self.input_container = input_container_instance
        self.audio_in = None
        self.video_in = None

    def open(self):
        if self.input_container is None:
            try:
                self.input_container = av.open(self.input_path)
            except Exception as e:
                print(f"Error opening input video: {e}", file=sys.stderr)
                return False

        # Check streams on the existing or newly opened container
        if self.input_container.streams.audio:
            self.audio_in = self.input_container.streams.audio[0]
            print(f"Input audio stream: {self.audio_in.name}")
        else:
            print("No audio stream found.")

        if self.input_container.streams.video:
            self.video_in = self.input_container.streams.video[0]
            print(f"Input video stream: {self.video_in.name}")
        else:
            print("No video stream found. Exiting.")
            return False
        return True

    def read_packets(self):
        if not self.input_container:
            raise RuntimeError("Input container not opened. Call .open() first.")

        for packet in self.input_container.demux():
            yield packet

    def close(self):
        if self.input_container:
            self.input_container.close()
            print("Input container closed.")
