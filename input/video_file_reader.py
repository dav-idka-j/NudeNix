from interfaces.base_frame_reader import BaseFrameReader
import av
import sys
import numpy as np
from typing import Tuple, Union, Optional

class VideoFileReader(BaseFrameReader):
    def __init__(self, input_path: str, input_container_instance: Optional[av.container.InputContainer] = None):
        self.input_path = input_path
        self.input_container = input_container_instance
        self.audio_in: Optional[av.AudioStream] = None
        self.video_in: Optional[av.VideoStream] = None
        self.packets = None

    def open(self) -> bool:
        if self.input_container is None:
            try:
                self.input_container = av.open(self.input_path)
            except Exception as e:
                print(f"Error opening input video: {e}", file=sys.stderr)
                return False

        if self.input_container.streams.audio:
            self.audio_in = self.input_container.streams.audio[0]
        else:
            pass

        if self.input_container.streams.video:
            self.video_in = self.input_container.streams.video[0]
        else:
            print("No video stream found. Exiting.", file=sys.stderr)
            return False

        self.packets = iter(self.read_packets())
        self.video_frames = None

        return True

    def read_frame(self) -> Tuple[Optional[np.ndarray], Optional[av.Packet], Optional[int]]:
        if not self.packets:
            raise RuntimeError("Input container not opened or not initialized. Call .open() first.")

        while True: # Loop to find the next valid frame or audio
            # 1. Try to get a video frame from the current video packet's iterator
            if self.video_frames is not None:
                try:
                    frame = next(self.video_frames)
                    if frame is not None:
                        video_frame_np = frame.to_ndarray(format="bgr24")
                        pts_seconds = float(frame.pts * frame.time_base) if frame.pts is not None else None
                        return video_frame_np, None, pts_seconds
                except StopIteration:
                    self.video_frames = None # Current video packet's frames are exhausted
                except Exception as e:
                    print(f"Error decoding video frame: {e}", file=sys.stderr)
                    self.video_frames = None # Clear iterator on error
                    continue # Try next packet

            # 2. If no video frames are available, try to get the next packet
            try:
                packet = next(self.packets)
                if packet.stream.type == 'video':
                    self.video_frames = iter(packet.decode())
                    # Immediately try to read from this new video packet
                    try:
                        frame = next(self.video_frames)
                        if frame is not None:
                            video_frame_np = frame.to_ndarray(format="bgr24")
                            pts_seconds = float(frame.pts * frame.time_base) if frame.pts is not None else None
                            return video_frame_np, None, pts_seconds
                    except StopIteration:
                        self.video_frames = None # This video packet yielded no frames
                        continue # Go to next packet
                    except Exception as e:
                        print(f"Error decoding video frame from new packet: {e}", file=sys.stderr)
                        self.video_frames = None # Clear iterator on error
                        continue # Go to next packet

                elif packet.stream.type == 'audio':
                    # The main loop needs the packet to write to output
                    return None, packet, None # Return the audio packet
                else: # Other stream types (subtitles, data, etc.)
                    continue # Get next packet

            except StopIteration:
                return None, None, None # All packets from self.packets are exhausted
            except Exception as e:
                print(f"Error decoding packet: {e}", file=sys.stderr)
                return None, None, None

    def close(self) -> None:
        if self.input_container:
            self.input_container.close()
        self.input_container = None
        self.audio_in = None
        self.video_in = None
        self.packets = None

    def read_packets(self):
        if not self.input_container:
            raise RuntimeError("Input container not opened. Call .open() first.")

        return self.input_container.demux(self.video_in, self.audio_in)


    def get_frame_size(self) -> Tuple[int, int]:
        if self.video_in:
            return self.video_in.width, self.video_in.height
        raise RuntimeError("Video stream not available.")

    def get_fps(self) -> float:
        if self.video_in:
            return float(self.video_in.average_rate)
        raise RuntimeError("Video stream not available.")

    def get_duration(self) -> float:
        if self.input_container and self.input_container.duration != av.time_base:
            return self.input_container.duration / av.time_base
        return 0.0 # Return 0 if duration is unknown or not set

    def get_audio_layout(self) -> Optional[av.AudioLayout]:
        if self.audio_in:
            return self.audio_in.layout
        return None

    def get_audio_rate(self) -> Optional[int]:
        if self.audio_in:
            return self.audio_in.sample_rate
        return None

    def get_video_stream(self) -> Optional[av.VideoStream]:
        return self.video_in

    def get_audio_stream(self) -> Optional[av.AudioStream]:
        return self.audio_in
