import sys
import os
import av
import numpy as np
import cv2
import argparse

from reader import VideoReader
from detector import NsfwDetector
from censor import CensorProcessor
from output import VideoOutputWriter
from config_loader import ConfigLoader
from postprocessor import BoxClusterer
from preprocess_detector import ImageHashDetector

# --- Configuration ---
DEFAULT_INPUT_PATH = "your_video.mp4"
DEFAULT_OUTPUT_PATH = "output.mp4"
DEFAULT_OUTPUT_FORMAT = "mp4"

class MainProcessor:
    def __init__(self, reader: VideoReader, detector: NsfwDetector | ImageHashDetector, censor_processor: CensorProcessor,
                 writer: VideoOutputWriter, config_loader: ConfigLoader, box_clusterer: BoxClusterer = None):
        self.reader = reader
        self.detector = detector
        self.censor_processor = censor_processor
        self.writer = writer
        self.config_loader = config_loader
        self.box_clusterer = box_clusterer

    def run(self):
        if not self.reader.open():
            sys.exit(1)
        if not self.writer.open():
            self.reader.close()
            sys.exit(1)

        try:
            frame_count = 0
            for packet in self.reader.read_packets():
                if packet.stream == self.reader.audio_in:
                    self.writer.write_audio_packet(packet)
                elif packet.stream == self.reader.video_in:
                    for frame in packet.decode():
                        frame_count += 1
                        img = frame.to_ndarray(format='rgb24')

                        # Detect NSFW content
                        detections = self.detector.detect(img)

                        # Apply clustering if enabled
                        if self.box_clusterer:
                            detections = self.box_clusterer.cluster_boxes(detections)

                        # Apply censorship
                        processed_img = self.censor_processor.apply_censorship(img, detections)

                        # Write processed video frame
                        processed_av_frame = av.VideoFrame.from_ndarray(processed_img, format='rgb24')
                        self.writer.write_video_frame(processed_av_frame)

            self.writer.flush_encoders()
            print(f"Video processing complete. Total frames processed: {frame_count}")

        except Exception as e:
            print(f"An error occurred during processing: {e}", file=sys.stderr)
            import traceback # Keep this import for the MainProcessor.run() method's exception handling
            traceback.print_exc(file=sys.stderr)
        finally:
            self.writer.close()
            self.reader.close()
            print("All containers closed.")

def main():
    parser = argparse.ArgumentParser(description="NudeNix: Real-time NSFW content detection and censoring for videos.")

    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', '-o', type=str, help='Path to the output video file (required if --output-type is "file").')
    parser.add_argument('--output-type', type=str, choices=['file', 'mpv', 'pyvidplayer2'], default='file',
                        help='Destination for the processed video stream.')
    parser.add_argument('--format', '-f', type=str, choices=['mp4', 'webm'], default='mp4',
                        help='Output video format (relevant if --output-type is "file").')
    parser.add_argument('--config', '-c', type=str, help='Path to a custom censor_settings.yaml file.')
    parser.add_argument('--disable-image-hash', action='store_true', help='Disable image hashing optimization, overriding config.')
    parser.add_argument('--enable-image-hash-debug', action='store_true', help='Enable image hash debug overlay, overriding config.')
    parser.add_argument('--nudenet-model-path', type=str, help='Override Nudenet model path from config.')
    parser.add_argument('--nudenet-resolution', type=int, help='Override Nudenet inference resolution from config.')

    args = parser.parse_args()

    # Validate arguments based on output-type
    if args.output_type == 'file' and not args.output:
        parser.error("--output is required when --output-type is 'file'.")
    if args.output_type != 'file' and args.output:
        print("Warning: --output argument is ignored when --output-type is not 'file'.", file=sys.stderr)
    if args.output_type != 'file' and args.format:
        print("Warning: --format argument is ignored when --output-type is not 'file'.", file=sys.stderr)

    if not os.path.exists(args.input):
        print(f"Error: Input video file not found at {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing video: {args.input}")
    if args.output_type == 'file':
        print(f"Output to: {args.output} (Format: {args.format})")
    else:
        print(f"Output to: {args.output_type} backend.")

    # --- Initialize Components ---
    config_loader = ConfigLoader(config_path=args.config)

    # Get Nudenet specific settings from config and apply CLI overrides
    nudenet_config = config_loader.get_nudenet_settings()
    model_path = args.nudenet_model_path if args.nudenet_model_path else nudenet_config.get('model_path')
    inference_resolution = args.nudenet_resolution if args.nudenet_resolution else nudenet_config.get('inference_resolution', 320)

    # Get Clustering specific settings from config
    clustering_config = config_loader.get_clustering_settings()
    box_clusterer = None
    if clustering_config.get('enabled', False):
        max_distance = clustering_config.get('max_distance', 50.0)
        box_clusterer = BoxClusterer(max_distance=max_distance)

    reader = VideoReader(args.input)
    # Initialize NsfwDetector
    nsfw_detector_instance = NsfwDetector(model_path=model_path, inference_resolution=inference_resolution)

    # Get image hash settings from config and apply CLI overrides
    image_hash_config = config_loader.get_image_hash_settings()
    if args.disable_image_hash:
        image_hash_config['enabled'] = False
    if args.enable_image_hash_debug:
        image_hash_config['debug_overlay'] = True

    detector_to_pass = ImageHashDetector(nsfw_detector_instance, image_hash_config)

    censor_processor = CensorProcessor(config_loader)

    if not reader.open():
        sys.exit(1)

    # Adjust writer based on output type
    if args.output_type == 'file':
        writer = VideoOutputWriter(
            args.output,
            reader, # Pass the reader object
            output_format=args.format
        )
    elif args.output_type == 'mpv':
        # TODO: Implement MPV writer logic
        print("Error: MPV output not yet implemented.", file=sys.stderr)
        sys.exit(1)
    elif args.output_type == 'pyvidplayer2':
        # TODO: Implement PyVidPlayer2 writer logic
        print("Error: PyVidPlayer2 output not yet implemented.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: Unknown output type {args.output_type}", file=sys.stderr)
        sys.exit(1)

    processor = MainProcessor(reader, detector_to_pass, censor_processor, writer, config_loader, box_clusterer)
    processor.run()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # sys.exit() was called, which is expected for some error conditions
        pass
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback # Ensure traceback is imported for this block
        traceback.print_exc(file=sys.stderr)
