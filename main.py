import sys
import os
import av
import numpy as np
import cv2
import argparse
import logging
import time
from fractions import Fraction
from urllib.parse import urlparse

from input.video_file_reader import VideoFileReader
from input.screen_reader import ScreenCaptureReader
from detector import NsfwDetector
from censor import CensorProcessor
from video_file_writer import VideoFileWriter
from output.screen_writer import ScreenWriter
from output.mpv_writer import MpvWriter
from config_loader import ConfigLoader
from postprocessor import BoxClusterer
from preprocess_detector import ImageHashDetector
from censor_drawing_contexts.in_place_drawing_context import InPlaceDrawingContext
from censor_drawing_contexts.overlay_drawing_context import OverlayDrawingContext
from censors_registry import CensorsRegistry
from statistics import (
    PerformanceStatistics,
    PipelineSummaryModule,
    ReaderReportModule,
    OutputReportModule
)

logger = logging.getLogger(__name__)

class MainProcessor:
    def __init__(self, reader: VideoFileReader | ScreenCaptureReader, detector: NsfwDetector | ImageHashDetector, censor_processor: CensorProcessor,
                 writer: VideoFileWriter | ScreenWriter, config_loader: ConfigLoader, box_clusterer: BoxClusterer = None,
                 stats: PerformanceStatistics = None):
        self.reader = reader
        self.detector = detector
        self.censor_processor = censor_processor
        self.writer = writer
        self.config_loader = config_loader
        self.box_clusterer = box_clusterer
        self.stats = stats or PerformanceStatistics(enabled=False)

        if self.stats.enabled:
            self.stats.register_module(PipelineSummaryModule(), priority=0)
            self.stats.register_module(ReaderReportModule(), priority=10)
            self.stats.register_module(OutputReportModule(), priority=50)

    def run(self):
        if not self.reader.open():
            sys.exit(1)
        if not self.writer.open():
            self.reader.close()
            sys.exit(1)

        try:
            self.stats.start()
            frame_count = 0

            with self.stats.timer('reader'):
                video_frame_np, audio_packet, pts = self.reader.read_frame()

            while video_frame_np is not None or audio_packet is not None:
                if audio_packet is not None:
                    if isinstance(self.writer, VideoFileWriter):
                        with self.stats.timer('output'):
                            self.writer.write_audio_packet(audio_packet)

                if video_frame_np is not None:
                    self.stats.increment_counter('frame_count')
                    frame_count += 1

                    # Detect NSFW content
                    detections = self.detector.detect(video_frame_np, stats=self.stats)

                    if self.box_clusterer:
                        detections = self.box_clusterer.cluster_boxes(detections)

                    if detections:
                        self.stats.increment_counter('frames_with_censorship')

                    # Apply censorship
                    with self.stats.timer('censors_total'):
                        processed_img = self.censor_processor.apply_censorship(video_frame_np, detections, stats=self.stats)

                    # Convert back to RGB for writing/display
                    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    processed_av_frame = av.VideoFrame.from_ndarray(processed_img_rgb, format='rgb24')

                    # Set PTS based on reader (original file PTS or calculated screen PTS)
                    if pts is not None:
                        # Use high-precision microseconds for the frame's timebase
                        processed_av_frame.time_base = Fraction(1, 1000000)
                        processed_av_frame.pts = int(pts * 1000000)
                    else:
                        # Fallback to frame index based on writer's FPS
                        fps = self.writer.get_fps()
                        tb = Fraction(1, int(fps)) if fps > 0 else Fraction(1, 30)
                        processed_av_frame.time_base = tb
                        processed_av_frame.pts = frame_count - 1

                    with self.stats.timer('output'):
                        self.writer.write_video_frame(processed_av_frame)

                with self.stats.timer('reader'):
                    video_frame_np, audio_packet, pts = self.reader.read_frame()

            self.writer.flush_encoders()
            self.stats.stop()
            print(f"Processing complete. Total frames processed: {frame_count}")
            self.stats.print_report()

        except Exception as e:
            logger.error(f"An error occurred during processing: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self.writer.close()
            self.reader.close()
            print("All containers closed.")

def main(args):
    input_uri = urlparse(args.input)
    output_uri = urlparse(args.output)

    output_path_final = None
    output_path_to_writer = None
    temp_output_path = None
    is_file_output = output_uri.scheme == 'file'

    if is_file_output:
        output_path_final = output_uri.path
        output_path_to_writer = output_path_final
        # Check if input and output paths are the same
        if input_uri.scheme == 'file' and input_uri.path == output_uri.path:
            temp_output_path = output_uri.path + ".part"
            output_path_to_writer = temp_output_path

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    config_loader = ConfigLoader()
    stats = PerformanceStatistics(enabled=args.performance_statistics)

    nudenet_config = config_loader.get_nudenet_settings()
    model_path = nudenet_config.get('model_path')
    inference_resolution = nudenet_config.get('inference_resolution', 320)

    clustering_config = config_loader.get_clustering_settings()
    box_clusterer = None
    if clustering_config.get('enabled', False):
        max_distance = clustering_config.get('max_distance', 50.0)
        box_clusterer = BoxClusterer(max_distance=max_distance)

    nsfw_detector_instance = NsfwDetector(model_path=model_path, inference_resolution=inference_resolution, stats=stats)
    image_hash_config = config_loader.get_image_hash_settings()
    detector_to_pass = ImageHashDetector(nsfw_detector_instance, image_hash_config, stats=stats)

    censors_registry = CensorsRegistry()

    reader = None
    if input_uri.scheme == 'file':
        reader = VideoFileReader(input_uri.path)
    elif input_uri.scheme == 'screen':
        monitor = int(input_uri.netloc or 1)
        reader = ScreenCaptureReader(monitor=monitor)

    writer = None
    drawing_context = None

    if output_uri.scheme == 'file':
        drawing_context = InPlaceDrawingContext(censors_registry)
        writer = VideoFileWriter(
            output_path_to_writer,
            reader,
            output_format=args.format
        )
    elif output_uri.scheme == 'overlay':
        drawing_context = OverlayDrawingContext(censors_registry)
        writer = ScreenWriter(
            reader=reader,
            overlay_context=drawing_context
        )
    elif output_uri.scheme == 'mpv':
        drawing_context = InPlaceDrawingContext(censors_registry)
        writer = MpvWriter(reader_instance=reader)
    elif output_uri.scheme == 'pyvidplayer2':
        print(f"Error: Output scheme '{output_uri.scheme}' not yet implemented.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: Unknown output scheme '{output_uri.scheme}'.", file=sys.stderr)
        sys.exit(1)

    censor_processor = CensorProcessor(config_loader, drawing_context, stats=stats)

    try:
        processor = MainProcessor(reader, detector_to_pass, censor_processor, writer, config_loader, box_clusterer, stats=stats)
        processor.run()

        # If processing is successful and a temporary file was used, rename it
        if temp_output_path and os.path.exists(temp_output_path):
            if os.path.exists(output_path_final):
                os.remove(output_path_final)
            os.rename(temp_output_path, output_path_final)
            print(f"Successfully saved to '{output_path_final}'.")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        # Clean up temporary file if an error occurred
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"Cleaned up temporary file: '{temp_output_path}'.", file=sys.stderr)
        sys.exit(1)
    finally:
        # reader and writer are closed within processor.run()
        pass

def _resolve_file_uri_path(parser, uri_parse_result, is_input=False):
    """
    Resolves a file:// URI path, handling relative paths and converting to an absolute file:/// URI.
    Performs existence checks if it's an input path.
    """
    file_path_component = ""
    if uri_parse_result.netloc and not uri_parse_result.path:
        # Handles 'file://filename.mp4' where filename.mp4 is in netloc
        file_path_component = uri_parse_result.netloc
    elif uri_parse_result.path:
        # Handles 'file:///absolute/path' or 'file://./relative/path'
        file_path_component = uri_parse_result.path
    else: # e.g., "file://"
        parser.error(f"File path cannot be empty for 'file://' scheme.")

    # Resolve to an absolute path for os.path.exists and consistency
    resolved_path = os.path.abspath(file_path_component)

    if is_input:
        if not os.path.exists(resolved_path):
            parser.error(f"Input file not found at '{resolved_path}'")

    # Return the absolute file:/// URI string
    return f"file:///{resolved_path.lstrip('/')}"


def parse_args(args):
    parser = argparse.ArgumentParser(description="NudeNix: Real-time NSFW content detection and censoring.")

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input source URI, e.g., "file:///path/to/video.mp4" or "screen://1" for monitor 1.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output destination URI, e.g., "file:///path/to/output.mp4" or "overlay://".')

    parser.add_argument('--force-overwrite', '-y', action='store_true',
                        help='Automatically confirm overwriting if output path is same as input.')
    parser.add_argument('--performance-statistics', action='store_true',
                        help='Collect and report performance statistics.')

    parsed_args = parser.parse_args(args)

    input_uri = urlparse(parsed_args.input)
    output_uri = urlparse(parsed_args.output)

    if input_uri.scheme == 'file':
        parsed_args.input = _resolve_file_uri_path(parser, input_uri, is_input=True)
        input_uri = urlparse(parsed_args.input)
    elif input_uri.scheme == 'screen':
        if input_uri.netloc and not input_uri.netloc.isdigit():
            parser.error(f"Invalid monitor number for screen input: '{input_uri.netloc}'")
    else:
        parser.error(f"Unsupported input scheme: '{input_uri.scheme}'. Supported schemes are 'file', 'screen'.")

    if output_uri.scheme == 'file':
        parsed_args.output = _resolve_file_uri_path(parser, output_uri, is_input=False)
        output_uri = urlparse(parsed_args.output)

        if input_uri.scheme == 'file' and output_uri.scheme == 'file' and input_uri.path == output_uri.path:
            if not parsed_args.force_overwrite:
                confirmation = input(
                    f"Output path '{output_uri.path}' is the same as the input path. Overwrite? (y/N): ").lower()
                if confirmation != 'y':
                    print("Operation cancelled by user.", file=sys.stderr)
                    sys.exit(0)

        # Infer format from the resolved output path
        _, ext = os.path.splitext(output_uri.path)
        # TODO: Add more robust format detection or mapping
        parsed_args.format = ext[1:].lower() if ext else 'mp4'
    elif output_uri.scheme in ['overlay', 'mpv', 'pyvidplayer2']:
        parsed_args.format = None # No format for stream outputs
    else:
        parser.error(f"Unsupported output scheme: '{output_uri.scheme}'. Supported schemes are 'file', 'overlay', 'mpv', 'pyvidplayer2'.")

    return parsed_args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        args = parse_args(sys.argv[1:])
        main(args)
    except SystemExit:
        # sys.exit() was called, which is expected for some error conditions
        pass
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
        sys.exit(1)
