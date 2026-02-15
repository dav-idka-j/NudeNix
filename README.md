# NudeNix: Video Censoring

## Project Description

A utility for real-time NSFW content detection and censoring for videos. Features a **modular architecture** that separates the core video processing logic from the output. This allows the processed video stream to be directed to various backends (e.g., a file, standard output for piping, or directly to media players like MPV or PyVidPlayer2) based on user needs.

## Technologies Used

*   [NudeNet](https://github.com/notAI-tech/NudeNet?tab=readme-ov-file) for detection
*   [PyAV](https://github.com/PyAV-Org/PyAV) for media handling
*   [numpy](https://github.com/numpy/numpy) for numerical operations
*   [OpenCV](https://opencv.org/) for image processing and censor effects
*   [PyYAML](https://pyyaml.org/) for configuration file parsing
*   [ImageHash](https://github.com/JohannesBuchner/imagehash) for perceptual image hashing
*   [Pillow (PIL Fork)](https://python-pillow.org/) for image manipulation (used by ImageHash)

## Features

*   **Modular Processing Pipeline:** Clear separation of input reading, NSFW detection, censoring logic, and output handling.
*   **Real-time NSFW Detection:** Utilizes the `nudenet` library to analyze video frames.
*   **Highly Configurable Censoring:** Settings defined in a `censor_settings.yaml` file allow granular control over:
    *   `min_score` threshold for detections.
    *   Enabling/disabling censoring per detection label.
    *   Definition of censor stacks (e.g., `box`, blur, pixelate).
    *   Per-censor settings.
    *   Global defaults and per-label overrides.
*   **Flexible Output Backends:**
    *   **File Output:** Write processed video to a new file (e.g., MP4, WebM).
    *   **MPV Streaming:** Stream processed video directly to the MPV player via `stdin` for real-time playback.
    *   **PyVidPlayer2 Backend:** Stream to a `PyVidPlayer2` instance for embedded display.
*   **Extensible Censor Types:** New censoring effects can be easily added by dropping new modules into the `censors/` directory.
*   **Configurable NudeNet Model:** Use different NudeNet models and inference resolutions for varying performance and accuracy.

## For Users

### Prerequisites

Before setting up and running NudeNix, ensure you have the following installed:

1.  **MPV Media Player:** The core video player (required for MPV streaming backend).
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install mpv`
    *   **macOS:** `brew install mpv`
    *   **Windows:** Download from the [MPV website](https://mpv.io/installation/).
2.  **Python 3.8+:** The project is developed using Python.
    *   Download from [python.org](https://www.python.org/downloads/).

### Installation

To use NudeNix, you can either install it directly from the source or use a pre-packaged executable.

#### Option 1: Install from Source (Recommended for users familiar with Python)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/NudeNix.git
    cd NudeNix
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install av opencv-python numpy nudenet pyyaml imagehash pillow
    ```

#### Option 2: Using a Pre-packaged Executable

Simply download and extract the `.zip` file. No Python installation or dependency management is required on your part.

### Usage

The main processing script is `main_processor.py`. When using the source code, run it directly. If using a packaged executable, run the `NudeNix` executable.

**General Command Structure:**

```bash
python main_processor.py -i <input_video_path> [OPTIONS]
# or for packaged executable:
./NudeNix -i <input_video_path> [OPTIONS]
```

**Key Arguments:**

*   `-i`, `--input <FILE_PATH>`: Path to the input video file (required).
*   `-o`, `--output <FILE_PATH>`: Path to the output video file (required if `--output-type file`).
*   `--output-type {file, mpv, pyvidplayer2}`: Destination for the processed video stream (`file` is default).
*   `-f`, `--format {mp4, webm}`: Output video format (relevant if `--output-type file`; `mp4` is default).
*   `-c`, `--config <FILE_PATH>`: Path to a custom `censor_settings.yaml` file.
*   `--disable-image-hash`: Disable the image hashing optimization, overriding config.
*   `--enable-image-hash-debug`: Enable image hash debug overlay (green/red/blue circles), overriding config.
*   `--nudenet-model-path <FILE_PATH>`: Override Nudenet model path from config.
*   `--nudenet-resolution <INT>`: Override Nudenet inference resolution from config.

**Examples:**

```bash
# Process video and write to an MP4 file (default output-type and format)
python main_processor.py -i test.mkv -o output.mp4

# Process video and save as a WebM file
python main_processor.py --input test.mkv --output output.webm --format webm

# Stream processed video directly to MPV (MPV backend not yet implemented)
# This will eventually launch MPV and pipe the video to it.
# Currently, this will result in an error indicating not implemented.
python main_processor.py -i /path/to/your/input_video.mp4 --output-type mpv

# Use a custom configuration file
python main_processor.py -i video.mp4 -o out.mp4 -c my_custom_settings.yaml

# Disable image hashing optimization via CLI
python main_processor.py -i video.mp4 -o out.mp4 --disable-image-hash

# Enable image hash debug overlay (shows circles on frames)
python main_processor.py -i video.mp4 -o out.mp4 --enable-image-hash-debug

# Override Nudenet model and resolution
python main_processor.py -i video.mp4 -o out.mp4 --nudenet-model-path custom_model.onnx --nudenet-resolution 640
```

#### Censoring Configuration

Censoring behavior is controlled by the `censor_settings.yaml` file. This YAML file defines global settings for all detection labels and allows for specific overrides per label. See `censor_settings.example.yaml`, is provided. **You should copy `censor_settings.example.yaml` to `censor_settings.yaml` and modify it.**  Customize `censor_settings.yaml` to define your desired censoring logic and place the file
 - next to your `main_processor.py` script (if running from source)
 - or next to the `NudeNix` executable (if running a packaged version).

#### Nudenet Model Configuration

You can configure the `nudenet` model's behavior via the top-level `nudenet_settings` block in `censor_settings.yaml`.
*   `model_path`: Specifies a custom ONNX model file to load. If left `null`, Nudenet will use its default model.
*   `inference_resolution`: Sets the resolution at which the model performs inference. This should match the resolution the model was trained on (e.g., 320 for `320n.onnx`, 640 for `640m.onnx`).

#### Clustering Configuration

The `clustering_settings` block in `censor_settings.yaml` controls the merging of closely located detection boxes.
*   `enabled`: Set to `true` to enable the box clustering feature.
*   `max_distance`: Defines how close two boxes of the same label must be (in pixels) to be considered for merging into a single larger box.

#### Image Hashing Optimization

This feature optimizes performance by avoiding redundant NSFW detection runs on consecutive frames that are visually similar. It uses image hashing to determine frame similarity.

*   `enabled` (boolean, default: `true`): Set to `false` to disable this optimization.
*   `algorithm` (string, default: `'average_hash'`): Specifies the image hashing algorithm to use. Options include:
    *   `'average_hash'` (fastest, good for slight changes)
    *   `'phash'` (perceptual hash, more robust to resizing/cropping)
    *   `'dhash'` (difference hash, good for small shifts)
    *   `'whash'` (wavelet hash)
*   `hash_threshold` (integer, default: `11`): The maximum "Hamming distance" between two image hashes for them to be considered "similar." A lower value means stricter similarity.
*   `max_skip_frames` (integer, default: `5`): The maximum number of consecutive frames for which detection can be skipped, even if hashes remain similar. This prevents stale detections if content subtly changes over a long period. After this many skipped frames, detection will run regardless of hash similarity.
*   `debug_overlay` (boolean, default: `false`): If `true`, a small circle will be drawn on the top-left corner of the video frame for debugging:
    *   **Green Circle:** Detection was skipped due to hash similarity.
    *   **Red Circle:** Detection was run (either first frame or hash difference).
    *   **Blue Circle:** Detection was forced to run (`max_skip_frames` reached).

#### Available Censor Types

This section details the censor types currently implemented and available for use in `censor_settings.yaml`.

##### 1. `box` Censor

*   **Type:** `box`
*   **Description:** Draws a rectangular box around the detected region. Can be configured to draw a border or fill the entire region.
*   **Settings:**
    *   `thickness` (integer, default: `2`): The thickness of the border in pixels. Set to `-1` to fill the entire box.
    *   `size_percent` (integer, default: `100`): The size of the box as a percentage of the detected bounding box (e.g., `90` for 90% of the original size).
    *   `color` (object, default: `{r: 0, g: 0, b: 255}` - Red): The color of the box in RGB format. Example: `{r: 0, g: 255, b: 0}` for Green.
    *   `display_label` (boolean, default: `true`): Whether to display the detection label as text above the box.
    *   `display_score` (boolean, default: `true`): Whether to display the detection score as text above the box.

##### 2. `pixelation` Censor

*   **Type:** `pixelation`
*   **Description:** Applies a pixelation (mosaic) effect to the detected region.
*   **Settings:**
    *   `strength` (integer, default: `10`): The block size in pixels for pixelation. A higher value means larger pixels and stronger effect.
    *   `type` (string, default: `'mosaic'`): The type of pixelation effect. Currently only `'mosaic'` is supported.
    *   `scale_with_detection_size` (boolean, default: `true`): If `true`, the `strength` is interpreted as a percentage of the detected region's smallest dimension, making the pixelation adaptive.
    *   `grayscale` (boolean, default: `false`): If `true`, the pixelated region will also be converted to grayscale.
    *   `sampling_mode` (string, default: `'accurate'`): How colors are sampled for the pixel blocks.
        *   `'accurate'`: Uses `cv2.INTER_AREA` for downscaling, which provides a good average color for the block.
        *   `'fast'`: Uses `cv2.INTER_NEAREST` for downscaling, which samples the center pixel's color, but is less accurate.
    *   `shape` (string, default: `'square'`): The shape of the pixelation blocks. Currently only `'square'` is supported.

### Supported Nudenet Labels

The `nudenet` model provides the following detection labels that can be configured for censoring:

*   `FEMALE_GENITALIA_COVERED`
*   `FACE_FEMALE`
*   `BUTTOCKS_EXPOSED`
*   `FEMALE_BREAST_EXPOSED`
*   `FEMALE_GENITALIA_EXPOSED`
*   `MALE_BREAST_EXPOSED`
*   `ANUS_EXPOSED`
*   `FEET_EXPOSED`
*   `BELLY_COVERED`
*   `FEET_COVERED`
*   `ARMPITS_COVERED`
*   `ARMPITS_EXPOSED`
*   `FACE_MALE`
*   `BELLY_EXPOSED`
*   `MALE_GENITALIA_EXPOSED`
*   `ANUS_COVERED`
*   `FEMALE_BREAST_COVERED`
*   `BUTTOCKS_COVERED`

## For Developers

### Installation

Follow the [User Installation](#installation-user---option-1-install-from-source-recommended-for-users-familiar-with-python) steps to set up the development environment.

### Extending Censoring Effects

The modular design of the censoring system allows for easy addition of new censoring effects. To add a new censor type:

1.  **Create a New File:** In the `censors/` directory, create a new Python file (e.g., `censors/my_custom_censor.py`).
2.  **Define a Censor Class:** Inside this file, define a class named according to the pattern `<CensorType>Censor` (e.g., `MyCustomCensor`). This class must have:
    *   An `__init__(self, settings)` method that takes a dictionary of settings specific to this censor type.
    *   An `apply(self, frame_ndarray, detection)` method that takes the NumPy array of the video frame (to be modified in-place) and a single detection dictionary.
3.  **Update `censor_settings.yaml`:** In your `censor_settings.yaml` file, add your new censor type under the `censors` list for `all_labels` or specific labels. The `type` key in the YAML should match the lowercase, underscore-separated version of your class name (e.g., `MyCustomCensor` -> `my_custom`).

### Packaging with PyInstaller

To distribute NudeNix as a standalone executable, you can use PyInstaller.

#### 1. Initial Build

Ensure `pyinstaller` is installed (`pip install pyinstaller`). Then, navigate to your project root and run a build:

```bash
pyinstaller NudeNix.spec
```

The `dist/NudeNix` directory will now contain your standalone application.

### Testing

The project uses `pytest` for unit testing. Test files are located in the `tests/` directory.

#### Running Tests

1.  **Install pytest:**
    Activate your virtual environment (e.g., `source venv/bin/activate`), then:
    ```bash
    pip install pytest
    ```
2.  **Run tests:**
    Activate your virtual environment (e.g., `source venv/bin/activate`), then execute:
    ```bash
    PYTHONPATH=. pytest
    ```
    This command will discover and run all tests in the `tests/` directory.
