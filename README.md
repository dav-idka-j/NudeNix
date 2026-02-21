# NudeNix: NSFW Content Censoring

## Project Description

NudeNix is a powerful, modular utility for NSFW content detection and censoring. It can process existing video files or capture your screen live, applying censorship effects and outputting the result to a new file or a transparent, click-through system overlay.

## Technologies Used

*   [NudeNet](https://github.com/notAI-tech/NudeNet) for NSFW detection
*   [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the transparent, click-through screen overlay
*   [PyAV](https://github.com/PyAV-Org/PyAV) for media handling and encoding
*   [OpenCV](https://opencv.org/) for image processing
*   [Pillow](https://python-pillow.org/) for screen capture
*   [NumPy](https://numpy.org/) for numerical operations
*   [ImageHash](https://github.com/JohannesBuchner/imagehash) for perceptual hashing

## Features

*   **Real-Time Screen Censoring:** Capture any monitor and apply a transparent overlay that hides NSFW content in real-time.
*   **Modular Architecture:** Easily switch between input sources (files, screen) and output destinations (files, overlay).
*   **Highly Configurable:** Control detection sensitivity, censor types, and per-label settings via `censor_settings.yaml`.
*   **Extensible:** Add new censor effects by simply dropping a Python file into the `censors/` directory.

### Prerequisites

**Python 3.8+:** The project is developed using Python.  Download from [python.org](https://www.python.org/downloads/).

## Installation

To use NudeNix, you can either install it directly from the source or use a pre-packaged executable.

#### Option 1: Install from Source

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/NudeNix.git
    cd NudeNix
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

#### Option 2: Using a Pre-packaged Executable

Simply download and extract the `.zip` file. No Python installation or dependency management is required on your part.

### Usage

NudeNix uses a URI-based system for inputs and outputs.

**General Command:**
```bash
python main.py -i <INPUT_URI> -o <OUTPUT_URI> [OPTIONS]
```

### URI Schemes

*   **Input (`-i`):**
    *   `file://path/to/video.mp4`: Process an existing video.
    *   `screen://1`: Capture monitor 1.
*   **Output (`-o`):**
    *   `file://path/to/output.mp4`: Save the censored result to a file.
    *   `overlay://`: Display censorship as a transparent system overlay.

### Examples

```bash
# Live screen censoring with overlay
python main.py -i screen://1 -o overlay://

# Censoring a video file and saving to MP4
python main.py -i file://input.mp4 -o file://censored.mp4

# Recording a censored screen capture
python main.py -i screen://1 -o file://recording.mp4
```

#### Censoring Configuration

Censoring behavior is controlled by the `censor_settings.yaml` file. This YAML file defines global settings for all detection labels and allows for specific overrides per label. See `censor_settings.example.yaml`, is provided. **You should copy `censor_settings.example.yaml` to `censor_settings.yaml` and modify it.**  Customize `censor_settings.yaml` to define your desired censoring logic and place the file
 - next to your `main.py` script (if running from source)
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
