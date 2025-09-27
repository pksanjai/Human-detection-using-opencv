# Human Detection Project using OpenCV

A Python project for detecting humans in **images**, **videos**, or via **live camera feed** using **OpenCV** and **HOG (Histogram of Oriented Gradients) + SVM**.

---

## Project Overview

This project implements a real-time **human detection system** using computer vision. It can detect people in:

* Single images
* Video files
* Live webcam feed

The system highlights detected humans with bounding boxes, counts the number of people, and optionally outputs a processed video.

Key features include:

* **Bounding boxes** around detected humans.
* **Person count** displayed on the screen.
* **Flexible input sources**: image, video, or webcam.
* **Optional output** video saving.

---

## How It Works

1. **HOG + SVM Detector**

   * Uses OpenCV's `HOGDescriptor` with `getDefaultPeopleDetector()`.
   * HOG features capture the gradient patterns typical of human shapes.

2. **Detection Process**

   * `detectMultiScale()` scans the input frame at multiple scales.
   * Returns bounding boxes for each detected person.

3. **Visualization**

   * Draws **green rectangles** around detected humans.
   * Displays **person count** and **status** text on the frame.
   * Uses `cv2.imshow()` to display the processed frames.

4. **Video/Image Handling**

   * Reads image/video files or webcam feed.
   * Resizes frames for consistency.
   * Optionally writes output video if path is provided.

---

## Usage

### Command-line Arguments

| Argument           | Description                         |
| ------------------ | ----------------------------------- |
| `-i` or `--image`  | Path to image file.                 |
| `-v` or `--video`  | Path to video file.                 |
| `-c` or `--camera` | Enable live camera detection.       |
| `-o` or `--output` | Optional path to save output video. |

### Example Commands

1. **Detect in an image**

```bash
python human_detector.py --image path/to/image.jpg --output path/to/output.jpg
```

2. **Detect in a video**

```bash
python human_detector.py --video path/to/video.mp4 --output path/to/output.avi
```

3. **Detect from webcam**

```bash
python human_detector.py --camera
```

* Press **`q`** to exit the live detection window.

---

## How to Explain in a Technical Interview

1. **Problem Statement**

   * "This project solves the problem of detecting humans in images, videos, or live feeds in real-time."

2. **Technical Approach**

   * "I used OpenCV's HOGDescriptor with a pre-trained SVM detector for human detection. HOG features capture the edges and gradient patterns of human shapes."

3. **Workflow**

   * "The system accepts image, video, or camera input, resizes frames for consistency, applies HOG detection, draws bounding boxes around humans, displays the count, and optionally saves output video."

4. **Key Functions**

   * `detect(frame)`: detects humans and draws bounding boxes.
   * `detectByPathImage(path, output_path)`: detects humans in images.
   * `detectByPathVideo(path, writer)`: detects humans in video.
   * `detectByCamera(writer)`: detects humans via webcam.

5. **Challenges & Learning**

   * Optimizing detection speed for live camera feed.
   * Handling different input sources efficiently.
   * Working with OpenCV’s video processing pipeline.

6. **Future Enhancements**

   * Integrate deep learning models (YOLO, SSD) for higher accuracy.
   * Add multi-class detection (e.g., vehicles, animals).
   * Implement tracking to follow people across frames.

---

## Libraries Used

* `opencv-python` for computer vision tasks.
* `imutils` for image/video resizing and convenience functions.
* `numpy` for array handling.
* `argparse` for command-line argument parsing.

---

This project demonstrates **real-time computer vision**, **object detection**, and **Python programming skills**, which are highly relevant for technical interviews in AI/ML or software development roles.
