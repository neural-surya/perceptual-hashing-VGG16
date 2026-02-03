# Perceptual Hashing & VGG16 Visual Validation

This project demonstrates a semantic visual validation approach using VGG16 feature extraction and perceptual hashing. It is designed to be robust against minor UI changes (like banner additions) while detecting significant structural regressions.

## Overview

The testing strategy involves two main techniques:
1.  **VGG16 Feature Extraction**: Uses a pre-trained VGG16 model to extract high-level features from UI screenshots. This allows for semantic comparison where minor visual shifts (e.g., a new banner) result in high similarity scores, avoiding false positives.
2.  **Perceptual Hashing**: Calculates the Hamming distance between the baseline and current images. This is useful for detecting structural similarities even when there are changes like filters or color shifts. A low Hamming distance (< 5) indicates structural similarity.

## Prerequisites

*   Python 3.x
*   Node.js (for Playwright)

## Installation

1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install Playwright browsers:
    ```bash
    playwright install
    ```

## Usage

To run the visual regression tests:

```bash
pytest test_login_visual.py
```

## Project Structure

*   `test_login_visual.py`: Contains the test logic using Pytest, Playwright, and VGG16.
*   `templates/`: Stores the baseline and current screenshots.
    *   `baseline_login.png`: The reference image.
    *   `current_login.png`: The captured image during the test run.
*   `requirements.txt`: List of dependencies.

## How it Works

1.  **Capture**: The test navigates to the target URL and captures a screenshot of the current UI.
2.  **Feature Extraction**: Both the baseline and current images are processed through the VGG16 model to obtain feature vectors.
3.  **Comparison**:
    *   **Cosine Similarity**: Calculates the similarity between the feature vectors. A score > 0.90 typically indicates a pass.
    *   **Perceptual Hashing**: (Mentioned in description) Checks Hamming distance for structural integrity.

## Notes

*   Ensure `templates/baseline_login.png` exists before running the tests for the first time, or update the test to generate it if missing.
