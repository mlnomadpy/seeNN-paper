# Image Classification Demo

This demo showcases a Flask-based API for image classification using a pre-trained Keras model. The API supports multiple input modalities, including RGB, depth, and normal images, as well as additional computed modalities like edge detection and entropy.

## Prerequisites

- Python 3.7+
- Flask
- TensorFlow 2.x
- Pillow (PIL)
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-classification-demo.git
   cd image-classification-demo
   ```

2. Install the required dependencies:
   ```
   pip install flask tensorflow pillow numpy
   ```

3. Place your pre-trained Keras model file (e.g., `saved_model.keras`) in the project directory.

## File Structure

```
image-classification-demo/
├── app.py
├── index.html
├── saved_model.keras
└── utils/
    ├── edge_layer.py
    └── self_entropy.py
```

## Usage

1. Update the model path in `app.py` to point to your saved Keras model:
   ```python
   model = load_model('path/to/your/saved_model.keras', custom_objects={
       'EdgeDetectionLayerV2': EdgeDetectionLayerV2,
       'SelfEntropyLayerV2': SelfEntropyLayerV2
   })
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:5000/index.html`

4. Use the web interface to:
   - Upload an RGB image (required)
   - Optionally upload depth and normal images
   - Select additional modalities (edge and entropy)
   - Submit the form to get a prediction

## API Endpoint

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `rgb`: RGB image file (required)
  - `depth`: Depth image file (optional)
  - `normal`: Normal image file (optional)
  - `modalities`: Comma-separated list of additional modalities (optional, can be "edge" and/or "entropy")

## Response

The API returns a JSON object with the following structure:

```json
{
  "predicted_class": 0,
  "confidence": 0.9876543
}
```

- `predicted_class`: The index of the predicted class
- `confidence`: The confidence score for the prediction (0-1)

## Customization

- To add or modify modalities, update the `predict()` function in `app.py`
- To change the model architecture or input processing, modify the relevant sections in `app.py`
- To update the user interface, edit `index.html`
