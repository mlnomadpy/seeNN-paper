from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from src.utils.edge_layer import EdgeDetectionLayerV2
from src.utils.self_entropy import SelfEntropyLayerV2

app = Flask(__name__)

# Load the model
model = load_model('saved_model.keras', custom_objects={
    'EdgeDetectionLayerV2': EdgeDetectionLayerV2,
    'SelfEntropyLayerV2': SelfEntropyLayerV2
})

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'rgb' not in request.files:
        return jsonify({'error': 'No RGB image provided'}), 400

    rgb_image = Image.open(io.BytesIO(request.files['rgb'].read()))
    rgb_input = preprocess_image(rgb_image)

    inputs = [rgb_input]

    if 'depth' in request.files:
        depth_image = Image.open(io.BytesIO(request.files['depth'].read()))
        depth_input = preprocess_image(depth_image)
        inputs.append(depth_input)

    if 'normal' in request.files:
        normal_image = Image.open(io.BytesIO(request.files['normal'].read()))
        normal_input = preprocess_image(normal_image)
        inputs.append(normal_input)

    # Process additional modalities if requested
    modalities = request.form.get('modalities', '').split(',')
    
    if 'edge' in modalities:
        edge_layer = EdgeDetectionLayerV2()
        edge_input = edge_layer(rgb_input)
        inputs.append(edge_input)

    if 'entropy' in modalities:
        entropy_layer = SelfEntropyLayerV2()
        entropy_input = entropy_layer(rgb_input)
        inputs.append(entropy_input)

    # Make prediction
    prediction = model.predict(inputs)
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])

    return jsonify({
        'predicted_class': int(predicted_class),
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)