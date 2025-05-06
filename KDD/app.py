from flask import Flask, request, jsonify, render_template
from FPG_CNN import PM25Predictor, PM25Dataset

app = Flask(__name__)

# Load model configuration and initialize predictor
config = {
    'aux_features': ['temperature', 'humidity', 'wind_speed', 'pressure'],
    'window_size': 7,
    'history_window': 7,
    'batch_size': 64,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'lr_patience': 5,
    'lr_factor': 0.5,
    'min_support': 0.01,
    'similarity_threshold': 0.3
}

predictor = PM25Predictor(config)
predictor.load_model('model.pth', 'fpgrowth.pkl', 'label_encoder_classes.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate input
        if not all(key in data for key in ['history', 'features', 'days']):
            return jsonify({'error': 'Missing required fields'}), 400

        if len(data['history']) != 7 or len(data['features']) != 7:
            return jsonify({'error': 'History and features must contain exactly 7 days of data'}), 400

        # Make prediction
        prediction = predictor.predict_next_days(
            data['history'],
            data['features'],
            data['days']
        )

        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)