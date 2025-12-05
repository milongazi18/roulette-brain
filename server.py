# --- filename: server.py ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gc # মেমোরি ক্লিন করার জন্য

app = Flask(__name__)
CORS(app)

# --- কনফিগারেশন ---
MODEL_FILE = 'roulette_lstm_model.keras'
SEQUENCE_LENGTH = 30
model = None # শুরুতে মডেল লোড করব না

@app.route('/', methods=['GET'])
def home():
    return "Roulette AI Server is Running... (Lazy Mode)"

@app.route('/predict-api', methods=['POST'])
def predict_api():
    global model
    
    # 🔥 LAZY LOADING: যখন রিকোয়েস্ট আসবে, তখনই শুধু মডেল লোড হবে
    if model is None:
        try:
            print("⏳ Loading Model for the first time...", flush=True)
            if os.path.exists(MODEL_FILE):
                model = load_model(MODEL_FILE)
                print("✅ Model Loaded Successfully!", flush=True)
            else:
                return jsonify({"error": "Model file missing"}), 500
        except Exception as e:
            return jsonify({"error": f"Model Load Error: {str(e)}"}), 500

    try:
        data = request.get_json(force=True)
        spins = data.get('spins', [])
        
        if len(spins) < SEQUENCE_LENGTH:
            return jsonify({"message": f"Need {SEQUENCE_LENGTH - len(spins)} more spins"}), 200

        last_30_spins = spins[-SEQUENCE_LENGTH:]
        input_seq = np.array(last_30_spins).astype(np.int32).reshape(1, SEQUENCE_LENGTH)
        
        probs = model.predict(input_seq, verbose=0)[0]
        prob_list = [float(p) for p in probs]
        
        # মেমোরি ক্লিন করা (ফ্রি সার্ভারের জন্য জরুরি)
        gc.collect()
        
        return jsonify({"probabilities": prob_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)