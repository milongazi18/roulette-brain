# --- filename: server.py ---
import os

# টেনসরফ্লোর লগ কমিয়ে মেমোরি বাঁচানো
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gc 

app = Flask(__name__)
CORS(app)

# --- কনফিগারেশন ---
MODEL_FILE = 'roulette_lstm_model.keras'
SEQUENCE_LENGTH = 30
model = None 

@app.route('/', methods=['GET'])
def home():
    return "Roulette AI Server is Running... (Optimized Mode)"

@app.route('/predict-api', methods=['POST'])
def predict_api():
    global model
    
    # মেমোরি চেক এবং গার্বেজ কালেকশন
    gc.collect()

    # 🔥 LAZY LOADING: যখন রিকোয়েস্ট আসবে, তখনই শুধু মডেল লোড হবে
    if model is None:
        try:
            print("⏳ Loading Model...", flush=True)
            if os.path.exists(MODEL_FILE):
                model = load_model(MODEL_FILE)
                print("✅ Model Loaded!", flush=True)
            else:
                return jsonify({"error": "Model file missing"}), 500
        except Exception as e:
            print(f"❌ Error loading model: {e}", flush=True)
            return jsonify({"error": "Server Memory Full. Try again in 10s."}), 503

    try:
        data = request.get_json(force=True)
        spins = data.get('spins', [])
        
        if len(spins) < SEQUENCE_LENGTH:
            return jsonify({"message": f"Need {SEQUENCE_LENGTH - len(spins)} more spins"}), 200

        last_30_spins = spins[-SEQUENCE_LENGTH:]
        input_seq = np.array(last_30_spins).astype(np.int32).reshape(1, SEQUENCE_LENGTH)
        
        # প্রেডিকশন
        probs = model.predict(input_seq, verbose=0)[0]
        prob_list = [float(p) for p in probs]
        
        # কাজ শেষে মেমোরি পরিষ্কার করার চেষ্টা
        # del input_seq
        # gc.collect()
        
        return jsonify({"probabilities": prob_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)