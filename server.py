# --- filename: server.py ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

# --- কনফিগারেশন ---
MODEL_FILE = 'roulette_lstm_model.keras'
SEQUENCE_LENGTH = 30
VOCAB_SIZE = 37

# --- মডেল লোড ---
# সার্ভার চালু হওয়ার সময় মডেল মেমোরিতে লোড হবে
try:
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        print("✅ Server Model Loaded")
    else:
        print("❌ Model file not found on server!")
        model = None
except Exception as e:
    print(f"Model Load Error: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return "Roulette AI Server is Running..."

@app.route('/predict-api', methods=['POST'])
def predict_api():
    if not model:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        data = request.get_json(force=True)
        spins = data.get('spins', [])
        
        # অন্তত ৩০টি স্পিন দরকার
        if len(spins) < SEQUENCE_LENGTH:
            return jsonify({"message": f"Need {SEQUENCE_LENGTH - len(spins)} more spins"}), 200

        # শেষ ৩০টি স্পিন নিয়ে প্রেডিকশন
        last_30_spins = spins[-SEQUENCE_LENGTH:]
        input_seq = np.array(last_30_spins).astype(np.int32).reshape(1, SEQUENCE_LENGTH)
        
        # টেনসরফ্লো প্রেডিকশন
        probs = model.predict(input_seq, verbose=0)[0]
        
        # রেজাল্ট প্রসেসিং
        prob_list = [float(p) for p in probs] # JSON এ পাঠানোর জন্য লিস্টে কনভার্ট
        
        return jsonify({"probabilities": prob_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # ক্লাউডে পোর্ট অটোমেটিক নেয়, তাই os.environ ব্যবহার করা হয়েছে
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)