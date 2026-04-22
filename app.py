from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import run_pipeline
import sys

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    text = data.get('text', '')
    
    # Suppress print output for clean API response
    old_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')  # Silence prints
    results = run_pipeline(text)
    sys.stdout = old_stdout
    
    return jsonify({"results": results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Disable reloader to prevent crashes during model loading
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)