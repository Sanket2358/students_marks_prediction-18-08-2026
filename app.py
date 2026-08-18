from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained linear model
MODEL_PATH = 'linear_model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# HTML and CSS template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4FACFE;
            --secondary: #00F2FE;
            --bg-gradient: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-gradient);
            color: #ffffff;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }

        .container {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            padding: 40px;
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 25px 45px rgba(0, 0, 0, 0.4);
            width: 100%;
            max-width: 480px;
            animation: fadeIn 1s ease-out forwards;
        }

        h2 {
            text-align: center;
            margin-top: 0;
            margin-bottom: 30px;
            font-weight: 800;
            font-size: 1.8rem;
            background: -webkit-linear-gradient(45deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .form-group {
            margin-bottom: 20px;
            animation: slideUp 0.6s ease-out forwards;
            opacity: 0;
        }

        .form-group:nth-child(1) { animation-delay: 0.1s; }
        .form-group:nth-child(2) { animation-delay: 0.2s; }
        .form-group:nth-child(3) { animation-delay: 0.3s; }
        .form-group:nth-child(4) { animation-delay: 0.4s; }
        .form-group:nth-child(5) { animation-delay: 0.5s; }

        label {
            display: block;
            margin-bottom: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            color: #d1d5db;
        }

        input, select {
            width: 100%;
            padding: 14px;
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.2);
            color: #ffffff;
            font-family: inherit;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
            box-sizing: border-box;
        }

        /* 1. FIX: Hide up/down arrows (spinners) on number inputs */
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type="number"] {
            -moz-appearance: textfield; /* For Firefox */
        }

        input::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }

        input:focus, select:focus {
            background: rgba(0, 0, 0, 0.4);
            border-color: var(--primary);
            box-shadow: 0 0 15px rgba(79, 172, 254, 0.3);
            transform: translateY(-2px);
        }

        /* 2. FIX: Dropdown styling & colors */
        select option {
            background-color: #1a2a33; /* Clean dark background */
            color: #ffffff;
            font-size: 1rem;
            padding: 10px;
        }

        button {
            width: 100%;
            padding: 16px;
            margin-top: 10px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            border: none;
            border-radius: 12px;
            color: #0f2027; /* Dark text for better contrast on cyan button */
            font-size: 1.1rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
            animation: slideUp 0.6s ease-out forwards;
            animation-delay: 0.6s;
            opacity: 0;
        }

        button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6);
            color: #ffffff;
        }
        
        button:active {
            transform: translateY(0) scale(0.98);
        }

        .result {
            margin-top: 30px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 800;
            background: rgba(79, 172, 254, 0.1);
            border: 1px solid var(--primary);
            padding: 20px;
            border-radius: 12px;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
            color: #fff;
            box-shadow: inset 0 0 15px rgba(79, 172, 254, 0.1);
            text-shadow: 0 0 10px rgba(79, 172, 254, 0.5);
        }

        .error {
            color: #ff6b6b;
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            background: rgba(255, 107, 107, 0.1);
            border-radius: 8px;
            border: 1px solid #ff6b6b;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popIn {
            0% { opacity: 0; transform: scale(0.8); }
            100% { opacity: 1; transform: scale(1); }
        }

        @media (max-width: 480px) {
            .container { padding: 30px 20px; }
            h2 { font-size: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>AI Performance Predictor</h2>
        
        {% if error_msg %}
            <div class="error">{{ error_msg }}</div>
        {% endif %}

        <form method="POST">
            <div class="form-group">
                <label>Hours Studied</label>
                <input type="number" step="any" name="hours_studied" placeholder="e.g. 5.5" required autocomplete="off">
            </div>
            
            <div class="form-group">
                <label>Previous Scores</label>
                <input type="number" step="any" name="previous_scores" placeholder="e.g. 85" required autocomplete="off">
            </div>
            
            <div class="form-group">
                <label>Extracurricular Activities</label>
                <select name="extracurricular" required>
                    <option value="" disabled selected>Select an option</option>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Sleep Hours</label>
                <input type="number" step="any" name="sleep_hours" placeholder="e.g. 7.5" required autocomplete="off">
            </div>
            
            <div class="form-group">
                <label>Sample Question Papers Practiced</label>
                <input type="number" step="any" name="papers_practiced" placeholder="e.g. 3" required autocomplete="off">
            </div>
            
            <button type="submit">Predict Score</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            🎯 {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_msg = None
    
    if model is None:
        error_msg = "Error: linear_model.pkl not found in the root directory. Please make sure the file is present."

    if request.method == 'POST' and model is not None:
        try:
            # Extract inputs from the form
            hours = float(request.form['hours_studied'])
            prev_scores = float(request.form['previous_scores'])
            extra = float(request.form['extracurricular'])
            sleep = float(request.form['sleep_hours'])
            papers = float(request.form['papers_practiced'])
            
            # Format inputs as a 2D numpy array
            features = np.array([[hours, prev_scores, extra, sleep, papers]])
            
            # Make the prediction
            pred_value = model.predict(features)[0]
            
            # 3. FIX: Handle Negative values and values above 100
            # A score shouldn't realistically be below 0 or above 100.
            pred_value = max(0.0, min(100.0, pred_value))
            
            # Format prediction to 2 decimal places
            prediction = f"{pred_value:.2f}"
            
        except Exception as e:
            error_msg = f"An error occurred during prediction: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error_msg=error_msg)

if __name__ == '__main__':
    # Run the app locally
    app.run(debug=True, host='0.0.0.0', port=5000)
