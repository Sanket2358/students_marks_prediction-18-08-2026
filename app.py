from flask import Flask, request, render_template_string, redirect, url_for, session
import pickle
import numpy as np
import os

app = Flask(__name__)
# Secret key is required to use Flask sessions for storing data between requests
app.secret_key = 'super_secret_ai_predictor_key' 

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
            --primary: #00F2FE;
            --secondary: #4FACFE;
            /* Darker, deeper background to make everything pop */
            --bg-gradient: linear-gradient(135deg, #020617, #0f172a, #020617);
            /* Slightly more transparent glass for dark mode */
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-highlight: #e0f2fe; /* Crisper white for labels */
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
            overflow-x: hidden;
        }

        /* Added Floating (Wavy) Animation to the Container */
        .container {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 40px;
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
            width: 100%;
            max-width: 480px;
            /* Two animations: First fades in, second floats continuously */
            animation: fadeInContainer 1s ease-out forwards, floatingBox 6s ease-in-out 1s infinite;
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
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
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
            color: var(--text-highlight);
            letter-spacing: 0.5px;
        }

        input, select {
            width: 100%;
            padding: 14px;
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.4);
            color: var(--primary); /* Typing color is neon cyan now */
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            outline: none;
            transition: all 0.3s ease;
            box-sizing: border-box;
        }

        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type="number"] {
            -moz-appearance: textfield; 
        }

        input::placeholder {
            color: rgba(255, 255, 255, 0.2);
            font-weight: 400;
        }

        input:focus, select:focus {
            background: rgba(0, 0, 0, 0.6);
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
            transform: translateY(-2px);
        }

        select {
            color: #ffffff;
        }

        select option {
            background-color: #0f172a; 
            color: #ffffff;
            font-size: 1rem;
            padding: 10px;
        }

        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            animation: slideUp 0.6s ease-out forwards;
            animation-delay: 0.6s;
            opacity: 0;
        }

        .btn-predict, .btn-reset {
            flex: 1;
            padding: 16px;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            text-decoration: none;
            display: flex;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
        }

        .btn-predict {
            background: linear-gradient(to right, var(--secondary), var(--primary));
            border: none;
            color: #020617; 
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
        }

        .btn-predict:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 30px rgba(0, 242, 254, 0.6);
            color: #ffffff;
        }
        
        .btn-predict:active, .btn-reset:active {
            transform: translateY(0) scale(0.98);
        }

        .btn-reset {
            background: rgba(0, 242, 254, 0.05);
            border: 2px solid var(--primary);
            color: var(--primary);
        }

        .btn-reset:hover {
            background: rgba(0, 242, 254, 0.15);
            transform: translateY(-3px) scale(1.02);
            color: #ffffff;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.2);
        }

        .result {
            margin-top: 30px;
            text-align: center;
            font-size: 2rem;
            font-weight: 800;
            background: rgba(0, 242, 254, 0.1);
            border: 1px solid var(--primary);
            padding: 20px;
            border-radius: 12px;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
            color: #fff;
            box-shadow: inset 0 0 20px rgba(0, 242, 254, 0.1), 0 0 20px rgba(0, 242, 254, 0.2);
            text-shadow: 0 0 10px rgba(0, 242, 254, 0.6);
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
        @keyframes fadeInContainer {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Wavy Floating Animation */
        @keyframes floatingBox {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-12px); }
            100% { transform: translateY(0px); }
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
            .button-group { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>AI Performance Predictor</h2>
        
        {% if error_msg %}
            <div class="error">{{ error_msg }}</div>
        {% endif %}

        <form method="POST" action="{{ url_for('home') }}">
            <div class="form-group">
                <label>Hours Studied</label>
                <input type="number" step="any" name="hours_studied" placeholder="e.g. 5.5" required autocomplete="off" value="{{ form_data.get('hours_studied', '') }}">
            </div>
            
            <div class="form-group">
                <label>Previous Scores</label>
                <input type="number" step="any" name="previous_scores" placeholder="e.g. 85" required autocomplete="off" value="{{ form_data.get('previous_scores', '') }}">
            </div>
            
            <div class="form-group">
                <label>Extracurricular Activities</label>
                <select name="extracurricular" required>
                    <option value="" disabled {% if not form_data.get('extracurricular') %}selected{% endif %}>Select an option</option>
                    <option value="1" {% if form_data.get('extracurricular') == '1' %}selected{% endif %}>Yes</option>
                    <option value="0" {% if form_data.get('extracurricular') == '0' %}selected{% endif %}>No</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Sleep Hours</label>
                <input type="number" step="any" name="sleep_hours" placeholder="e.g. 7.5" required autocomplete="off" value="{{ form_data.get('sleep_hours', '') }}">
            </div>
            
            <div class="form-group">
                <label>Sample Question Papers Practiced</label>
                <input type="number" step="any" name="papers_practiced" placeholder="e.g. 3" required autocomplete="off" value="{{ form_data.get('papers_practiced', '') }}">
            </div>
            
            <div class="button-group">
                <button type="submit" class="btn-predict">Predict Score</button>
                {% if prediction %}
                    <a href="{{ url_for('reset') }}" class="btn-reset">Predict More</a>
                {% endif %}
            </div>
        </form>

        {% if prediction %}
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
    error_msg = None
    
    if model is None:
        error_msg = "Error: linear_model.pkl not found in the root directory. Please make sure the file is present."

    if request.method == 'POST' and model is not None:
        try:
            # Save inputs in session so they don't disappear after submit
            session['form_data'] = request.form.to_dict()
            
            hours = float(request.form['hours_studied'])
            prev_scores = float(request.form['previous_scores'])
            extra = float(request.form['extracurricular'])
            sleep = float(request.form['sleep_hours'])
            papers = float(request.form['papers_practiced'])
            
            features = np.array([[hours, prev_scores, extra, sleep, papers]])
            pred_value = model.predict(features)[0]
            
            # Keep score between 0 and 100
            pred_value = max(0.0, min(100.0, pred_value))
            
            # Save the prediction result in session
            session['prediction'] = f"{pred_value:.2f}"
            
            # Redirect back to GET route
            return redirect(url_for('home'))
            
        except Exception as e:
            error_msg = f"An error occurred during prediction: {str(e)}"
            
    # GET Request: Retrieve data from session if it exists
    prediction = session.get('prediction')
    form_data = session.get('form_data', {})
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, form_data=form_data, error_msg=error_msg)

# Route to clear data for "Predict More"
@app.route('/reset')
def reset():
    session.pop('prediction', None)
    session.pop('form_data', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
