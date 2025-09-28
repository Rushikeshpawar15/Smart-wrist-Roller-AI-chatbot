# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

MODEL_FILE = "next_weight_model.joblib"

app = Flask(__name__)

# ========================
# Load model if available
# ========================
model = None
feature_columns = None
if os.path.exists(MODEL_FILE):
    try:
        artifact = joblib.load(MODEL_FILE)
        model = artifact.get('model')
        feature_columns = artifact.get('feature_columns')
        print("Loaded model:", MODEL_FILE)
    except Exception as e:
        print("Model load failed:", e)
else:
    print("No trained model found — using heuristic fallback.")

# ========================
# Exercise guides library
# ========================
EXERCISE_GUIDES = {
    "wrist curl": (
        "**Wrist Curl (Smart Wrist Roller):**\n"
        "- Sit on a bench, forearms resting on thighs, wrists just beyond knees.\n"
        "- Hold the roller with palms up.\n"
        "- Curl wrists upward, squeeze forearms, then lower **slowly**.\n"
        "- 3–4 sets of 12–20 reps, keep RPE ~7.\n"
        "- Focus on **slow eccentrics** for tendon health.\n"
        "- Adjust resistance by ±0.5–1kg depending on RPE."
    ),
    "hammer curl": (
        "**Hammer Curl (dumbbells/rope):**\n"
        "- Stand tall, elbows tucked, neutral grip (thumbs up).\n"
        "- Curl upward without swinging, wrists neutral.\n"
        "- Lower slowly under control.\n"
        "- 3–4 sets of 10–15 reps.\n"
        "- Trains brachioradialis + grip — complements wrist roller work."
    ),
    "bench press": (
        "**Bench Press:**\n"
        "- Lie flat, feet planted, bar over chest.\n"
        "- Lower bar to mid-chest with elbows ~75°.\n"
        "- Press up until arms extend (don’t lock hard).\n"
        "- Keep shoulders retracted, avoid bouncing.\n"
        "- 3–5 sets of 6–12 reps."
    )
}

EXERCISE_SUGGESTIONS = {
    'wrist_curl': [
        'Reverse curls',
        'Hammer curls',
        'Farmer’s carries',
        'Grip squeezes',
        'Rice-bucket training'
    ],
    'bench_press': ['Incline dumbbell press', 'Push-ups', 'Dumbbell flyes'],
    'squat': ['Front squat', 'Goblet squat', 'Walking lunges'],
    'deadlift': ['Romanian deadlift', 'Hip thrusts', 'Kettlebell swings'],
    'overhead_press': ['Dumbbell shoulder press', 'Lateral raises', 'Face pulls'],
    'barbell_row': ['One-arm dumbbell row', 'Seated cable row', 'Lat pulldown']
}

# ========================
# Prediction helpers
# ========================
def build_feature_row(user_id, exercise, last_session):
    numeric_fields = ['session_index','sets','reps','weight','rpe','rest_sec','volume',
                      'prev_weight','prev_volume','prev_rpe']
    row = {fld: last_session.get(fld, 0) for fld in numeric_fields}
    if not row['volume']:
        row['volume'] = row['weight'] * row['reps'] * row['sets']
    if feature_columns:
        full = {c: 0 for c in feature_columns}
        for k, v in row.items():
            if k in full: full[k] = v
        ex_col = f"ex_{exercise}"
        if ex_col in full: full[ex_col] = 1
        if 'user_id' in full: full['user_id'] = user_id
        return pd.DataFrame([full], columns=feature_columns)
    return pd.DataFrame([row])

def model_predict_next_weight(user_id, exercise, last_session):
    if model and feature_columns:
        try:
            X = build_feature_row(user_id, exercise, last_session)
            pred = model.predict(X)[0]
            return round(pred * 2) / 2.0
        except:
            pass

    # Heuristic fallback
    w = float(last_session.get('weight', 0))
    rpe = float(last_session.get('rpe', 7))
    reps = int(last_session.get('reps', 8))
    delta = 0.0
    if rpe <= 7 and reps >= 8:
        delta = 0.5
    elif rpe >= 9:
        delta = -0.5

    pred = w + delta

    # cap change for wrist curl (±2kg max)
    if exercise == "wrist_curl":
        if pred - w > 2: pred = w + 2
        if w - pred > 2: pred = w - 2

    return max(0.5, round(pred * 2) / 2.0)

# ========================
# Analyze session
# ========================
def analyze_session(user_id, exercise, last_session):
    try:
        lw = float(last_session.get('weight', 0))
        sets = int(last_session.get('sets', 3))
        reps = int(last_session.get('reps', 8))
        rpe = float(last_session.get('rpe', 7))
        vol = lw * reps * sets
    except Exception as e:
        return {"error": str(e)}

    last_session['volume'] = vol
    pred = model_predict_next_weight(user_id, exercise, last_session)
    delta = round(pred - lw, 2)

    if delta > 0.4: decision = "increase"
    elif delta < -0.4: decision = "decrease"
    else: decision = "maintain"

    msg = f"Predicted next working resistance for *{exercise}*: **{pred} kg-equivalent**.\n"
    if decision == "increase":
        msg += f"Recommendation: Increase by ~{abs(delta)} kg.\n"
    elif decision == "decrease":
        msg += f"Recommendation: Decrease by ~{abs(delta)} kg.\n"
    else:
        msg += "Recommendation: Maintain current resistance.\n"

    if exercise == "wrist_curl":
        msg += "Focus on slow eccentrics and steady tempo. MR damper adapts smoothly, so increase only in small steps."

    return {
        "predicted_next_weight": pred,
        "decision": decision,
        "delta": delta,
        "message": msg,
        "alternatives": EXERCISE_SUGGESTIONS.get(exercise, [])
    }

# ========================
# Chat logic
# ========================
def craft_response(text, context=None):
    t = text.lower()

    # Technique guides
    for key, guide in EXERCISE_GUIDES.items():
        if key in t:
            return guide

    # Device / damper
    if "mr damper" in t or "magnetorheological" in t:
        return ("**MR Damper:** Viscosity changes with magnetic field → controllable resistance in real time. "
                "Safe, adaptive, and ideal for rehab + training.")

    if "rehab" in t or "therapy" in t:
        return ("**Rehab Tips:** Start light, keep RPE ≤7. Focus on controlled motion, 2–3 sessions/week. "
                "Increase resistance slowly as recovery progresses.")

    if "recommend" in t or "next weight" in t:
        if context and "exercise" in context and "last_session" in context:
            res = analyze_session(context.get("user_id",1), context["exercise"], context["last_session"])
            return res.get("message","Could not analyze.")
        return "Please provide exercise + last session details (sets, reps, weight, RPE)."

    return "Ask me about **wrist curls, hammer curls, bench press, or MR damper rehab guidance**."

# ========================
# Flask routes
# ========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze_session", methods=["POST"])
def analyze_route():
    payload = request.json or {}
    return jsonify(analyze_session(payload.get("user_id",1),
                                   payload.get("exercise","wrist_curl"),
                                   payload.get("last_session",{})))

@app.route("/chat", methods=["POST"])
def chat_route():
    payload = request.json or {}
    return jsonify({"reply": craft_response(payload.get("text",""),
                                            payload.get("context"))})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
