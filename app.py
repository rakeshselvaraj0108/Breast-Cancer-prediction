# app.py
import os
import pickle
import io
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
MODEL_DIR = 'model_files'
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load model artifacts if present; otherwise tell user to run training script
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, 'cancer_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    feat_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(feat_path):
        return None, None, None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feat_path, 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, EXPECTED_FEATURES = load_artifacts()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_col_name(s: str):
    # Lowercase, strip, remove punctuation, common synonyms
    if not isinstance(s, str):
        return str(s)
    s = s.strip().lower()
    replacements = {
        ' ': '_',
        '-': '_',
        '/': '_',
        'mean_': 'mean_',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # remove parentheses and percent signs
    for ch in '()%':
        s = s.replace(ch, '')
    return s

def normalize_columns(cols):
    return [normalize_col_name(c) for c in cols]

def best_column_match(upload_cols, expected_cols):
    """Try to match uploaded columns to expected feature names.
    Returns dict: expected_col -> uploaded_col (or None if no match)"""
    u_norm = {normalize_col_name(c): c for c in upload_cols}
    matches = {}
    for e in expected_cols:
        e_norm = normalize_col_name(e)
        # exact match
        if e_norm in u_norm:
            matches[e] = u_norm[e_norm]
            continue
        # try partial match heuristics (prefix/suffix)
        tokens = [t for t in e_norm.split('_') if len(t) > 2]
        found = None
        for un, original in u_norm.items():
            if all(tok in un for tok in tokens[:2]):
                found = original
                break
        matches[e] = found
    return matches

@app.route('/')
def index():
    global model, scaler, EXPECTED_FEATURES
    model, scaler, EXPECTED_FEATURES = load_artifacts()
    return render_template('index.html', model_ready = (model is not None))

@app.route('/upload', methods=['POST'])
def upload():
    global model, scaler, EXPECTED_FEATURES
    if 'file' not in request.files:
        flash("No file part")
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        # read
        try:
            df = pd.read_csv(path)
        except Exception as e:
            flash(f"Could not read CSV: {e}")
            return redirect(url_for('index'))
        temp_name = filename  # saved under uploads
        if EXPECTED_FEATURES is None:
            flash("Model not trained yet. Run `python model/train_model.py` to create model files, then reload this page.")
            return redirect(url_for('index'))
        uploaded_cols = list(df.columns)
        matches = best_column_match(uploaded_cols, EXPECTED_FEATURES)
        matched_count = sum(1 for v in matches.values() if v is not None)
        if matched_count / len(EXPECTED_FEATURES) >= 0.7:
            feature_df = pd.DataFrame()
            for e, u in matches.items():
                if u is not None:
                    feature_df[e] = df[u]
                else:
                    feature_df[e] = np.nan
            valid_rows_before = len(feature_df)
            feature_df = feature_df.dropna()
            if len(feature_df) == 0:
                flash("After matching, no rows remain (missing values). Consider opening the mapping page to map columns manually and handle missing values.")
                return redirect(url_for('mapping', filename=temp_name))
            X_scaled = scaler.transform(feature_df.values)
            preds = model.predict(X_scaled)
            label_map = {0: 'Malignant', 1: 'Benign'}
            result = feature_df.copy()
            result['Prediction'] = [label_map.get(int(p), str(p)) for p in preds]
            out_csv = os.path.join(app.config['UPLOAD_FOLDER'], f"pred_{filename}")
            result.to_csv(out_csv, index=False)
            return render_template('result.html', tables=[result.head(200).to_html(classes='data', index=False)], filename=os.path.basename(out_csv))
        else:
            return redirect(url_for('mapping', filename=temp_name))
    else:
        flash("Invalid file type. Please upload a .csv")
        return redirect(url_for('index'))

@app.route('/mapping')
def mapping():
    filename = request.args.get('filename')
    if not filename:
        flash("No file specified for mapping.")
        return redirect(url_for('index'))
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        flash("Uploaded file not found.")
        return redirect(url_for('index'))
    df = pd.read_csv(path)
    _, _, EXPECTED_FEATURES = load_artifacts()
    uploaded_cols = list(df.columns)
    return render_template('mapping.html', uploaded_cols=uploaded_cols, expected_features=EXPECTED_FEATURES, filename=filename)

@app.route('/do_map', methods=['POST'])
def do_map():
    filename = request.form.get('filename')
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        flash("Uploaded file not found for mapping.")
        return redirect(url_for('index'))
    df = pd.read_csv(path)
    _, scaler, EXPECTED_FEATURES = load_artifacts()
    mapping = {}
    for feat in EXPECTED_FEATURES:
        mapped = request.form.get(feat)
        if mapped and mapped != 'NONE':
            mapping[feat] = mapped
        else:
            mapping[feat] = None
    feature_df = pd.DataFrame()
    for e, u in mapping.items():
        if u is not None and u in df.columns:
            feature_df[e] = df[u]
        else:
            feature_df[e] = np.nan
    feature_df = feature_df.dropna()
    if len(feature_df) == 0:
        flash("After mapping and dropping missing values, no rows remain. Please map more columns or fix the uploaded file.")
        return redirect(url_for('mapping', filename=filename))
    X_scaled = scaler.transform(feature_df.values)
    model_path = os.path.join(MODEL_DIR, 'cancer_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(X_scaled)
    label_map = {0: 'Malignant', 1: 'Benign'}
    result = feature_df.copy()
    result['Prediction'] = [label_map.get(int(p), str(p)) for p in preds]
    out_csv = os.path.join(app.config['UPLOAD_FOLDER'], f"pred_{filename}")
    result.to_csv(out_csv, index=False)
    return render_template('result.html', tables=[result.head(200).to_html(classes='data', index=False)], filename=os.path.basename(out_csv))

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        flash("File not found")
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
