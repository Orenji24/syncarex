import os
import uuid

from flask import Flask, render_template, request, send_file, abort, jsonify
from werkzeug.utils import secure_filename

from dp_model import run_dp_pipeline


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
RUN_RESULTS = {}

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/guide")
def guide():
    return render_template("guide.html")


@app.route("/generate", methods=["POST"])
def generate():
    file = request.files.get("file")
    if not file or not file.filename:
        abort(400, "Please upload a CSV file.")

    original_name = secure_filename(file.filename)
    if not original_name.lower().endswith(".csv"):
        abort(400, "Only CSV files are supported right now.")

    run_id = uuid.uuid4().hex
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{run_id}_{original_name}")
    file.save(input_path)

    try:
        epochs = int(request.form.get("epochs", 300))
        sample_size_raw = request.form.get("sample_size", "").strip()
        sample_size = int(sample_size_raw) if sample_size_raw else None
        noise_level = float(request.form.get("noise_level", 0.5))
        epsilon = float(request.form.get("epsilon", 1.0))
    except ValueError:
        abort(400, "Invalid control value.")

    epochs = max(100, min(1000, epochs))
    sample_size = max(1, sample_size) if sample_size else None
    noise_level = max(0.0, min(1.0, noise_level))
    epsilon = max(0.1, min(10.0, epsilon))

    result = run_dp_pipeline(
        input_path=input_path,
        output_folder=app.config["OUTPUT_FOLDER"],
        run_id=run_id,
        epochs=epochs,
        sample_size=sample_size,
        noise_level=noise_level,
        epsilon=epsilon,
    )
    RUN_RESULTS[run_id] = result

    return render_template("result.html", **result)


@app.route("/research-report/<run_id>")
def research_report(run_id):
    result = RUN_RESULTS.get(run_id)
    if not result:
        abort(404)
    return render_template("research_report.html", **result)


@app.route("/assistant", methods=["POST"])
def assistant():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").lower()

    if "reliable" in question or "trust" in question:
        answer = "Reliability depends on the dashboard: high correlation similarity, low KL/KS scores, low duplicate rate, and passing medical checks are good signs. It is still synthetic and should be validated before research use."
    elif "age" in question and ("clip" in question or "clipped" in question or "bounded" in question):
        answer = "Age is bounded row-by-row so every synthetic age stays within the requested +/- 5 range from its matching original row. That keeps values close without copying the exact original."
    elif "epsilon" in question or "privacy" in question:
        answer = "Lower epsilon means stronger privacy but usually less similarity to the real dataset. In this prototype epsilon is simulated for the dashboard; formal DP needs DP-SGD or another privacy-accounted training method."
    elif "kl" in question or "ks" in question:
        answer = "KL divergence and KS score compare real and synthetic distributions. Lower values usually mean the synthetic distribution is closer to the original."
    else:
        answer = "Check the summary, quality metrics, privacy panel, clinical margin audit, and medical consistency checks together. No single metric is enough on its own."

    return jsonify({"answer": answer})


@app.route("/download/<path:filename>")
def download(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(app.config["OUTPUT_FOLDER"], safe_name)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
