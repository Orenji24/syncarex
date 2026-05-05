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
        epochs = int(request.form.get("epochs", 100))
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
    result = RUN_RESULTS.get(payload.get("run_id", ""))
    quality = result.get("quality_metrics", {}) if result else {}
    summary = result.get("summary", {}) if result else {}
    privacy_level = result.get("privacy_level", "N/A") if result else "N/A"
    epsilon = result.get("epsilon", "N/A") if result else "N/A"
    medical_checks = result.get("medical_checks", []) if result else []

    def metric(name, fallback="N/A"):
        return quality.get(name, fallback)

    def medical_summary():
        if not medical_checks:
            return "No medical consistency checks were available for this dataset."
        passed = sum(1 for check in medical_checks if check.get("status") in {"Pass", "Checked"})
        return f"{passed}/{len(medical_checks)} medical consistency checks passed or were successfully checked."

    if "reliable" in question or "trust" in question:
        answer = (
            "Yes, this synthetic dataset is designed for research use, especially for exploratory analysis, "
            "prototype model training, teaching, and privacy-conscious data sharing. For this run, the dashboard reports "
            f"correlation similarity = {metric('correlation_similarity')}, KL divergence = {metric('kl_divergence')}, "
            f"KS score = {metric('ks_score')}, duplicate rate = {metric('duplicate_rate')}, and {medical_summary()} "
            "That means you should judge it as research-ready when those values are acceptable for your study objective. "
            "For clinical deployment or publication-grade claims, include these metrics in the methodology section."
        )
    elif "research" in question or "study" in question:
        answer = (
            "Use this synthetic data for research workflows where patient-level privacy matters: feasibility studies, "
            "model prototyping, statistical comparison, classroom demos, and sharing non-identifiable examples. "
            f"This run contains {summary.get('rows_uploaded', 'N/A')} uploaded rows, "
            f"{summary.get('columns_detected', 'N/A')} columns, and a simulated epsilon of {epsilon} "
            f"with privacy level {privacy_level}."
        )
    elif "age" in question and ("clip" in question or "clipped" in question or "bounded" in question):
        answer = (
            "Age is bounded row-by-row, so each synthetic age stays within +/- 5 of the matching original row. "
            "This keeps the synthetic record medically plausible while avoiding exact copying. The same idea is used for BP and cholesterol with their requested margins."
        )
    elif "bp" in question or "blood pressure" in question or "cholesterol" in question or "cholestrol" in question:
        answer = (
            "The generator applies row-level medical bounds after synthesis: age within +/- 5, blood pressure within +/- 10, "
            "and cholesterol within +/- 20 of the corresponding original row. Check the Clinical Margin Audit to confirm the maximum observed difference."
        )
    elif "epsilon" in question or "privacy" in question:
        answer = (
            f"This run uses epsilon = {epsilon}, giving a privacy level of {privacy_level}. "
            "Lower epsilon means stronger privacy and usually less similarity to the original data. "
            "In the current prototype epsilon is simulated from the control slider; for a formal DP guarantee, the next upgrade is DP-SGD or another privacy-accounted training method."
        )
    elif "kl" in question or "ks" in question:
        answer = (
            f"For this run, KL divergence is {metric('kl_divergence')} and the KS score is {metric('ks_score')}. "
            "KL divergence measures distribution difference, while KS checks the maximum gap between real and synthetic cumulative distributions. Lower values indicate closer statistical behavior."
        )
    elif "duplicate" in question or "copy" in question or "same" in question:
        answer = (
            f"The synthetic duplicate rate is {metric('duplicate_rate')}. A low duplicate rate is important because it means the generator is less likely to repeat records. "
            "The privacy risk card also checks exact row matches between real and synthetic data."
        )
    elif "correlation" in question or "relationship" in question:
        answer = (
            f"The correlation similarity score for this run is {metric('correlation_similarity')}. "
            "A higher score means relationships between numeric columns were better preserved. Use the heatmap toggle to inspect which relationships changed."
        )
    elif "medical" in question or "glucose" in question or "disease" in question or "diagnosis" in question:
        details = " ".join(f"{check.get('name')}: {check.get('status')} - {check.get('detail')}" for check in medical_checks)
        answer = details or "No medical consistency checks were triggered because the expected medical columns were not detected."
    else:
        answer = (
            "I can help interpret this run for research use. Ask about reliability, epsilon/privacy, KL or KS scores, duplicate rate, correlations, medical checks, or why age/BP/cholesterol are bounded."
        )

    return jsonify({"answer": answer})


@app.route("/download/<path:filename>")
def download(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(app.config["OUTPUT_FOLDER"], safe_name)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
