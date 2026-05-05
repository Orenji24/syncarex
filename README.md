# Privacy-Preserving Synthetic Healthcare Data Generator

A Flask web app that generates synthetic healthcare CSV data, applies clinical consistency rules, and shows a real-vs-synthetic evaluation dashboard.

## Features

- CSV upload and synthetic CSV download
- CTGAN-based tabular synthetic data generation using SDV
- Interactive real vs synthetic profile graphs
- Dataset summary and data quality metrics
- KL divergence, KS test score, duplicate rate, and correlation similarity
- Correlation heatmaps with real, synthetic, and difference views
- Clinical checks for blood pressure, blood sugar/glucose, heart rate, and condition labels
- Research report page
- Chat-style assistant for explaining metrics and privacy

## Backend Software

- Python: main backend language
- Flask: web server and routing
- Pandas: CSV and DataFrame processing
- NumPy: numerical operations and controlled value changes
- SDV / CTGAN: synthetic tabular data generation
- Scikit-learn: utility model testing
- SciPy: statistical metrics such as KL divergence and KS score
- Plotly: interactive frontend charts
- Gunicorn: production WSGI server for deployment

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Optional Debug Mode

Windows PowerShell:

```powershell
$env:FLASK_DEBUG="1"
python app.py
```

## Deployment Notes

This is a Flask/Python project, so it cannot run directly on GitHub Pages. GitHub Pages only supports static websites.

Use one of these for live hosting:

- Render
- Railway
- PythonAnywhere
- Hugging Face Spaces
- AWS / Azure / GCP

For Render or Railway, the included `Procfile` can start the app:

```text
web: gunicorn app:app
```

The app reads the `PORT` environment variable automatically, which is required by many hosting platforms.

## Important Privacy Note

The current epsilon value is simulated for dashboard explanation. For a formal Differential Privacy guarantee, the project should be extended with a privacy-accounted training method such as DP-SGD or another validated DP mechanism.

## Generated Files

Uploaded CSV files and generated synthetic outputs are ignored by Git:

- `temp_uploads/`
- `static/outputs/`

This keeps private or generated data out of the GitHub repository.
