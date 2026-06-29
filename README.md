# Machine Learning Based Masonry Wall Sensitivity Web App

This Streamlit app provides a graphical interface for estimating flood-induced out-of-plane collapse fragility of masonry walls. It uses pretrained machine learning surrogate models to predict the lognormal fragility parameters `mu` and `beta`, then supports single case prediction and sensitivity heatmaps for capacity, dispersion, and collapse probability interpretation.

The app does not retrain models. It loads the bundled ML pretrained models from `models/`, checks user inputs against the constrained study domain, and runs inference only for valid wall configurations.

## Features

- Single wall prediction with dynamic input controls.
- Current configuration summary with scientific notation and symbol definitions.
- Sensitivity heatmaps for:
  - median collapse flood depth, `mu` (m);
  - fragility dispersion, `beta` (-);
  - normalized median collapse depth, `mu/Z` (-);
  - collapse probability at selected flood depth, `P(C | h)` (%);
  - collapse probability at selected relative flood depth, `P(C | h/Z)` (%).
- Representative wall case table and project visuals.
- Default wall length/span `L = 5.0 m`.

## Folder Structure

```text
strength-streamlit-app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
│   ├── logo_interreg_strength.jpg
│   ├── mechanics_based_collapse_modelling.png
│   ├── representative_wall_configuration.png
│   ├── representative_wall_cases_3d.png
│   ├── wall_mechanics_panels.png
│   └── wall_parameter_icons/
├── models/
│   ├── best_models_manifest.json
│   ├── feature_names.json
│   ├── final_model_mu.joblib
│   └── final_model_beta.joblib
└── src/
    ├── plotting.py
    ├── predict.py
    ├── preprocess.py
    ├── sensitivity.py
    └── utils.py
```

## Local Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

## Model Files

ML pretrained models are stored in `models/`:

- `models/final_model_mu.joblib`
- `models/final_model_beta.joblib`
- `models/best_models_manifest.json`
- `models/feature_names.json`

The `mu` model is a Support Vector Machine surrogate and the `beta` model is a Histogram Gradient Boosting surrogate. Both are serialized scikit-learn `Pipeline` objects with embedded preprocessing. The app uses these ML pretrained models directly and does not retrain them.

The bundled beta model uses the compact `HGB_phys` model, so the model files are small enough for a standard GitHub repository without Git LFS.

## Assets

Images and icons used by the app are stored in `assets/`. The app uses relative paths only, for example `assets/...`, `models/...`, and `src/...`.

Large EPS source/export files are not required by the Streamlit runtime and are ignored by `.gitignore`.

## Scientific Scope

The app evaluates wall configurations inside the constrained parameter range considered by the underlying study and training dataset. It checks representative wall cases, geometry, boundary condition, upper-floor loading, and other domain constraints before inference.

The app computes collapse probability using the lognormal fragility relation:

```text
P(C | h) = Phi((ln(h) - ln(mu)) / beta)
```

where `Phi` is the standard normal cumulative distribution function, `mu` is the predicted median collapse flood depth, and `beta` is the predicted logarithmic dispersion.

## Disclaimer

This web app is intended for research and demonstration purposes. Predictions are valid only within the parameter ranges considered in the underlying study and training dataset.

## Deployment Status

This repository has been prepared for GitHub upload, but it has not been deployed to Streamlit Community Cloud.
