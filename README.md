# Smart Health Risk Prediction — ML Demonstration

An educational machine-learning demo that illustrates a simple binary classification workflow using age and symptom-like input features.

> ⚠️ **Not a medical tool.** This project uses a synthetic dataset and labels generated from a simple rule. It is not clinically validated, must not be used for diagnosis or treatment decisions, and cannot assess a person's actual health.

## Project overview

The notebook-style script creates sample data, defines a rule-derived target, trains a Logistic Regression classifier, and reports a basic evaluation result. The included interface/output image and project documents describe the prototype.

## Features

- Generate a reproducible synthetic dataset
- Use age, fever, tiredness, and cough as example features
- Create a target label using a transparent heuristic
- Split data into training and testing subsets
- Train a Logistic Regression baseline
- Display basic model evaluation and example predictions

## Workflow

```text
Synthetic data → Rule-based target → Train/test split
     → Logistic Regression → Evaluation → Demo output
```

## Tech stack

Python · Pandas · NumPy · scikit-learn

## Run locally

```bash
git clone https://github.com/Madhum4107/amd-health-prediction-system.git
cd amd-health-prediction-system
python -m venv .venv
```

Activate:

**Windows**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install and execute:

```bash
python -m pip install pandas numpy scikit-learn
python "project code.py"
```

## Repository contents

- `project code.py` — synthetic data generation and ML demonstration
- `ai_health_prediction_output.png` — example output
- `AI-Powered-Smart-Health-Prediction-System.pdf` — project document
- `AI_Health_Prediction_White_Paper.pdf` — accompanying write-up

## Limitations

- Synthetic data only; no real patient data.
- Target labels are created by a heuristic, so evaluation measures how well the model reproduces that rule—not real-world medical risk.
- No clinical validation, calibration, external validation, or intended-use assessment.
- Predictions are for learning ML concepts only.

## Author

**Madhumitha M** · AI & ML student

[GitHub](https://github.com/Madhum4107) · [LinkedIn](https://www.linkedin.com/in/madhumitha4107/)
