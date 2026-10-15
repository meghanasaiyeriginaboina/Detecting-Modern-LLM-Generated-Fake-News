# Detecting Modern LLM-Generated Fake News: A Cross-Generator Robustness Study

A web application and machine-learning study that detects fake news, including text written by large language models, using TF-IDF and stylometric classifiers, and tests how well those detectors hold up across different text generators, noise, and short inputs.

![App screenshot](docs/screenshot.png)

## Features

- **Live detector:** paste a headline and article, choose a model, and get a fake-probability gauge with a REAL/FAKE verdict.
- **Explainability:** for the linear models, see the words that pushed the score toward fake or real.
- **Model comparison:** run one article through all five detectors side by side.
- **Results dashboard:** browse the experiment tables and figures from the notebooks inside the app.
- **Reliability warning:** flags short texts (under 80 words), where the models are less reliable.

## Models

| Model | Features |
|---|---|
| Logistic Regression | TF-IDF word 1–2-grams |
| Linear SVM | TF-IDF word 1–2-grams |
| SGD (modified Huber) | TF-IDF word 1–2-grams |
| Complement Naive Bayes | TF-IDF word 1–2-grams |
| Gradient Boosting (HistGB) | Stylometric features (length, capitalization, punctuation, lexical diversity, headline style) |

Fake news is the positive class in all metrics (label 0 = fake, 1 = real).

## Results (test set)

Evaluated once on a held-out test split of 8,023 articles after duplicate and leakage removal.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogReg (TF-IDF) | 0.9788 | 0.9773 | 0.9767 | 0.9770 | 0.9968 |
| Linear SVM (TF-IDF) | 0.9806 | 0.9781 | 0.9797 | 0.9789 | 0.9970 |
| SGD-Huber (TF-IDF) | 0.9806 | 0.9786 | 0.9792 | 0.9789 | 0.9960 |
| Complement NB (TF-IDF) | 0.9425 | 0.9274 | 0.9497 | 0.9384 | 0.9820 |
| HistGB (Stylometric) | 0.9777 | 0.9749 | 0.9767 | 0.9758 | 0.9983 |

Beyond in-distribution accuracy, the study measures:

- **Cross-generator transfer:** detectors trained on human news are tested on text from LLM generators they have never seen, and compared with detectors trained with generator text included.
- **Robustness:** performance under text perturbations and noise.
- **Length sensitivity:** F1 as the number of available words shrinks.

Figures and tables are in `outputs/figures` and `outputs/tables` and are displayed on the app's **Results** tab.

## Methodology

1. **Data preparation:** load the data, remove duplicates and train/validation/test overlap, and scrub source-specific giveaway tokens (agency names, URLs, handles) so the models cannot rely on them.
2. **Training:** hyperparameter tuning on the validation split, then models saved to `outputs/models/phase2_models.joblib`.
3. **Test and robustness evaluation:** a single pass over the untouched test set, followed by perturbation and length stress tests.
4. **Cross-generator detection:** LLM-generated fake news is produced with several generators, then evaluated with zero-shot, trained-with-generator, and leave-one-generator-out settings. All texts are truncated to 80 words so article length cannot act as a shortcut.

## Project structure

```
.
├── app.py                 # Flask server and API routes
├── detector.py            # scrub(), stylometric features, detect(), explain()
├── templates/index.html   # Web page
├── static/                # style.css, app.js
├── outputs/
│   ├── models/            # phase2_models.joblib
│   ├── data/              # scrub_patterns.json
│   ├── tables/            # result CSVs
│   └── figures/           # result figures
├── notebooks/             # Data, Train, Test, Detection notebooks
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/meghanasaiyeriginaboina/Detecting-Modern-LLM-Generated-Fake-News.git
cd Detecting-Modern-LLM-Generated-Fake-News
pip install -r requirements.txt
python app.py
```

Open <http://localhost:5000> in your browser.

The saved model was trained with **scikit-learn 1.9.0**. Use the same version to avoid loading errors.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/detect` | POST | `{title, text, model}` returns fake probability, verdict, and word contributions |
| `/api/compare` | POST | `{title, text}` returns results from all models |
| `/api/results` | GET | Experiment tables and figure list |

## Limitations

- Predictions are statistical estimates, not proof that a story is true or false.
- Texts under about 80 words are less reliable.
- Models reflect the patterns of their training data and may not transfer to new sources or newer LLMs.
- Do not use this tool as the sole basis for judging the truthfulness of a story.

## Tech stack

Python, Flask, scikit-learn, pandas, NumPy, joblib, HTML/CSS/JavaScript.

## Author

**Meghana** — [GitHub](https://github.com/meghanasaiyeriginaboina)
