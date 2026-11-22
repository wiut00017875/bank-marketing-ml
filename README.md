# Bank Marketing – Who Will Open a Term Deposit?

Coursework 1 for **6COSC017C-n Machine Learning and Data Analytics** (WIUT, 2026–2027).

A Portuguese bank called clients by phone to sell a term deposit. Only about 11% of the clients said "yes".
In this project, I build machine learning models that predict **before the call** if a client will subscribe.
With these models, the bank can call the most promising clients first.

- **Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
  file `bank-additional-full.csv` (41,188 rows, 20 inputs + target). Licence: CC BY 4.0.
  Moro, S., Cortez, P. and Rita, P. (2014) *Decision Support Systems*, 62, pp. 22–31.
- **Task:** supervised binary classification (`y` = yes / no).
- **Models:** Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest
  (all tuned with `GridSearchCV` and 5-fold stratified cross-validation).
- **Live app:** https://bank-marketing-ml-00017875.streamlit.app/

## Project structure

```
bank-marketing-ml/
├── app/                    Streamlit multi-page app
│   ├── Home.py             start page
│   ├── common.py           cached loading of data and models
│   └── pages/              1 Data Exploration, 2 Preprocessing, 3 Model Training,
│                           4 Evaluation (dashboard), 5 Prediction
├── data/                   original UCI data + its description file
├── models/                 trained pipelines (.joblib) + metrics.json
├── notebooks/
│   └── bank_marketing.ipynb   full analysis: load → EDA → preparation → training → evaluation
├── reports/figures/        charts saved by the notebook (used in the report)
├── src/bank_ml.py          shared cleaning, feature engineering and preprocessing code
├── requirements.txt        direct dependencies (exact versions)
├── requirements.lock       full locked environment (pip freeze)
├── Dockerfile              full containerisation of the app
└── LICENSE                 MIT
```

## Setup

You need **Python 3.13** (other 3.11+ versions should also work with `requirements.txt`).

```bash
git clone https://github.com/wiut00017875/bank-marketing-ml.git
cd bank-marketing-ml
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt        # or, for the exact same environment:
pip install -r requirements.lock
```

## Run the notebook

```bash
jupyter notebook notebooks/bank_marketing.ipynb
```

Run all cells (it takes about 3–5 minutes). The notebook saves the trained models to `models/`
and the charts to `reports/figures/`. You can also run it from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/bank_marketing.ipynb
```

## Run the Streamlit app

```bash
streamlit run app/Home.py
```

Then open http://localhost:8501. The trained models are already in `models/`, so the app works
without running the notebook first.

### With Docker

```bash
docker build -t bank-marketing-ml .
docker run -p 8501:8501 bank-marketing-ml
```

## Main decisions

- `duration` is **removed**: it is only known after the call, so it leaks the answer.
- `emp.var.rate` and `nr.employed` are removed: correlation > 0.9 with `euribor3m`.
- `unknown` values are treated as missing and filled with the most frequent value (fitted on the training set only).
- `pdays = 999` was wrong for 4,110 rows, so "contacted before" is taken from `previous`.
- The data is imbalanced, so models are compared with **PR-AUC**, recall, precision and F1, not accuracy.
- The decision threshold of each model is chosen on a separate **validation set**; the **test set** is used only once.

## Results (test set)

| Model | Recall | TNR | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy (always "no") | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.113 |
| Logistic Regression | 0.520 | 0.929 | 0.481 | 0.500 | 0.796 | 0.456 |
| KNN | 0.544 | 0.919 | 0.462 | 0.500 | 0.796 | 0.444 |
| Decision Tree | 0.596 | 0.907 | 0.449 | 0.512 | 0.797 | 0.461 |
| **Random Forest** | 0.595 | 0.917 | 0.476 | 0.529 | 0.817 | 0.495 |

Final model: **Random Forest**. Calling the top 20% of clients (by model score) reaches 66.8% of all subscribers.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, UCI Machine Learning Repository.
