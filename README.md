# Used-Car Price Prediction: Web App

Predicts the price of a used car in Israel and serves the model through a small Flask web app: fill in the car's details in a form and get a predicted price.

**Context:** final project, Advanced Data Mining and Analysis in Python · **Team:** Amit Stein, Nadav Mashiach

## How it works
| File | Role |
|---|---|
| `car_data_prep.py` | Cleans the listings: numeric parsing, median imputation, IQR outlier removal on year, model-name cleanup |
| `model_training.py` | 80/20 split, scaling and one-hot encoding, ElasticNet tuned with `GridSearchCV` (10-fold CV), permutation importance; saves `trained_model.pkl` and `preprocessor.pkl` |
| `api.py` | Flask app: `/` serves the form, `/predict` returns the predicted price |
| `index.html` | The input form |
| `dataset (2).csv` | 1,500 used-car listings (20 columns) |

## Run locally
```bash
pip install -r requirements.txt
python model_training.py   # trains and saves the model
python api.py              # Flask expects index.html inside a templates/ folder
```

**Tools:** Python · pandas · scikit-learn · Flask

The analysis notebook with the evaluation (test RMSE ₪11,057) is in [Final-project-in-a-machine-learning-course](https://github.com/steinamit/Final-project-in-a-machine-learning-course).
