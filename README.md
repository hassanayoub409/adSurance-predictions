# ML Predictor — Desktop App

A small desktop app (Python + Tkinter) that bundles two beginner machine learning projects I built with scikit-learn:

| Tab | Task | Model | Test performance |
|---|---|---|---|
| **Medical Cost** | Predict yearly medical insurance charges | Polynomial (degree 2) Linear Regression | R² ≈ 0.87, MAE ≈ $2,730 |
| **Ad Click** | Predict whether a user clicks an ad | Logistic Regression | Accuracy ≈ 60% |

Enter values, press **Predict**, and get an instant result. The models train in a second or two when the app starts, using the CSV files in `data/`. Due to the scope of project, I have not bothered storing trained parameters.

## Run it

Requires Python 3.9+ (Tkinter ships with Python on Windows and macOS; on Linux you may need `sudo apt install python3-tk`).

```bash
git clone https://github.com/hassanayoub409/adSurance-predictions.git
cd <adSurance-predictions>
pip install -r requirements.txt
python app.py
```

## Project structure

```
app.py          # Tkinter GUI (two tabs)
models.py       # Training + prediction code for both models
data/           # insurance.csv, ad_click_dataset.csv
notebooks/      # Original exploratory notebooks (EDA, plots, evaluation)
```

## What I learned / notes

- Both models use scikit-learn `Pipeline`s, so scaling and feature expansion happen automatically at prediction time.
- **Medical cost:** adding polynomial features lifted R² from ~0.78 (plain linear) to ~0.87, since smoking and BMI interact strongly.
- **Ad click:** accuracy is only around 60%. These features carry limited signal for a linear model, so this is an  baseline rather than a polished result. Next steps would be trying tree-based models and feature engineering.
- The `UserID` column was removed from the ad model's features, since an ID carries no real predictive meaning.

## Disclaimer

This is a learning project. The predictions are **not** real insurance quotes or marketing advice.

## License

MIT — see [LICENSE](LICENSE). Datasets keep their original licenses.