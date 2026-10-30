"""Desktop app: Medical Insurance Cost + Ad Click prediction.

Run with:  python app.py
"""
import tkinter as tk
from tkinter import ttk, messagebox

import models

# (label, min, max) used to validate numeric input. Ranges match the training data.
MEDICAL_RANGES = {
    "Age": (18, 64),
    "BMI": (15, 54),
    "Children": (0, 5),
}
AD_RANGES = {
    "Age": (18, 65),
    "Daily internet usage (min)": (30, 360),
    "Daily time on site (min)": (5, 120),
    "Ad topic score (0-100)": (0, 100),
    "Area income ($)": (20000, 120000),
    "Hour of day (0-23)": (0, 23),
}


def read_number(var, label, lo, hi):
    """Parse a text field as a number and check it is within [lo, hi]."""
    try:
        value = float(var.get())
    except ValueError:
        raise ValueError(f"'{label}' must be a number.")
    if not lo <= value <= hi:
        raise ValueError(f"'{label}' should be between {lo} and {hi} (the range in the training data).")
    return value


class Tab(ttk.Frame):
    """Shared layout: a grid of labelled inputs, a Predict button, a result area."""

    def __init__(self, parent, title, model):
        super().__init__(parent, padding=20)
        self.model = model
        self.row = 0
        ttk.Label(self, text=title, font=("TkDefaultFont", 14, "bold")).grid(
            row=self.row, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self.row += 1

    def add_entry(self, label, default):
        var = tk.StringVar(value=str(default))
        ttk.Label(self, text=label).grid(row=self.row, column=0, sticky="w", pady=4, padx=(0, 16))
        ttk.Entry(self, textvariable=var, width=18).grid(row=self.row, column=1, sticky="w")
        self.row += 1
        return var

    def add_choice(self, label, options, default):
        var = tk.StringVar(value=default)
        ttk.Label(self, text=label).grid(row=self.row, column=0, sticky="w", pady=4, padx=(0, 16))
        ttk.Combobox(self, textvariable=var, values=options, state="readonly",
                     width=16).grid(row=self.row, column=1, sticky="w")
        self.row += 1
        return var

    def add_footer(self, command):
        ttk.Button(self, text="Predict", command=self.safe(command)).grid(
            row=self.row, column=0, columnspan=2, pady=16, sticky="w")
        self.row += 1
        self.result = ttk.Label(self, text="", font=("TkDefaultFont", 13, "bold"))
        self.result.grid(row=self.row, column=0, columnspan=2, sticky="w")
        self.row += 1
        metrics = "   ".join(
            f"{k}: {v:.3f}" if v < 100 else f"{k}: {v:,.0f}"
            for k, v in self.model.metrics.items())
        ttk.Label(self, text=f"Model performance on test data  |  {metrics}",
                  foreground="gray").grid(row=self.row, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def safe(self, command):
        def run():
            try:
                self.result.config(text=command())
            except ValueError as err:
                messagebox.showwarning("Invalid input", str(err))
        return run


class MedicalTab(Tab):
    def __init__(self, parent, model):
        super().__init__(parent, "Medical Insurance Cost", model)
        self.age = self.add_entry("Age", 35)
        self.sex = self.add_choice("Sex", list(models.SEX_CODES), "Female")
        self.bmi = self.add_entry("BMI", 28.0)
        self.num_children = self.add_entry("Children", 0)
        self.smoker = tk.BooleanVar(value=False)
        ttk.Label(self, text="Smoker").grid(row=self.row, column=0, sticky="w", pady=4)
        ttk.Checkbutton(self, variable=self.smoker).grid(row=self.row, column=1, sticky="w")
        self.row += 1
        self.region = self.add_choice("Region", list(models.REGION_CODES), "Northeast")
        self.add_footer(self.predict)

    def predict(self):
        r = MEDICAL_RANGES
        charges = models.predict_charges(
            self.model,
            read_number(self.age, "Age", *r["Age"]),
            self.sex.get(),
            read_number(self.bmi, "BMI", *r["BMI"]),
            int(read_number(self.num_children, "Children", *r["Children"])),
            self.smoker.get(),
            self.region.get(),
        )
        return f"Estimated yearly charges: ${charges:,.2f}"


class AdTab(Tab):
    def __init__(self, parent, model):
        super().__init__(parent, "Ad Click Prediction", model)
        defaults = [35, 180, 50, 50, 65000, 12]
        self.vars = [self.add_entry(label, d) for label, d in zip(AD_RANGES, defaults)]
        self.add_footer(self.predict)

    def predict(self):
        values = [read_number(v, label, lo, hi)
                  for v, (label, (lo, hi)) in zip(self.vars, AD_RANGES.items())]
        prob = models.predict_click(self.model, *values)
        verdict = "likely to click" if prob >= 0.5 else "unlikely to click"
        return f"Click probability: {prob:.0%}  ({verdict})"


def main():
    root = tk.Tk()
    root.title("ML Predictor")
    root.resizable(False, False)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    notebook.add(MedicalTab(notebook, models.train_medical_model()), text="Medical Cost")
    notebook.add(AdTab(notebook, models.train_ad_model()), text="Ad Click")
    root.mainloop()


if __name__ == "__main__":
    main()
