
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
from api_utils import (get_model_and_features,
                       load_featured_data,
                       prepare_input_for_forecast,
                       recursive_forecast)

app = Flask(__name__)

@app.route("/")
def index():
    return "Bienvenue sur l'API de prédiction pharmaceutique !"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    code_prdt = data["code_prdt"]
    mois = int(data["mois"])
    annee = int(data["annee"])
    target_date = datetime(annee, mois, 1)

    try:
        date_cible = datetime(int(annee), int(mois), 1)
        model, top_features = get_model_and_features()
        df_featured = load_featured_data()

        last_date = df_featured["Date"].max()
        if target_date <= last_date:
            X_in = prepare_input_for_forecast(df_featured, code_prdt, target_date, top_features)
            y_pred = model.predict(X_in)[0]

    
        else:
            df_prod = df_featured[df_featured["Code prdt"] == code_prdt].copy()
            y_pred = recursive_forecast(model, df_prod, last_date, target_date, top_features)

        return jsonify({
            "Code prdt": code_prdt,
            "Mois": mois,
            "Année": annee,
            "Total prédit": round(float(y_pred), 2)
    })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)