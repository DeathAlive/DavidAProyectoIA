from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Recibir datos del formulario (debes ajustar nombres según columnas)
            age = float(request.form['age'])
            balance = float(request.form['balance'])
            day = float(request.form['day'])
            duration = float(request.form['duration'])
            campaign = float(request.form['campaign'])
            pdays = float(request.form['pdays'])
            previous = float(request.form['previous'])

            # Aquí debes añadir la codificación para las variables categóricas que uses
            # Por simplicidad se ponen solo numéricas

            # Crear array para predecir
            X_input = np.array([[age, balance, day, duration, campaign, pdays, previous]])
            
            # Escalar
            X_scaled = scaler.transform(X_input)

            # Predecir probabilidad y aplicar umbral 0.6
            proba = model.predict_proba(X_scaled)[0,1]
            pred = 1 if proba >= 0.6 else 0

            return render_template('index.html', prediction=pred, probability=proba)
        except Exception as e:
            return f"Error en los datos: {e}"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
