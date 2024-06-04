from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
import logging

print(sklearn.__version__)


try:
    # Loading models
    dtr = pickle.load(open('dtr.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except Exception as e:
    logging.error(f"Error loading models: {e}")

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            Year = request.form.get('Year')
            average_rain_fall_mm_per_year = request.form.get('average_rain_fall_mm_per_year')
            pesticides_tonnes = request.form.get('pesticides_tonnes')
            avg_temp = request.form.get('avg_temp')
            Area = request.form.get('Area')
            Item = request.form.get('Item')

            # Log form values for debugging
            logging.info(f"Predict form data: Year={Year}, average_rain_fall_mm_per_year={average_rain_fall_mm_per_year}, "
                         f"pesticides_tonnes={pesticides_tonnes}, avg_temp={avg_temp}, Area={Area}, Item={Item}")

            if not (Year and average_rain_fall_mm_per_year and pesticides_tonnes and avg_temp and Area and Item):
                raise ValueError("Missing input data")

            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            transformed_features = preprocessor.transform(features)
            prediction = dtr.predict(transformed_features).reshape(1, -1)

            return render_template('index.html', prediction=prediction)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', error="An error occurred during prediction.")

@app.route("/croprecommendationaction", methods=['POST'])
def croprecommendationaction():
    try:
        if request.method == 'POST':
            # logging.info("Hi Im here")
            N = request.form.get('Nitrogen')
            P = request.form.get('Phosphorus')
            K = request.form.get('Potassium')
            temp = request.form.get('Temperature')
            humidity = request.form.get('Humidity')
            ph = request.form.get('pH')
            rainfall = request.form.get('Rainfall')

            # Log form values for debugging
            logging.info(f"Crop recommendation form data: N={N}, P={P}, K={K}, temp={temp}, "
                         f"humidity={humidity}, ph={ph}, rainfall={rainfall}")

            if not (N and P and K and temp and humidity and ph and rainfall):
                raise ValueError("Missing input data")

            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)
            prediction = model.predict(final_features)

            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                         8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                         14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            return render_template('index.html', result=result)
    except ValueError as ve:
        logging.error(f"ValueError during crop recommendation: {ve}")
        return render_template('index.html', error=str(ve))
    except Exception as e:
        logging.error(f"Error during crop recommendation: {e}")
        return render_template('index.html', error="An error occurred during crop recommendation.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
