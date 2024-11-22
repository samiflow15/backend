from flask import Flask, request, jsonify
import joblib
import pandas as pd
from openai import OpenAI
import logging
import os
from flask_cors import CORS


app = Flask(__name__)

CORS(app)  # This will allow all origins by default
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load the model (ensure the path is correct)
model = joblib.load('trained_model.pkl')

openai_api_key = os.getenv('OPENAI_API_KEY')

def create_gpt_completion(prompt):
  client = OpenAI(
      # This is the default and can be omitted
      api_key=openai_api_key,
  )

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model="gpt-3.5-turbo",
  )
  return chat_completion.choices[0].message.content

def construir_prompt(edad, genero, frecuencia_cardiaca, presion_sistolica, presion_diastolica, azucar_sangre, ck_mb, troponina, clase_ataque_cardiaco):
    # Definir la categoría del resultado de la clase
    estado_ataque_cardiaco = "No presenta riezgo de ataque al corazón" if clase_ataque_cardiaco == "negative" else "Presenta riezgo de ataque al corazón"

    # Construir el prompt
    prompt = (
        f"Detalles del paciente:\n"
        f"Edad: {edad}\n"
        f"Género: {'femenino' if genero==0 else 'masculino'}\n"
        f"Frecuencia cardíaca (impulso): {frecuencia_cardiaca}\n"
        f"Presión arterial sistólica: {presion_sistolica}\n"
        f"Presión arterial diastólica: {presion_diastolica}\n"
        f"Azúcar en sangre: {azucar_sangre}\n"
        f"CK-MB: {ck_mb}\n"
        f"Troponina: {troponina}\n"
        f"Conclusión diagnóstica: {estado_ataque_cardiaco}\n\n"
        f"Con base en la información anterior, por favor proporcione una sugerencia para la condición del paciente. Sé analitico con el resultado y los datos para dar una sugerencia muy concreta. Ten en cuenta que en la sugerencia es importante referenciar la conclusión diagnótica, y mencionar  los valores de los indicador que más contribuyeron al diagnóstico y porqué"
    )

    return prompt

def predict_with_model(model, age, gender, impluse, pressure_high, pressure_low, glucose, kcm, troponin):
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'impluse': [impluse],
        'pressurehight': [pressure_high],
        'pressurelow': [pressure_low],
        'glucose': [glucose],
        'kcm': [kcm],
        'troponin': [troponin]
    })

    prediction = model.predict(input_data)

    return prediction[0]



@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.get_json()
    prediction = predict_with_model(
        model,
        user_input['age'],
        user_input['gender'],
        user_input['impulse'],
        user_input['pressure_high'],
        user_input['pressure_low'],
        user_input['glucose'],
        user_input['kcm'],
        user_input['troponin'],
    )

    return jsonify({'prediction': prediction})

@app.route('/suggest', methods=['POST'])
def suggest():
    user_input = request.get_json()
    
    prompt = construir_prompt(
        user_input['age'],
        user_input['gender'],
        user_input['impulse'],
        user_input['pressure_high'],
        user_input['pressure_low'],
        user_input['glucose'],
        user_input['kcm'],
        user_input['troponin'],
        user_input['prediction']
    )
    
    logger.info(f"Prompt: {prompt}")
    prediction = create_gpt_completion(prompt)
    
    
    return jsonify({'suggestion': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
