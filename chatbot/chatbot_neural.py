import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json

lemmatizer = WordNetLemmatizer()

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')

def clean_text(text):
    """
    Limpa todas as sentenças inseridas.
    """
    sentence_words = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def create_bag_of_words(text, words):
    """
    Pega as sentenças que são limpas e cria um pacote de palavras que são usadas 
    para classes de previsão que são baseadas nos resyultados que obtivemos treinando o modelo.
    """
    sentence_words = clean_text(text)
    bag = [0] * len(words)
    
    for sentence_word in sentence_words:
        for i, word in enumerate(words):
            if word == sentence_word:
                bag[i] = 1
                
    return np.array(bag)

def predict_class(text, model):
    """
    Faz a previsao do pacote de palavras, usamos como limite de erro 0.25 para evitarmos overfitting
    e classificamos esses resultados por força da probabilidade.
    """
    prediction = create_bag_of_words(text, words)
    response_prediction = model.predict(np.array([prediction]))[0]
    results = [[index, response] for index, response in enumerate(response_prediction) if response > 0.25]
    
    if "1" not in str(prediction) or len(results) == 0:
        results = [[0, response_prediction[0]]]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents, intents_json):
    """
    pega a lista gerada e verifica o arquivo json e produz a maior parte das respostas com a maior probabilidade.
    """
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    
    for idx in list_of_intents:
        if idx['tag'] == tag:
            result = random.choice(idx['responses']) if len(idx['responses']) > 1 else idx['responses'][0]
            break
            
    return result

from flask import Flask, request

app = Flask(__name__)

@app.route('/receive_message', methods=['POST'])
def receive_message():
    # Receba os dados da mensagem do corpo da solicitação POST
    message_data = request.get_json()
    print(message_data)
    with open('intents.json', 'r', encoding='utf-8') as intents_json:
        intents_json = json.load(intents_json)
    
    # Responda à solicitação com uma mensagem
    intents = predict_class(message_data['conteudo'], model)
    response = get_response(intents, intents_json)
    
    return response.encode('utf-8', 'ignore').decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
