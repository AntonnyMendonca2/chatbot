from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np
import json
import pickle
import nltk
import random

# Baixando os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Inicialização do lematizador
lemmatizer = WordNetLemmatizer()

# Carregamento das intenções do arquivo JSON
with open('intents.json', 'r', encoding='utf-8') as intents_json:
    intents = json.load(intents_json)
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["#", "!", "@", "$", "%", "?", "*"]

# Inicialização das listas de palavras e documentos
words = []
documents = []

# Preenchimento das listas de palavras e documentos a partir dos padrões das intenções
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))

# Lematização das palavras, excluindo aquelas na lista de palavras ignoradas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Ordenação e salvamento das palavraas e classes em arquivos pickle
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicialização dos dados de treinamento
training = []
output_empty = [0] * len(classes)

# Preenchimento dos dados de treinamento com o Bag-of-words e as linhas de saída
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Embaralhamento e conversão dos dados de treinamento em um array numpy
random.shuffle(training)
training = np.asarray(training, dtype="object")
x = list(training[:, 0])
y = list(training[:, 1])

# Inicialização e configuração do modelo de rede neural
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Treinamento e salvamento do modelo
history = model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', history)

print("Fim")
