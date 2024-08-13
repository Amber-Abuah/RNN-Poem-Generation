import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import os

SEQ_LENGTH = 50
STEP_SIZE = 2

model_output_path = "PoemGen.keras"
input_text_path = "poem.txt"
generated_text_length = 700
retrain = False
batch_size = 256
epoch_num = 150

text = open(input_text_path, "rb").read().decode(encoding="utf-8").lower()
character_set = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(character_set))
index_to_char = dict((i, c) for i, c in enumerate(character_set))

class PreloadedRNNModel:
    def __init__(self):
        self.model = tf.keras.models.load_model(model_output_path)

    def generate_text(self, temperature, output_length):
        start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
        sentence = text[start_index: start_index + SEQ_LENGTH]
        generated_text = sentence

        for _ in range(output_length):
            x = np.zeros((1, SEQ_LENGTH, len(character_set)))
            for j, character in enumerate(sentence):
                x[0, j, char_to_index[character]] = 1

            predictions = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(predictions, temperature)
            next_character = index_to_char[next_index]

            generated_text += next_character
            sentence = sentence[1:] + next_character

        return generated_text

    def sample(self, preds, temperature):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        return np.argmax(probs)
    

# Create RNN model
if not os.path.exists(model_output_path) or retrain:
    sentences = []
    next_characters = []

    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i: i + SEQ_LENGTH])
        next_characters.append(text[i + SEQ_LENGTH])

    x = np.zeros((len(sentences), SEQ_LENGTH, len(character_set)), dtype=np.bool_)
    y = np.zeros((len(sentences), len(character_set)), dtype=np.bool_)

    for i, sentence in enumerate(sentences):
        for j, character in enumerate(sentence):
            x[i, j, char_to_index[character]] = 1
        y[i, char_to_index[next_characters[i]]] = 1

    if not os.path.exists(model_output_path) or retrain:
        model = Sequential()
        model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(character_set))))
        model.add(Dense(len(character_set)))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=RMSprop(0.01))
        model.fit(x, y, batch_size=batch_size, epochs=epoch_num)
        model.save(model_output_path)
        print("Model saved to path", model_output_path)

if __name__ == "__main__":
    model = PreloadedRNNModel()

    for temperature in [0.2, 0.4, 0.6, 0.8, 1]:
        print("\nGenerated text with temperature: ", temperature)
        print(model.generate_text(temperature, 500))