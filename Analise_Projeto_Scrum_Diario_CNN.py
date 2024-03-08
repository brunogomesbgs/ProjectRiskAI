# Curso de Pós-Graduação em Inteligência Artificial e Aprendizado de Máquina
# Matéria: Inteligência Artificial e ML
# Aluno: Bruno Gomes da Silva
# RA: 623103792
# contato: bruno.gomes.silva@uni9.edu.br
# Data: 15 de Dezembro de 2023
# Declaro ser o autor do código fonte abaixo
from keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

start_time = datetime.now()

for i in range(10000):
    i ** 100


PATH = '/home/hal9000/PycharmProjects/Modelos IA/Dados/'

filepath_dict = {
    'treinamento':   PATH + 'resumo_reuniao_diaria.txt',
    'teste': PATH + 'resumo_reuniao_diario_teste.txt'
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['resume', 'sentiment'], sep=',')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)

df_treinamento = df[df['source'] == 'treinamento']
df_teste = df[df['source'] == 'teste']

MAX_LEN = 200
EMBEDDING_DIMENSION = 50
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 1
TRAIN_SIZE = 8
TEST_SIZE = 0.2
LEARNING_RATE = 0.1
NEURONS_SIZE = 16
DROPOUT_RATE = 0.25

X_train, X_test, y_train, y_test = train_test_split(
    df_treinamento['resume'].values,
    df_treinamento['sentiment'].values,
    test_size=TEST_SIZE,
    random_state=NUM_CLASSES
)

tokenizer = Tokenizer(num_words=MAX_LEN)
tokenizer.fit_on_texts(X_train)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    padding='post',
    maxlen=MAX_LEN
)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    padding='post',
    maxlen=MAX_LEN
)

encoder = LabelEncoder()

y_test_labels = encoder.fit_transform(y_test)
y_train_labels = encoder.fit_transform(y_train)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()

model.add(
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIMENSION,
        input_length=MAX_LEN
    )
)

model.add(layers.Dense(NEURONS_SIZE, activation='relu'))
model.add(Dropout(DROPOUT_RATE))

model.add(layers.Flatten())

model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
        X_train,
        y_train_labels,
        epochs=EPOCHS,
        verbose=False,
        validation_data=(
            X_test,
            y_test_labels
        ),
        batch_size=BATCH_SIZE
)

loss, accuracy = model.evaluate(X_train, y_train_labels, verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy))
print("Training Loss Accuracy: {:.2f}".format(loss))

loss, accuracy = model.evaluate(X_test, y_test_labels, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy))
print("Training Loss Accuracy: {:.2f}".format(loss))

print('\n')
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.show()

end_time = datetime.now()

time_difference = (end_time - start_time).total_seconds() * 10**3
print("Classificação CNN durou: ", time_difference, "ms")