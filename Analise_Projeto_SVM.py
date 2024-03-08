# Curso de Pós-Graduação em Inteligência Artificial e Aprendizado de Máquina
# Matéria: Inteligência Artificial e ML
# Aluno: Bruno Gomes da Silva
# RA: 623103792
# contato: bruno.gomes.silva@uni9.edu.br
# Data: 15 de Dezembro de 2023
# Declaro ser o autor do código fonte abaixo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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

X_train, X_test, y_train, y_test = train_test_split(
    df_treinamento['resume'].values,
    df_treinamento['sentiment'].values,
    test_size=0.2,
    random_state=1
)

vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy: : {:.2f}". format(score))

end_time = datetime.now()
time_difference = (end_time - start_time).total_seconds() * 10**3
print("Classificação SVM durou: ", time_difference, "ms")