# Curso de Pós-Graduação em Inteligência Artificial e Aprendizado de Máquina
# Matéria: Inteligência Artificial e ML
# Aluno: Bruno Gomes da Silva
# RA: 623103792
# contato: bruno.gomes.silva@uni9.edu.br
# Data: 15 de Dezembro de 2023
# Declaro ser o autor do código fonte abaixo

import pandas as pd
from transformers import pipeline
from datetime import datetime

start_time = datetime.now()

for i in range(10000):
    i ** 100


PATH = '/home/hal9000/PycharmProjects/Modelos IA/Dados/'

text = open(PATH + "resumo_reuniao_diaria.txt", "r")
resumo_reuniao_diaria_projetos = text.readlines()

classifier = pipeline("text-classification")
outputs = classifier(resumo_reuniao_diaria_projetos)
print('\n')
resultado = pd.DataFrame(outputs)
print(resultado)

end_time = datetime.now()
time_difference = (end_time - start_time).total_seconds() * 10**3
print("Classificação Transformers durou: ", time_difference, "ms")