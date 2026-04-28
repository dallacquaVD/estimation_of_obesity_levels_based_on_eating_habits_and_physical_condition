# SISTEMAS INTELIGENTES
# Modelos não supervisionados
# Base de Obesidade (ObesityDataSet)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
import pickle

# 1. Abrir os dados
# Verifique se o nome do arquivo csv está exato
dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# 2. Normalizar os dados
# 2.1 Separar atributos numéricos e categóricos dinamicamente
dados_num = dados.select_dtypes(include=[np.number]) # Pega Idade, Peso, Altura, etc
dados_cat = dados.select_dtypes(exclude=[np.number]) # Pega Gênero, Histórico Familiar, etc

# 2.2 Normalizar os dados numéricos
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

# Salvar o normalizador para uso posterior (inferência e descritor)
pic le.dump(normalizador, open('normalizador_obesity.pkl', 'wb'))

dados_num_norm = normalizador.transform(dados_num)

# 2.3 Normalizar os dados categóricos (One-Hot Encoding)
dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)

# 2.4 Reagrupar os objetos normalizados em um data frame
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)
dados_norm = dados_num_norm.join(dados_cat_norm)

# 3. HIPERPARAMETRIZAR  
# Vamos determinar o número ótimo de clusters (Método do Cotovelo)
distortions = []
# Limitamos o K até 20 para evitar sobrecarga de processamento
K = range(1, 21) 

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
    # Calcular e armazenar a distorção
    distortions.append(
        sum(np.min(cdist(dados_norm, cluster_model.cluster_centers_, 'euclidean'), axis=1)) / dados_norm.shape[0]
    )

# Determinar o número ótimo de clusters pelo cálculo da maior distância à reta
x0 = K[0]
y0 = distortions[0]
xn = K[-1]    
yn = distortions[-1]
distances = []

for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distances.append(numerador/denominador)

k_otimo = K[distances.index(max(distances))]
print(f"O número ótimo de clusters encontrado foi: {k_otimo}")

# (Opcional) Descomente abaixo para ver o gráfico do cotovelo
# fig, ax = plt.subplots()
# ax.plot(K, distortions)
# ax.plot([x0, xn], [y0, yn], 'r--') # Reta auxiliar
# ax.set(xlabel='n Clusters', ylabel='Distorcoes')
# ax.grid()
# plt.show()

# 4. TREINAR O MODELO FINAL
modelo_final = KMeans(n_clusters=k_otimo, random_state=42).fit(dados_norm)

# Salvar o modelo treinado
pickle.dump(modelo_final, open('cluster_obesity.pkl', 'wb'))
print("Treinamento finalizado e modelo salvo com sucesso!")