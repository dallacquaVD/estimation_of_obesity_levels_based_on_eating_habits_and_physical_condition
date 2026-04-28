import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
import pickle

dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

dados_num = dados.select_dtypes(include=[np.number])
dados_cat = dados.select_dtypes(exclude=[np.number])

scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

pickle.dump(normalizador, open('normalizador_obesity.pkl', 'wb'))

dados_num_norm = normalizador.transform(dados_num)

dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)

dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)
dados_norm = dados_num_norm.join(dados_cat_norm)

distortions = []
K = range(1, 21) 

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
    distortions.append(
        sum(np.min(cdist(dados_norm, cluster_model.cluster_centers_, 'euclidean'), axis=1)) / dados_norm.shape[0]
    )

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

modelo_final = KMeans(n_clusters=k_otimo, random_state=42).fit(dados_norm)

pickle.dump(modelo_final, open('cluster_obesity.pkl', 'wb'))
print("Treinamento finalizado e modelo salvo com sucesso!")