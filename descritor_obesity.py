import pickle
import pandas as pd
import numpy as np

dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
colunas_num = dados.select_dtypes(include=[np.number]).columns.tolist()
colunas_cat = dados.select_dtypes(exclude=[np.number]).columns.tolist()

dados_cat_dummy = pd.get_dummies(dados[colunas_cat], prefix_sep='_', dtype=int)
todas_as_colunas = colunas_num + dados_cat_dummy.columns.tolist()

cluster_model = pickle.load(open('cluster_obesity.pkl','rb'))
normalizador = pickle.load(open('normalizador_obesity.pkl', 'rb'))

df_centroides = pd.DataFrame(
    cluster_model.cluster_centers_,
    columns=todas_as_colunas
)

atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(df_centroides[colunas_num]),
    columns=colunas_num
).round(2)

class_dataframe = pd.DataFrame(index=df_centroides.index)

for col in colunas_cat:
    dummy_cols = [c for c in dados_cat_dummy.columns if c.startswith(col + '_')]
    
    predominante = df_centroides[dummy_cols].idxmax(axis=1)
    
    class_dataframe[col] = predominante.apply(lambda x: x.replace(col + '_', '', 1))

perfil_clusters = atributos_num_desnorm.join(class_dataframe)
perfil_clusters.index.name = 'Cluster'

print("=== PERFIL DOS CLUSTERS ===")
pd.set_option('display.max_columns', None)
print(perfil_clusters)