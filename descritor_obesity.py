# Imports
import pickle
import pandas as pd
import numpy as np

# 1. Abrir os dados originais apenas para recuperar as colunas
dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
colunas_num = dados.select_dtypes(include=[np.number]).columns.tolist()
colunas_cat = dados.select_dtypes(exclude=[np.number]).columns.tolist()

# Precisamos do formato exato das colunas categóricas pós get_dummies
dados_cat_dummy = pd.get_dummies(dados[colunas_cat], prefix_sep='_', dtype=int)
todas_as_colunas = colunas_num + dados_cat_dummy.columns.tolist()

# 2. Carregar modelo e normalizador
cluster_model = pickle.load(open('cluster_obesity.pkl','rb'))
normalizador = pickle.load(open('normalizador_obesity.pkl', 'rb'))

# 3. Converter os centroides em DataFrame
df_centroides = pd.DataFrame(
    cluster_model.cluster_centers_,
    columns=todas_as_colunas
)

# 4. Desnormalizar os atributos numéricos
atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(df_centroides[colunas_num]),
    columns=colunas_num
).round(2)

# 5. Organizar os atributos categóricos (CORRIGIDO)
# Vamos pegar a característica predominante de cada categoria, evitando empates
class_dataframe = pd.DataFrame(index=df_centroides.index)

for col in colunas_cat:
    # Pega as opções dummy dessa categoria (ex: Gender_Male e Gender_Female)
    dummy_cols = [c for c in dados_cat_dummy.columns if c.startswith(col + '_')]
    
    # idxmax pega a coluna que teve a maior média (a mais frequente no cluster)
    predominante = df_centroides[dummy_cols].idxmax(axis=1)
    
    # Limpa o texto para ficar bonito de ler (tira 'Gender_' e deixa só 'Female')
    class_dataframe[col] = predominante.apply(lambda x: x.replace(col + '_', '', 1))

# 6. Unir tudo para ver o perfil completo do Cluster
perfil_clusters = atributos_num_desnorm.join(class_dataframe)
perfil_clusters.index.name = 'Cluster'

print("=== PERFIL DOS CLUSTERS ===")
# Exibir todas as colunas no print sem cortar
pd.set_option('display.max_columns', None)
print(perfil_clusters)