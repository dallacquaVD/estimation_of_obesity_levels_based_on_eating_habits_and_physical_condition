import pandas as pd
import pickle
import numpy as np

# 1. Carregar os arquivos salvos no treinamento
try:
    normalizador = pickle.load(open('normalizador_obesity.pkl', 'rb'))
    modelo_cluster = pickle.load(open('cluster_obesity.pkl', 'rb'))
except FileNotFoundError:
    print("Erro: Arquivos .pkl não encontrados. Rode o script de treinamento primeiro.")
    exit()

# 2. Definir a estrutura de colunas que o modelo espera
colunas_num = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Adicionamos as colunas NObeyesdad para ficar idêntico ao modelo treinado
colunas_finais = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender_Female', 'Gender_Male', 
    'family_history_with_overweight_no', 'family_history_with_overweight_yes',
    'FAVC_no', 'FAVC_yes', 
    'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
    'SMOKE_no', 'SMOKE_yes', 
    'SCC_no', 'SCC_yes',
    'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking',
    'NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight', 'NObeyesdad_Obesity_Type_I',
    'NObeyesdad_Obesity_Type_II', 'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I',
    'NObeyesdad_Overweight_Level_II'
]

# 3. Dados do Paciente Desconhecido (Exemplo para teste)
paciente_dados_num = [[
    25.0,  # Age
    1.75,  # Height
    80.0,  # Weight
    2.0,   # FCVC (Consumo de vegetais)
    3.0,   # NCP (Refeições principais)
    2.0,   # CH2O (Água)
    1.0,   # FAF (Atividade física)
    0.0    # TUE (Tempo de tela)
]]

# Defina as categorias que o paciente se encaixa
paciente_categorias = {
    'Gender_Male': 1,
    'family_history_with_overweight_yes': 1,
    'FAVC_yes': 1,
    'CAEC_Sometimes': 1,
    'SMOKE_no': 1,
    'SCC_no': 1,
    'CALC_Sometimes': 1,
    'MTRANS_Public_Transportation': 1
}

# 4. Processamento para Inferência
# 4.1 Normalizar a parte numérica
paciente_num_norm = normalizador.transform(paciente_dados_num)
df_paciente_num = pd.DataFrame(paciente_num_norm, columns=colunas_num)

# 4.2 Criar DataFrame vazio com todas as colunas (Dummies) e preencher com o paciente
df_template = pd.DataFrame(columns=colunas_finais)
df_paciente_cat = pd.DataFrame([paciente_categorias])

# 4.3 Concatenar e preencher o que faltou com 0
paciente_final = pd.concat([df_paciente_num, df_paciente_cat], axis=1)
paciente_final = pd.concat([paciente_final, df_template]).fillna(0)

# Garantir que as colunas estejam na ordem exata que o modelo foi treinado
paciente_final = paciente_final[colunas_finais]

# 5. Predição
cluster_designado = modelo_cluster.predict(paciente_final)

print(f"\n--- Resultado da Inferência ---")
print(f"O paciente foi classificado no: Cluster {cluster_designado[0]}")
print("Para entender o perfil desse grupo, consulte a saída do 'descritor_obesity.py'.")