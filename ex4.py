# pip install scikit-learn
# pip install pandas
# pip install joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Resultado esperado
labels = [[6, 1, 3, 4, 7, 4]]
Resultado = pd.DataFrame(data = labels)
Resultado = Resultado.T
# ou as 3 linhas por: Resultado = pd.DataFrame([[6], [1], [3], [4], [7], [4]])

# Dados de entrada:
tabela = [[1,4,3],
          [2,1,1],
          [3,2,2],
          [4,2,3],
          [5,5,3],
          [6,3,2]]

tabela_att = pd.DataFrame(data=tabela,columns=["amostra","x1","x2"])

# Codificação de categoria (se necessário)
# tabela_att["amostra"] = tabela_att["amostra"].astype("category")
# cat_col = tabela_att.select_dtypes(["category"]).columns
# tabela_att[cat_col] = tabela_att[cat_col].apply(lambda x:x.cat.codes)
tabela_att["amostra"] = tabela_att["amostra"].astype("category")
tabela_att["amostra"] = tabela_att["amostra"].cat.codes

# Normalização:
escala = StandardScaler()
x_normalizado = escala.fit_transform(tabela_att)

# Criação da IA:
rna = MLPRegressor(hidden_layer_sizes=(9, 3),
                   max_iter=1000,
                   tol=0.001,
                   learning_rate_init=0.1,
                   solver='sgd',
                   activation='logistic',
                   learning_rate='constant',
                   verbose=0,
                   random_state=42)

# começando o treinamento:
rna.fit(x_normalizado, Resultado.values.ravel())
print("IA treinada com sucesso :)")

# Testando ela:
saida = rna.predict(x_normalizado)
print("Saídas esperadas:", labels)
print("Resultado obtidos:",saida)

# Salvando a IA em disco
joblib.dump(rna, "minha_IA.pkl")

# Para usar depois:
# rna = joblib.load("minha_IA.pkl")