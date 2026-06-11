import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=== INICIANDO O PROCESSAMENTO DOS DADOS DE QUALIDADE DE VINHO ===")

# ==========================================
# 1. CARREGAMENTO E COMBINAÇÃO DOS ARQUIVOS LOCAIS
# ==========================================
# Lendo os CSVs originais do UC
df_red = pd.read_csv('winequality/winequality-red.csv', sep=';')
df_white = pd.read_csv('winequality/winequality-white.csv', sep=';')

# Criando a coluna identificadora 'type' antes da união
df_red['type'] = 'red'
df_white['type'] = 'white'

# Concatenando os dois datasets em um único DataFrame
df = pd.concat([df_red, df_white], ignore_index=True)
print(f"-> Sucesso: {df.shape[0]} amostras totais carregadas na memória.")

# ==========================================
# 2. TRATAMENTO DE OUTLIERS (Açúcar Residual)
# ==========================================
# Aplicando o método do IQR para o residual sugar
Q1 = df['residual sugar'].quantile(0.25)
Q3 = df['residual sugar'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrando o dataset para remover as anomalias
df_limpo = df[(df['residual sugar'] >= limite_inferior) & (df['residual sugar'] <= limite_superior)].copy()
print(f"-> Outliers removidos: Dataset reduzido de {df.shape[0]} para {df_limpo.shape[0]} amostras.")

# ==========================================
# 3. DISCRETIZAÇÃO DA VARIÁVEL ALVO (Quality)
# ==========================================
# Criando as 3 faixas naturais de qualidade (Baixa, Média, Alta)
bins = [0, 4, 6, 10]
labels = ['Baixa', 'Média', 'Alta']
df_limpo['faixa_qualidade'] = pd.cut(df_limpo['quality'], bins=bins, labels=labels)
print("-> Discretização concluída. Distribuição das faixas:")
print(df_limpo['faixa_qualidade'].value_counts())

# ==========================================
# 4. NORMALIZAÇÃO / ESCALONAMENTO DOS ATRIBUTOS
# ==========================================
# Lista dos 11 atributos físico-químicos preditores
atributos_quimicos = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol']

# Criando uma cópia para preservar os valores originais
df_exportar = df_limpo.copy()

# Inicializando o StandardScaler (Média = 0, Variância = 1)
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df_limpo[atributos_quimicos])
print("-> Normalização concluída: Todos os atributos químicos estão na mesma escala física.")

# ==========================================
# 5. REDUÇÃO DE DIMENSIONALIDADE (PCA)
# ==========================================
# Aplicando o PCA para extrair 2 Componentes Principais (PC1 e PC2)
pca = PCA(n_components=2)
componentes_principais = pca.fit_transform(dados_normalizados)

# Adicionando as coordenadas do PCA diretamente no nosso DataFrame de exportação
df_exportar['PC1'] = componentes_principais[:, 0]
df_exportar['PC2'] = componentes_principais[:, 1]

print(f"-> Redução de Dimensionalidade (PCA) concluída.")
print(f"   Variância total explicada pelos 2 componentes: {pca.explained_variance_ratio_.sum():.2%}")

# ==========================================
# 6. EXPORTAÇÃO DO ARQUIVO FINAL
# ==========================================
# Salvando o DataFrame final em um novo arquivo CSV preparado
nome_arquivo_saida = 'resultados-winequality.csv'
df_exportar.to_csv(nome_arquivo_saida, index=False)

print("\n=== SCRIPT FINALIZADO COM SUCESSO ===")
