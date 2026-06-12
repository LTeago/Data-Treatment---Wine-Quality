import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=== INICIANDO O PROCESSAMENTO DOS DADOS DE QUALIDADE DE VINHO ===")

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS
# ==========================================
# Caminhos dos arquivos originais do dataset Wine Quality da UCI
PASTA_DADOS = Path("winequality")
ARQUIVO_TINTO = PASTA_DADOS / "winequality-red.csv"
ARQUIVO_BRANCO = PASTA_DADOS / "winequality-white.csv"

# Arquivo final gerado pelo processamento
ARQUIVO_SAIDA = "resultados-winequality.csv"

# Lista dos 11 atributos físico-químicos usados como preditores
ATRIBUTOS_QUIMICOS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Coluna que será analisada para remoção de outliers
COLUNA_OUTLIER = "residual sugar"


# ==========================================
# 2. CARREGAMENTO E COMBINAÇÃO DOS ARQUIVOS
# ==========================================
# Verifica se os arquivos existem antes de tentar carregar
if not ARQUIVO_TINTO.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {ARQUIVO_TINTO}")

if not ARQUIVO_BRANCO.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {ARQUIVO_BRANCO}")

# Lendo os CSVs originais do dataset da UCI
df_red = pd.read_csv(ARQUIVO_TINTO, sep=";")
df_white = pd.read_csv(ARQUIVO_BRANCO, sep=";")

# Criando a coluna identificadora do tipo de vinho antes da união
df_red["type"] = "red"
df_white["type"] = "white"

# Concatenando os dois datasets em um único DataFrame
df = pd.concat([df_red, df_white], ignore_index=True)

print(f"-> Sucesso: {df.shape[0]} amostras totais carregadas na memória.")
print(f"   Vinhos tintos: {df_red.shape[0]}")
print(f"   Vinhos brancos: {df_white.shape[0]}")


# ==========================================
# 3. VALIDAÇÃO INICIAL DOS DADOS
# ==========================================
# Confere se todas as colunas esperadas estão presentes
colunas_esperadas = ATRIBUTOS_QUIMICOS + ["quality", "type"]
colunas_ausentes = [coluna for coluna in colunas_esperadas if coluna not in df.columns]

if colunas_ausentes:
    raise ValueError(f"As seguintes colunas estão ausentes no dataset: {colunas_ausentes}")

# Verifica valores ausentes
total_ausentes = df.isnull().sum().sum()
print(f"-> Valores ausentes encontrados: {total_ausentes}")

# Verifica registros duplicados
total_duplicados = df.duplicated().sum()
print(f"-> Registros duplicados encontrados: {total_duplicados}")

# Observação:
# Os duplicados não foram removidos automaticamente porque podem representar
# amostras semelhantes ou repetidas do processo de avaliação, e não necessariamente erro.


# ==========================================
# 4. TRATAMENTO DE OUTLIERS
# ==========================================
# Aplicando o método IQR na coluna residual sugar.
# O IQR identifica valores muito distantes da distribuição principal dos dados.
q1 = df[COLUNA_OUTLIER].quantile(0.25)
q3 = df[COLUNA_OUTLIER].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Filtrando o dataset para remover os valores considerados outliers
df_limpo = df[
    (df[COLUNA_OUTLIER] >= limite_inferior) &
    (df[COLUNA_OUTLIER] <= limite_superior)
].copy()

outliers_removidos = df.shape[0] - df_limpo.shape[0]

print("-> Tratamento de outliers concluído.")
print(f"   Coluna analisada: {COLUNA_OUTLIER}")
print(f"   Limite inferior: {limite_inferior:.3f}")
print(f"   Limite superior: {limite_superior:.3f}")
print(f"   Outliers removidos: {outliers_removidos}")
print(f"   Amostras restantes: {df_limpo.shape[0]}")


# ==========================================
# 5. DISCRETIZAÇÃO DA VARIÁVEL ALVO
# ==========================================
# A coluna quality possui notas numéricas.
# Aqui criamos uma nova coluna categórica para facilitar análises e visualizações.
bins = [0, 4, 6, 10]
labels = ["Baixa", "Média", "Alta"]

df_limpo["faixa_qualidade"] = pd.cut(
    df_limpo["quality"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print("-> Discretização concluída. Distribuição das faixas de qualidade:")
print(df_limpo["faixa_qualidade"].value_counts().sort_index())


# ==========================================
# 6. NORMALIZAÇÃO DOS ATRIBUTOS
# ==========================================
# Criando uma cópia para exportação.
# Os valores originais são preservados para facilitar interpretação no dashboard.
df_exportar = df_limpo.copy()

# O StandardScaler transforma os atributos para média 0 e desvio padrão 1.
# Isso evita que atributos com escalas maiores dominem o PCA.
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df_limpo[ATRIBUTOS_QUIMICOS])

print("-> Normalização concluída.")
print("   Os atributos químicos foram colocados na mesma escala estatística.")


# ==========================================
# 7. REDUÇÃO DE DIMENSIONALIDADE COM PCA
# ==========================================
# O PCA reduz os 11 atributos químicos para 2 componentes principais.
# Isso facilita a visualização dos dados em gráficos 2D.
pca = PCA(n_components=2)
componentes_principais = pca.fit_transform(dados_normalizados)

# Adicionando as coordenadas do PCA no DataFrame final
df_exportar["PC1"] = componentes_principais[:, 0]
df_exportar["PC2"] = componentes_principais[:, 1]

variancia_explicada = pca.explained_variance_ratio_.sum()

print("-> Redução de dimensionalidade com PCA concluída.")
print(f"   Variância total explicada por PC1 e PC2: {variancia_explicada:.2%}")


# ==========================================
# 8. ANÁLISE DE CORRELAÇÃO
# ==========================================
# Correlação de Pearson entre os atributos químicos e a nota quality.
# Valores positivos indicam relação direta.
# Valores negativos indicam relação inversa.
correlacoes = (
    df_exportar[ATRIBUTOS_QUIMICOS + ["quality"]]
    .corr(numeric_only=True)["quality"]
    .drop("quality")
    .sort_values(ascending=False)
)

print("-> Correlações dos atributos com a qualidade:")
print(correlacoes.round(3))


# ==========================================
# 9. EXPORTAÇÃO DO ARQUIVO FINAL
# ==========================================
# Salvando o DataFrame final em um novo arquivo CSV preparado.
# O encoding utf-8-sig ajuda na abertura correta em Excel/Power BI.
df_exportar.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

print(f"\n-> Arquivo final gerado: {ARQUIVO_SAIDA}")
print("=== SCRIPT FINALIZADO COM SUCESSO ===")