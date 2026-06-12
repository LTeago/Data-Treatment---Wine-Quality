# Wine Quality — Tratamento de Dados

Projeto desenvolvido para a disciplina de **Programação para Ciência de Dados**, com foco no tratamento e preparação do dataset **Wine Quality**, disponível no repositório da UCI Machine Learning Repository.

## Sobre o dataset

O dataset Wine Quality reúne informações físico-químicas de vinhos tintos e brancos da região do **Vinho Verde**, em Portugal. A proposta da base é relacionar medições laboratoriais com uma nota de qualidade atribuída por avaliação sensorial humana.

Fonte dos dados: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

A base utilizada possui:

- **6.497 amostras** no total;
- **1.599 amostras** de vinho tinto;
- **4.898 amostras** de vinho branco;
- **11 atributos físico-químicos**;
- 1 variável alvo: `quality`, com notas de 0 a 10.

## Objetivo do projeto

O objetivo do projeto é preparar os dados para análise e visualização, permitindo investigar quais características físico-químicas podem influenciar a qualidade dos vinhos.

Em um contexto comercial, esse tipo de análise pode apoiar:

- controle de qualidade durante a produção;
- identificação de possíveis falhas químicas;
- comparação entre vinhos tintos e brancos;
- apoio à tomada de decisão sobre aceitação e classificação de lotes.

## Atributos analisados

| Atributo | Descrição resumida |
|---|---|
| `fixed acidity` | Acidez fixa do vinho |
| `volatile acidity` | Acidez volátil, associada a compostos que podem gerar gosto de vinagre |
| `citric acid` | Presença de ácido cítrico |
| `residual sugar` | Açúcar restante após a fermentação |
| `chlorides` | Quantidade de sais/cloretos |
| `free sulfur dioxide` | Dióxido de enxofre livre |
| `total sulfur dioxide` | Dióxido de enxofre total |
| `density` | Densidade do vinho |
| `pH` | Nível de acidez/basicidade |
| `sulphates` | Sulfatos presentes no vinho |
| `alcohol` | Teor alcoólico |
| `quality` | Nota de qualidade atribuída ao vinho |

## Etapas do tratamento

O script `main.py` realiza as seguintes etapas:

1. Carregamento dos arquivos `winequality-red.csv` e `winequality-white.csv`;
2. Criação da coluna `type`, identificando se o vinho é tinto ou branco;
3. Combinação dos dois datasets em um único DataFrame;
4. Validação das colunas esperadas;
5. Verificação de valores ausentes e registros duplicados;
6. Tratamento de outliers na coluna `residual sugar` usando IQR;
7. Criação da coluna `faixa_qualidade`, separando os vinhos em qualidade baixa, média e alta;
8. Normalização dos atributos físico-químicos com `StandardScaler`;
9. Aplicação de PCA para gerar as colunas `PC1` e `PC2`;
10. Exportação do arquivo final `resultados-winequality.csv`.

## Como executar

> Observação: em distribuições Linux/WSL mais recentes, o Python pode bloquear instalações globais com `pip` e exibir o erro `externally-managed-environment`. Por isso, o recomendado é criar um ambiente virtual antes de instalar as dependências.

### 1. Crie o ambiente virtual

No Linux ou WSL:

```bash
python3 -m venv .venv
```

Caso apareça erro dizendo que o módulo `venv` não está instalado, rode:

```bash
sudo apt update
sudo apt install python3-venv python3-full
python3 -m venv .venv
```

No Windows PowerShell:

```powershell
py -m venv .venv
```

### 2. Ative o ambiente virtual

No Linux ou WSL:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o terminal normalmente mostra `(.venv)` no início da linha.

### 3. Instale as dependências

Com o ambiente virtual ativado, execute:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Execute o script

```bash
python main.py
```

### 5. Verifique o arquivo gerado

Após a execução, será criado/atualizado o arquivo:

```text
resultados-winequality.csv
```

Esse arquivo contém os dados tratados, a coluna de tipo do vinho, a faixa de qualidade e as componentes principais geradas pelo PCA.

### 6. Finalize o ambiente virtual quando terminar

```bash
deactivate
```

Não é recomendado usar `--break-system-packages`, pois isso pode alterar pacotes do Python do sistema operacional.

## Estrutura do projeto

```text
.
├── main.py
├── README.md
├── requirements.txt
├── resultados-winequality.csv
└── winequality/
    ├── winequality-red.csv
    ├── winequality-white.csv
    └── winequality.names
```

## Principais resultados observados

A análise de correlação mostra que alguns atributos possuem relação mais forte com a qualidade dos vinhos:

- No vinho tinto, `alcohol` aparece com correlação positiva relevante com `quality`, enquanto `volatile acidity` aparece com correlação negativa.
- No vinho branco, `alcohol` também aparece como atributo positivo importante, enquanto `density` e `chlorides` aparecem com correlação negativa.
- A maior parte das amostras está concentrada em notas intermediárias, principalmente entre 5 e 6, o que indica uma base desbalanceada.

## Curiosidade sobre a formação do vinho

O vinho é formado principalmente pelo processo de fermentação alcoólica. Nesse processo, leveduras transformam os açúcares naturais da uva em álcool e dióxido de carbono. Por isso, atributos como `residual sugar`, `alcohol`, `density` e acidez são importantes para entender diferenças entre os vinhos.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- PCA
- StandardScaler

## Autores

- Kauã do Vale
- Leonardo Machado
- Thiago Lopes
