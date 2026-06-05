"""
=============================================================
COPA DO MUNDO 2026
=============================================================
"""

import pandas as pd
import numpy  as np
import os
from pathlib import Path

from scripts.ranking_utils import add_ranking_features, load_historical_rankings

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output"
os.makedirs(OUTPUT, exist_ok=True)

#Primeiro, vamos carregar o dataset processado na etapa anterior (EDA) para realizar a limpeza e padronização dos dados, preparando-os para a modelagem preditiva.
print("Carregando jogos_processado.csv...")
jogos = pd.read_csv(os.path.join(OUTPUT, "jogos_processado.csv"))
print(f"Shape: {jogos.shape}")
print(f"Seleções únicas: {pd.concat([jogos['home_team'], jogos['away_team']]).nunique()}")
shape_original = jogos.shape

#Segundo, vamos padronizar os nomes das seleções para evitar inconsistências causadas por mudanças históricas, variações de nome ou erros de digitação. Isso é crucial para garantir que o modelo reconheça corretamente as seleções e possa aprender padrões históricos de desempenho.
#Problema: o dataset tem nomes históricos que mudaram ao longo das décadas.
#Exemplos encontrados nos dados:
#   "West Germany"          → "Germany"       (reunificação 1990)
#   "Soviet Union"          → "Russia"        (dissolução 1991)
#   "Yugoslavia"            → "Serbia"        (dissolução 1992)
#   "Czechoslovakia"        → "Czech Republic"(separação 1993)
#   "Dutch East Indies"     → "Indonesia"     (independência 1945)
#   "Serbia and Montenegro" → "Serbia"
#   "Zaire"                 → "DR Congo"      (renomeado 1997)
#   "East Germany"          → "Germany"       (reunificação 1990)
#   "Republic of Ireland"   → "Ireland"       (nome oficial)
#   "Ivory Coast"           → "Cote d'Ivoire" (nome FIFA oficial)

PAISES_MAP = {
    # Alemanha
    "West Germany": "Germany",
    "East Germany": "Germany",
    # União Soviética
    "Soviet Union": "Russia",
    # Iugoslávia e sucessores
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    # Tchecoslováquia
    "Czechoslovakia": "Czech Republic",
    # Indonésia
    "Dutch East Indies": "Indonesia",
    # República do Congo
    "Zaire": "DR Congo",
    # Irlanda
    "Republic of Ireland" : "Ireland",
    # Costa do Marfim
    "Ivory Coast": "Cote d'Ivoire",
    # Bósnia
    "Bosnia and Herzegovina": "Bosnia",
    # Trinidad
    "Trinidad and Tobago": "Trinidad & Tobago",
}

print("\n" + "="*55)
print("PADRONIZAÇÃO DE NOMES DE SELEÇÕES")
print("="*55)

#Aplicar nas colunas de time
for col in ["home_team", "away_team"]:
    antes = jogos[col].nunique()
    jogos[col] = jogos[col].replace(PAISES_MAP)
    depois = jogos[col].nunique()
    if antes != depois:
        print(f"  {col}: {antes} → {depois} seleções únicas")

#Relatório: quais foram renomeadas e quantas partidas afetadas
print("\n  Mapeamentos aplicados:")
for antigo, novo in PAISES_MAP.items():
    n_home = (jogos["home_team"] == novo).sum()
    n_away = (jogos["away_team"] == novo).sum()
    # Verificar se existia antes (difícil após substituição, então só informamos)
    print(f"    '{antigo}' → '{novo}'")

total_selecoes = pd.concat([jogos["home_team"], jogos["away_team"]]).nunique()
print(f"\n Seleções únicas após padronização: {total_selecoes}")

#Terceiro, vamos transformar a coluna "result" (resultado) em uma variável numérica que possa ser usada para modelagem preditiva. O resultado original é categórico ("home team win", "draw", "away team win"), e vamos mapear isso para valores numéricos (1, 0, -1) para facilitar a análise e a construção de modelos de machine learning.
#Transformar "home team win" / "draw" / "away team win" em 1 / 0 / -1
RESULTADO_MAP = {
    "home team win" :  1,   # vitória do mandante
    "draw"          :  0,   # empate
    "away team win" : -1,   # vitória do visitante
}
jogos["resultado_num"] = jogos["result"].map(RESULTADO_MAP)

#Verificar se sobrou algum nulo (resultado não mapeado)
nulos_resultado = jogos["resultado_num"].isnull().sum()
if nulos_resultado > 0:
    print(f"  [AVISO] {nulos_resultado} resultados não mapeados:")
    print(jogos[jogos["resultado_num"].isnull()]["result"].value_counts())
else:
    print("  Todos os resultados mapeados com sucesso!")

print(f"\n  Distribuição resultado_num:")
print(jogos["resultado_num"].value_counts().to_string())

"""
Deste modo, a coluna "resultado_num" agora contém valores numéricos que representam o resultado de cada partida, facilitando a análise estatística e a construção de modelos preditivos para prever os resultados futuros com base em características históricas dos jogos e das seleções.
"""

#Quarto, vamos consolidar as fases do torneio em categorias mais simples para facilitar a análise e a modelagem. O dataset original possui várias descrições de fases (ex: "group stage", "second group stage", "final round", "round of 16", "quarter-finals", etc.), e vamos agrupar essas fases em categorias mais amplas (ex: "group", "knockout") para reduzir a complexidade e melhorar a capacidade do modelo de aprender padrões relevantes.
print("  Fases originais:")
print(jogos["stage_name"].value_counts().to_string())
#Simplificar fases similares
FASE_SIMPLES = {
    "group stage" : "group",
    "second group stage": "group",    # Copa de 1974/78 tinha 2ª fase de grupos
    "final round" : "group",    # Copa de 1950 tinha fase final em grupo
    "round of 16": "round_of_16",
    "quarter-finals": "quarter_finals",
    "semi-finals" : "semi_finals",
    "third-place match" : "third_place",
    "final" : "final",
}
jogos["fase_simples"] = jogos["stage_name"].str.lower().map(FASE_SIMPLES).fillna("group")
print("\n  Fases simplificadas:")
print(jogos["fase_simples"].value_counts().to_string())


#Quinto, tratar os jogos que foram decididos por prorrogação ou pênaltis. O dataset original tem colunas indicando se a partida teve prorrogação ou pênaltis, mas o placar registrado é o do tempo normal + prorrogação, e o resultado final (vencedor) é determinado pelos pênaltis. Para a modelagem de regressão de gols, usaremos o placar sem pênaltis, mas para a classificação de resultado, usaremos o resultado final que já reflete o vencedor correto.
# Em jogos com pênaltis, o placar registrado é o do tempo normal + prorrogação
# O vencedor é determinado pelos pênaltis, não pelo placar de gols
# Para a regressão de gols: usar o placar sem pênaltis (já está assim)
# Para classificação de resultado: o resultado já reflete o vencedor correto
n_et = int(jogos["extra_time"].sum())
n_pen = int(jogos["penalty_shootout"].sum())
print(f" Partidas com prorrogação: {n_et}")
print(f" Partidas com pênaltis: {n_pen}")
print(f" Nota: placar registrado = tempo normal + prorrogação (sem pênaltis)")
# Criar flag combinada útil para o modelo
jogos["decidido_penaltis"] = jogos["penalty_shootout"].astype(int)


#Sexto, criar variáveis de aproveitamento histórico de cada seleção em Copas anteriores, para capturar o desempenho passado e a experiência em torneios, o que pode ser um fator importante para prever resultados futuros. O aproveitamento será calculado como a proporção de pontos conquistados (3 por vitória, 1 por empate, 0 por derrota) em relação ao total possível de pontos nas edições anteriores da Copa do Mundo.
# Para cada partida, calcular o aproveitamento acumulado de cada
# seleção nas edições ANTERIORES (evita vazamento de dados)
# Calcular pontos por partida (3 vitória, 1 empate, 0 derrota)
def calcular_historico(jogos):
    """Retorna features históricas pré-jogo, sem usar informação da própria partida."""
    historico = {}  # {team: pontos, partidas, gols_pro, gols_contra}
    linhas = []

    def estado(team):
        return historico.get(team, {"pontos": 0, "partidas": 0, "gols_pro": 0, "gols_contra": 0})

    def features_time(h):
        partidas = h["partidas"]
        if partidas == 0:
            return {
                "aprov": 0.5,
                "media_gols_pro": 0.0,
                "media_gols_contra": 0.0,
                "saldo_medio": 0.0,
                "partidas": 0,
            }

        return {
            "aprov": h["pontos"] / (partidas * 3),
            "media_gols_pro": h["gols_pro"] / partidas,
            "media_gols_contra": h["gols_contra"] / partidas,
            "saldo_medio": (h["gols_pro"] - h["gols_contra"]) / partidas,
            "partidas": partidas,
        }

    for _, row in jogos.iterrows():
        ta = row["home_team"]
        tb = row["away_team"]
        ha = estado(ta)
        hb = estado(tb)
        fa = features_time(ha)
        fb = features_time(hb)

        linhas.append({
            "aprov_hist_home": fa["aprov"],
            "aprov_hist_away": fb["aprov"],
            "media_gols_pro_hist_home": fa["media_gols_pro"],
            "media_gols_pro_hist_away": fb["media_gols_pro"],
            "media_gols_contra_hist_home": fa["media_gols_contra"],
            "media_gols_contra_hist_away": fb["media_gols_contra"],
            "saldo_medio_hist_home": fa["saldo_medio"],
            "saldo_medio_hist_away": fb["saldo_medio"],
            "partidas_hist_home": fa["partidas"],
            "partidas_hist_away": fb["partidas"],
        })

        res = row["resultado_num"]
        pts_a = 3 if res == 1 else (1 if res == 0 else 0)
        pts_b = 3 if res == -1 else (1 if res == 0 else 0)
        gols_a = int(row["home_score"])
        gols_b = int(row["away_score"])

        historico[ta] = {
            "pontos": ha["pontos"] + pts_a,
            "partidas": ha["partidas"] + 1,
            "gols_pro": ha["gols_pro"] + gols_a,
            "gols_contra": ha["gols_contra"] + gols_b,
        }
        historico[tb] = {
            "pontos": hb["pontos"] + pts_b,
            "partidas": hb["partidas"] + 1,
            "gols_pro": hb["gols_pro"] + gols_b,
            "gols_contra": hb["gols_contra"] + gols_a,
        }

    return pd.DataFrame(linhas)

# Ordenar por ano e índice para garantir ordem cronológica
jogos = jogos.sort_values(["year", jogos.index.name or "index"] if jogos.index.name else ["year"]).reset_index(drop=True)
historico_features = calcular_historico(jogos)
jogos = pd.concat([jogos, historico_features], axis=1)
jogos["diff_aprov_hist"] = jogos["aprov_hist_home"] - jogos["aprov_hist_away"]
jogos["diff_media_gols_pro_hist"] = jogos["media_gols_pro_hist_home"] - jogos["media_gols_pro_hist_away"]
jogos["diff_media_gols_contra_hist"] = jogos["media_gols_contra_hist_home"] - jogos["media_gols_contra_hist_away"]
jogos["diff_saldo_medio_hist"] = jogos["saldo_medio_hist_home"] - jogos["saldo_medio_hist_away"]
jogos["diff_partidas_hist"] = jogos["partidas_hist_home"] - jogos["partidas_hist_away"]

rankings = load_historical_rankings()
jogos = add_ranking_features(jogos, rankings)

print(f"aprov_hist_home — média: {jogos['aprov_hist_home'].mean():.3f}")
print(f"aprov_hist_away — média: {jogos['aprov_hist_away'].mean():.3f}")
print(f"diff_aprov_hist — range: {jogos['diff_aprov_hist'].min():.3f} a {jogos['diff_aprov_hist'].max():.3f}")


#Sétimo, calcular o saldo acumulado de gols para cada seleção em cada edição da Copa, considerando apenas as partidas anteriores dentro da mesma edição. O saldo acumulado é a diferença entre os gols marcados e sofridos por uma seleção até aquele ponto do torneio, e pode ser um indicador importante do desempenho atual da equipe na competição.
# Para cada partida, saldo acumulado das partidas ANTERIORES
# dentro da mesma edição (somente fase de grupos)
saldo_home_acum = []
saldo_away_acum = []

for _, row in jogos.iterrows():
    ano = row["year"]
    ta  = row["home_team"]
    tb  = row["away_team"]
    idx = row.name

    # Partidas anteriores na mesma Copa
    anteriores = jogos[(jogos["year"] == ano) & (jogos.index < idx)]

    # Saldo acumulado do time A
    gols_a_pro  = (anteriores[anteriores["home_team"] == ta]["home_score"].sum() + anteriores[anteriores["away_team"] == ta]["away_score"].sum())
    gols_a_con  = (anteriores[anteriores["home_team"] == ta]["away_score"].sum() + anteriores[anteriores["away_team"] == ta]["home_score"].sum())

    # Saldo acumulado do time B
    gols_b_pro  = (anteriores[anteriores["home_team"] == tb]["home_score"].sum() + anteriores[anteriores["away_team"] == tb]["away_score"].sum())
    gols_b_con  = (anteriores[anteriores["home_team"] == tb]["away_score"].sum() + anteriores[anteriores["away_team"] == tb]["home_score"].sum())

    saldo_home_acum.append(int(gols_a_pro - gols_a_con))
    saldo_away_acum.append(int(gols_b_pro - gols_b_con))

jogos["saldo_acum_home"] = saldo_home_acum
jogos["saldo_acum_away"] = saldo_away_acum
jogos["diff_saldo_acum"] = jogos["saldo_acum_home"] - jogos["saldo_acum_away"]

print(f"saldo_acum_home — média: {jogos['saldo_acum_home'].mean():.2f}")
print(f"saldo_acum_away — média: {jogos['saldo_acum_away'].mean():.2f}")


#Oitavo, realizar uma análise de qualidade dos dados para identificar e tratar valores nulos, inconsistências ou outliers que possam afetar a modelagem preditiva. Isso inclui verificar se todas as partidas têm um resultado registrado, se os tipos de dados estão corretos e se há alguma anomalia nos valores que precisa ser corrigida antes de avançar para a etapa de construção do modelo.

# Nulos
nulos = jogos.isnull().sum()
nulos_existem = nulos[nulos > 0]
if nulos_existem.empty:
    print("  Nenhum valor nulo!")
else:
    print("  Nulos encontrados:")
    print(nulos_existem.to_string())

# Tipos
print(f"\n  Tipos das colunas principais:")
cols_check = ["home_score","away_score","total_gols","resultado_num",
              "fase_ordinal","fase_knockout","aprov_hist_home",
              "saldo_acum_home","diff_saldo_acum"]
for col in cols_check:
    if col in jogos.columns:
        print(f"    {col:<25} {str(jogos[col].dtype):<12} | range: {jogos[col].min()} a {jogos[col].max()}")

# Integridade: toda partida deve ter resultado
sem_resultado = jogos["resultado_num"].isnull().sum()
print(f"\n  Partidas sem resultado mapeado: {sem_resultado}")

# Shape final
print(f"\n Shape original : {shape_original}")
print(f" Shape final : {jogos.shape}")
print(f" Colunas novas : {jogos.shape[1] - shape_original[1]}")
print(f"\n  Colunas disponíveis para a Etapa 4 (features):")
for col in jogos.columns:
    print(f"{col}")

#Por fim, vamos salvar o dataset limpo e processado em um novo arquivo CSV para uso nas próximas etapas do projeto, como modelagem preditiva e análise de desempenho. O arquivo será salvo com um nome que indica que é a versão limpa dos dados, para facilitar a organização e o acesso futuro.
path_out = os.path.join(OUTPUT, "jogos_clean.csv")
jogos.to_csv(path_out, index=False)
print(f"\nDataset limpo salvo: jogos_clean.csv  ({jogos.shape})")
print("\n[Etapa 3 concluída]")

"""
Por fim, esse arquivo trata sobre a limpeza e padronização dos dados, preparando-os para a modelagem preditiva;
Ele inclui a padronização dos nomes das seleções, a transformação do resultado em uma variável numérica, a simplificação das fases do torneio, o tratamento de jogos decididos por prorrogação
ou pênaltis, a criação de variáveis de aproveitamento histórico e saldo acumulado de gols, e uma análise de qualidade dos dados para garantir que estejam prontos para a construção do modelo.
"""
