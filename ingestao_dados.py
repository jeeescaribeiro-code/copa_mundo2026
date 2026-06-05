"""
=============================================================
COPA DO MUNDO 2026
=============================================================
"""

import sqlite3
import os
import re
import pandas as pd
from pathlib import Path

#Configurações
BASE_DIR = Path(__file__).resolve().parent
SQL_DIR = BASE_DIR / "dataset"
OUTPUT  = BASE_DIR / "output"
DB_PATH = BASE_DIR / "copa_mundo.db"
os.makedirs(OUTPUT, exist_ok=True)

#Primeiro, criamos o banco SQLite e executamos os scripts SQL para criar as tabelas e inserir os dados
conn = sqlite3.connect(DB_PATH)
arquivos = sorted([f for f in os.listdir(SQL_DIR) if f.endswith(".sql")])
print(f"Encontrados {len(arquivos)} arquivos .sql\n")

erros = []
for fname in arquivos:
    path = os.path.join(SQL_DIR, fname)
    try:
        sql = open(path, encoding="utf-8").read()
        sql = re.sub(r"/\*![\s\S]*?\*/;", "", sql)
        sql = re.sub(r"LOCK TABLES.*?;", "", sql, flags=re.DOTALL)
        sql = re.sub(r"UNLOCK TABLES;", "", sql)
        sql = sql.replace("AUTO_INCREMENT", "")
        sql = re.sub(r"\)\s*ENGINE=.*?;", ");", sql)
        sql = re.sub(r"\bUNLOCK TABLES\b.*?;", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^\s*UN\s*$", "", sql, flags=re.MULTILINE)
        sql = sql.replace("\\'", "''")
        sql = re.sub(r"^\s*KEY .*?$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"^\s*UNIQUE KEY .*?$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r",\s*\)", ")", sql)
        conn.executescript(sql)
        print(f"  [OK] {fname}")
    except Exception as e:
        erros.append(fname)
        print(f"  [ERRO] {fname}: {e}")

if erros:
    print(f"\nArquivos com erro: {erros}")

"""
Aqui começa a etapa de leitura dos dados do banco e criação dos DataFrames.
Cada tabela é lida usando pd.read_sql() e, quando necessário, fazemos ajustes como renomear colunas, converter tipos ou extrair informações.
"""

#Segundo, listamos as tabelas criadas e o número de registros em cada uma
tabelas = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
)
print(f"\nTabelas no banco ({len(tabelas)}):")
for t in tabelas["name"]:
    count = pd.read_sql(f'SELECT COUNT(*) as n FROM "{t}"', conn)["n"][0]
    print(f"  {t:<50} {count:>6} registros")

#Terceiro, carregamos os dados das tabelas principais em DataFrames do pandas, fazendo ajustes quando necessário
print("\nCarregando DataFrames principais...")

#Jogos
jogos_raw = pd.read_sql('SELECT * FROM "copa_todos_jogos"', conn)

#Renomear colunas para nomes sem espaço
RENAME_JOGOS = {
    "Key Id"                    : "key_id",
    "Tournament Id"             : "tournament_id",
    "tournament Name"           : "tournament_name",
    "Match Id"                  : "match_id",
    "Match Name"                : "match_name",
    "Stage Name"                : "stage_name",
    "Group Name"                : "group_name",
    "Group Stage"               : "group_stage",
    "Knockout Stage"            : "knockout_stage",
    "Replayed"                  : "replayed",
    "Replay"                    : "replay",
    "Match Date"                : "match_date",
    "Match Time"                : "match_time",
    "Stadium Id"                : "stadium_id",
    "Stadium Name"              : "stadium_name",
    "City Name"                 : "city_name",
    "Country Name"              : "country_name",
    "Home Team Id"              : "home_team_id",
    "Home Team Name"            : "home_team",
    "Home Team Code"            : "home_team_code",
    "Away Team Id"              : "away_team_id",
    "Away Team Name"            : "away_team",
    "Away Team Code"            : "away_team_code",
    "Score"                     : "score",
    "Home Team Score"           : "home_score",
    "Away Team Score"           : "away_score",
    "Home Team Score Margin"    : "home_score_margin",
    "Away Team Score Margin"    : "away_score_margin",
    "Extra Time"                : "extra_time",
    "Penalty Shootout"          : "penalty_shootout",
    "Score Penalties"           : "score_penalties",
    "Home Team Score Penalties" : "home_score_penalties",
    "Away Team Score Penalties" : "away_score_penalties",
    "Result"                    : "result",
    "Home Team Win"             : "home_team_win",
    "Away Team Win"             : "away_team_win",
    "Draw"                      : "draw",
}
jogos = jogos_raw.rename(columns=RENAME_JOGOS)

# Extrair o ano do tournament_id
jogos["year"] = jogos["tournament_id"].str.extract(r"(\d{4})").astype(int)

# Garantir tipos numéricos nos placares
jogos["home_score"] = pd.to_numeric(jogos["home_score"], errors="coerce").fillna(0).astype(int)
jogos["away_score"] = pd.to_numeric(jogos["away_score"], errors="coerce").fillna(0).astype(int)

print(f"jogos:{jogos.shape}")
print(f"Colunas: {jogos.columns.tolist()}\n")

#Gols
try:
    gols_a = pd.read_sql('SELECT * FROM "copa_gols_1930-1970"', conn)
    gols_b = pd.read_sql('SELECT * FROM "copa_gols_1974-1998"', conn)
    gols_c = pd.read_sql('SELECT * FROM "copa_gols_2002-2022"', conn)
    gols   = pd.concat([gols_a, gols_b, gols_c], ignore_index=True)
    print(f"  gols:         {gols.shape}")
    print(f"  Colunas: {gols.columns.tolist()}\n")
except Exception as e:
    print(f"  [ERRO gols]: {e}")
    gols = pd.DataFrame()

#Assistências
assist_frames = []
for ano in range(1958, 2023, 4):
    tname = f"copa_assistencias_{ano}"
    try:
        df = pd.read_sql(f'SELECT * FROM "{tname}"', conn)
        df["year"] = ano
        assist_frames.append(df)
    except Exception as e:
        print(f"  [aviso assist {ano}]: {e}")

assistencias = pd.concat(assist_frames, ignore_index=True) if assist_frames else pd.DataFrame()
print(f"  assistencias: {assistencias.shape}")
if not assistencias.empty:
    print(f"  Colunas: {assistencias.columns.tolist()}\n")

#Jogadores
try:
    jogadores = pd.read_sql('SELECT * FROM "copa_jogadores_1930"', conn)
    print(f"  jogadores:    {jogadores.shape}")
    print(f"  Colunas: {jogadores.columns.tolist()}\n")
except Exception as e:
    print(f"  [ERRO jogadores]: {e}")
    jogadores = pd.DataFrame()

#Jogadores destaque
try:
    destaque = pd.read_sql('SELECT * FROM "copa_jogadores_destaque"', conn)
    print(f"  destaque:     {destaque.shape}")
    print(f"  Colunas: {destaque.columns.tolist()}\n")
except Exception as e:
    print(f"  [ERRO destaque]: {e}")
    destaque = pd.DataFrame()

#Campeões
try:
    campeoes = pd.read_sql('SELECT * FROM "copa_campeoes"', conn)
    print(f"  campeoes:     {campeoes.shape}")
    print(f"  Colunas: {campeoes.columns.tolist()}\n")
except Exception as e:
    print(f"  [ERRO campeoes]: {e}")
    campeoes = pd.DataFrame()

#País sede
try:
    sede = pd.read_sql('SELECT * FROM "copa_pais_sede_publico"', conn)
    print(f"  sede:         {sede.shape}")
    print(f"  Colunas: {sede.columns.tolist()}\n")
except Exception as e:
    print(f"  [ERRO sede]: {e}")
    sede = pd.DataFrame()

#Premiações
try:
    premiacoes = pd.read_sql('SELECT * FROM "copa_premiacoes"', conn)
    print(f"  premiacoes:   {premiacoes.shape}")
    print(f"  Colunas: {premiacoes.columns.tolist()}\n")
except Exception as e:
    print(f"  [ERRO premiacoes]: {e}")
    premiacoes = pd.DataFrame()

#Quarto, salvamos cada DataFrame em um arquivo CSV separado na pasta de output
print("\nSalvando CSVs...")
def salvar(df, nome):
    if not df.empty:
        path = os.path.join(OUTPUT, f"{nome}.csv")
        df.to_csv(path, index=False)
        print(f"  [OK] {nome}.csv  →  {df.shape[0]} linhas × {df.shape[1]} colunas")
    else:
        print(f"  [VAZIO] {nome} — não salvo")
salvar(jogos,       "jogos")
salvar(gols,        "gols")
salvar(assistencias,"assistencias")
salvar(jogadores,   "jogadores")
salvar(destaque,    "destaque")
salvar(campeoes,    "campeoes")
salvar(sede,        "sede")
salvar(premiacoes,  "premiacoes")

# Quinto, geramos um relatório final com o resumo dos dados carregados e encerramos a conexão com o banco 
print("\n" + "="*55)
print("RESUMO ETAPA 1")
print("="*55)
dataframes = {
    "jogos"      : jogos,
    "gols"       : gols,
    "assistencias": assistencias,
    "jogadores"  : jogadores,
    "destaque"   : destaque,
    "campeoes"   : campeoes,
    "sede"       : sede,
    "premiacoes" : premiacoes,
}
for nome, df in dataframes.items():
    status = f"{df.shape[0]:>5} linhas × {df.shape[1]:>2} colunas" if not df.empty else "  VAZIO"
    print(f"  {nome:<15} {status}")

print(f"\nEdições disponíveis em 'jogos': {sorted(jogos['year'].unique())}")
print("\n[Etapa 1 concluída com sucesso]")
conn.close()

"""
Em resumo, este código realiza a ingestão dos dados do banco SQLite, criando DataFrames para cada tabela relevante e salvando-os em arquivos CSV. 
Ele também gera um relatório final com o resumo dos dados carregados, incluindo o número de linhas e colunas de cada DataFrame, bem como as edições disponíveis na tabela de jogos.
"""
