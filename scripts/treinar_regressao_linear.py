import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.ranking_utils import add_ranking_features, load_historical_rankings

OUTPUT = BASE_DIR / "output"
TABELA_OFICIAL_2026 = BASE_DIR / "dataset" / "tabela_oficial_copa_2026_grupos.csv"

FEATURES = [
    "aprov_hist_home",
    "diff_aprov_hist",
    "media_gols_pro_hist_home",
    "media_gols_contra_hist_away",
    "diff_media_gols_pro_hist",
    "diff_media_gols_contra_hist",
    "diff_saldo_medio_hist",
    "partidas_hist_home",
    "diff_partidas_hist",
    "ranking_home",
    "diff_ranking",
    "ranking_points_home",
    "diff_ranking_points",
    "ranking_available_home",
    "ranking_available_away",
    "fase_ordinal",
    "fase_knockout",
]
TARGET = "home_score"
CLASS_TARGET = "home_win"


def fit_linear_model(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X_scaled = (X - mean) / std
    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    coef = np.linalg.lstsq(X_design, y, rcond=None)[0]
    return {"mean": mean, "std": std, "intercept": coef[0], "coef": coef[1:]}


def predict(model, X):
    X_scaled = (X - model["mean"]) / model["std"]
    return model["intercept"] + X_scaled @ model["coef"]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def fit_logistic_model(X, y, lr=0.08, epochs=3500, l2=0.01):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X_scaled = (X - mean) / std
    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    weights = np.zeros(X_design.shape[1])

    for _ in range(epochs):
        probs = sigmoid(X_design @ weights)
        gradient = X_design.T @ (probs - y) / len(y)
        gradient[1:] += l2 * weights[1:] / len(y)
        weights -= lr * gradient

    return {"mean": mean, "std": std, "intercept": weights[0], "coef": weights[1:]}


def predict_proba_logistic(model, X):
    X_scaled = (X - model["mean"]) / model["std"]
    return sigmoid(model["intercept"] + X_scaled @ model["coef"])


def metrics(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    return r2, rmse, mae


def classification_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    accuracy = float((tp + tn) / len(y_true))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    brier = float(np.mean((prob - y_true) ** 2))

    order = np.argsort(prob)
    y_sorted = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos and n_neg:
        ranks = np.arange(1, len(y_true) + 1)
        sum_ranks_pos = ranks[y_sorted == 1].sum()
        roc_auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier": brier,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def cross_validate(X, y, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    folds = np.array_split(indices, n_splits)
    rows = []

    for fold in folds:
        test_idx = fold
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        model = fit_linear_model(X[train_idx], y[train_idx])
        pred_train = predict(model, X[train_idx])
        pred_test = predict(model, X[test_idx])
        r2_train, _, _ = metrics(y[train_idx], pred_train)
        r2_test, rmse, mae = metrics(y[test_idx], pred_test)
        rows.append({"r2_train": r2_train, "r2_test": r2_test, "rmse": rmse, "mae": mae})

    return pd.DataFrame(rows)


def cross_validate_logistic(X, y, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    folds = np.array_split(indices, n_splits)
    rows = []

    for fold in folds:
        test_idx = fold
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        model = fit_logistic_model(X[train_idx], y[train_idx])
        prob = predict_proba_logistic(model, X[test_idx])
        rows.append(classification_metrics(y[test_idx], prob))

    return pd.DataFrame(rows)


def build_historical_state(jogos):
    hist = {}

    for _, row in jogos.sort_values("year").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        res = row["resultado_num"]
        hs = int(row["home_score"])
        avs = int(row["away_score"])
        hh = hist.get(home, {"pontos": 0, "partidas": 0, "gols_pro": 0, "gols_contra": 0})
        ah = hist.get(away, {"pontos": 0, "partidas": 0, "gols_pro": 0, "gols_contra": 0})
        pts_home = 3 if res == 1 else (1 if res == 0 else 0)
        pts_away = 3 if res == -1 else (1 if res == 0 else 0)

        hist[home] = {
            "pontos": hh["pontos"] + pts_home,
            "partidas": hh["partidas"] + 1,
            "gols_pro": hh["gols_pro"] + hs,
            "gols_contra": hh["gols_contra"] + avs,
        }
        hist[away] = {
            "pontos": ah["pontos"] + pts_away,
            "partidas": ah["partidas"] + 1,
            "gols_pro": ah["gols_pro"] + avs,
            "gols_contra": ah["gols_contra"] + hs,
        }

    return hist


def team_features(hist, team):
    h = hist.get(team)
    if not h or h["partidas"] == 0:
        return {
            "aprov": 0.5,
            "media_gols_pro": 0.0,
            "media_gols_contra": 0.0,
            "saldo_medio": 0.0,
            "partidas": 0,
        }

    partidas = h["partidas"]
    return {
        "aprov": h["pontos"] / (partidas * 3),
        "media_gols_pro": h["gols_pro"] / partidas,
        "media_gols_contra": h["gols_contra"] / partidas,
        "saldo_medio": (h["gols_pro"] - h["gols_contra"]) / partidas,
        "partidas": partidas,
    }


def add_2026_features(jogos_2026, hist):
    rows = []
    for _, match in jogos_2026.iterrows():
        home = team_features(hist, match["home_team"])
        away = team_features(hist, match["away_team"])
        rows.append({
            "aprov_hist_home": home["aprov"],
            "aprov_hist_away": away["aprov"],
            "diff_aprov_hist": home["aprov"] - away["aprov"],
            "media_gols_pro_hist_home": home["media_gols_pro"],
            "media_gols_pro_hist_away": away["media_gols_pro"],
            "diff_media_gols_pro_hist": home["media_gols_pro"] - away["media_gols_pro"],
            "media_gols_contra_hist_home": home["media_gols_contra"],
            "media_gols_contra_hist_away": away["media_gols_contra"],
            "diff_media_gols_contra_hist": home["media_gols_contra"] - away["media_gols_contra"],
            "saldo_medio_hist_home": home["saldo_medio"],
            "saldo_medio_hist_away": away["saldo_medio"],
            "diff_saldo_medio_hist": home["saldo_medio"] - away["saldo_medio"],
            "partidas_hist_home": home["partidas"],
            "partidas_hist_away": away["partidas"],
            "diff_partidas_hist": home["partidas"] - away["partidas"],
            "fase_ordinal": 1,
            "fase_knockout": 0,
        })

    return pd.concat([jogos_2026.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def mirror_features(df):
    mirrored = pd.DataFrame(index=df.index)
    mirrored["aprov_hist_home"] = df["aprov_hist_away"]
    mirrored["diff_aprov_hist"] = -df["diff_aprov_hist"]
    mirrored["media_gols_pro_hist_home"] = df["media_gols_pro_hist_away"]
    mirrored["media_gols_contra_hist_away"] = df["media_gols_contra_hist_home"]
    mirrored["diff_media_gols_pro_hist"] = -df["diff_media_gols_pro_hist"]
    mirrored["diff_media_gols_contra_hist"] = -df["diff_media_gols_contra_hist"]
    mirrored["diff_saldo_medio_hist"] = -df["diff_saldo_medio_hist"]
    mirrored["partidas_hist_home"] = df["partidas_hist_away"]
    mirrored["diff_partidas_hist"] = -df["diff_partidas_hist"]
    mirrored["ranking_home"] = df["ranking_away"]
    mirrored["diff_ranking"] = -df["diff_ranking"]
    mirrored["ranking_points_home"] = df["ranking_points_away"]
    mirrored["diff_ranking_points"] = -df["diff_ranking_points"]
    mirrored["ranking_available_home"] = df["ranking_available_away"]
    mirrored["ranking_available_away"] = df["ranking_available_home"]
    mirrored["fase_ordinal"] = df["fase_ordinal"]
    mirrored["fase_knockout"] = df["fase_knockout"]
    return mirrored[FEATURES]


def main():
    rankings = load_historical_rankings()
    jogos = add_ranking_features(pd.read_csv(OUTPUT / "jogos_clean.csv"), rankings)
    jogos[CLASS_TARGET] = (jogos["resultado_num"] == 1).astype(int)
    df_features = jogos[["year", "home_team", "away_team", "stage_name"] + FEATURES + [TARGET, "resultado_num", CLASS_TARGET]].copy()
    df_features.to_csv(OUTPUT / "jogos_features.csv", index=False)
    (OUTPUT / "features_selecionadas.txt").write_text("\n".join(FEATURES) + "\n", encoding="utf-8")

    X = df_features[FEATURES].to_numpy(dtype=float)
    y = df_features[TARGET].to_numpy(dtype=float)
    y_class = df_features[CLASS_TARGET].to_numpy(dtype=int)
    cv = cross_validate(X, y)
    cv_log = cross_validate_logistic(X, y_class)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    n_test = int(np.ceil(len(X) * 0.2))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    holdout_model = fit_linear_model(X[train_idx], y[train_idx])
    pred_test = predict(holdout_model, X[test_idx])
    r2_test, rmse_test, mae_test = metrics(y[test_idx], pred_test)
    gap = float(cv["r2_train"].mean() - cv["r2_test"].mean())

    holdout_log_model = fit_logistic_model(X[train_idx], y_class[train_idx])
    prob_test = predict_proba_logistic(holdout_log_model, X[test_idx])
    log_test_metrics = classification_metrics(y_class[test_idx], prob_test)

    final_model = fit_linear_model(X, y)
    final_log_model = fit_logistic_model(X, y_class)
    hist = build_historical_state(jogos)
    jogos_2026 = add_2026_features(pd.read_csv(TABELA_OFICIAL_2026), hist)
    jogos_2026 = add_ranking_features(jogos_2026, rankings)
    jogos_2026["pred_home"] = np.clip(predict(final_model, jogos_2026[FEATURES].to_numpy(dtype=float)), 0, None)
    jogos_2026["pred_away"] = np.clip(predict(final_model, mirror_features(jogos_2026).to_numpy(dtype=float)), 0, None)
    jogos_2026["prob_home_win"] = predict_proba_logistic(final_log_model, jogos_2026[FEATURES].to_numpy(dtype=float))
    jogos_2026["home_win_previsto"] = (jogos_2026["prob_home_win"] >= 0.5).astype(int)
    jogos_2026["classificacao_prevista"] = np.where(
        jogos_2026["home_win_previsto"] == 1,
        jogos_2026["home_pt"] + " vence",
        jogos_2026["home_pt"] + " não vence",
    )
    jogos_2026["resultado_previsto"] = np.where(
        jogos_2026["pred_home"] > jogos_2026["pred_away"],
        jogos_2026["home_pt"] + " vence",
        np.where(jogos_2026["pred_away"] > jogos_2026["pred_home"], jogos_2026["away_pt"] + " vence", "Empate"),
    )

    cols = [
        "grupo", "rodada", "data", "horario_et", "horario_local", "estadio", "cidade",
        "home_pt", "away_pt", "home_team", "away_team",
        "aprov_hist_home", "aprov_hist_away", "diff_aprov_hist",
        "media_gols_pro_hist_home", "media_gols_pro_hist_away", "diff_media_gols_pro_hist",
        "media_gols_contra_hist_home", "media_gols_contra_hist_away", "diff_media_gols_contra_hist",
        "saldo_medio_hist_home", "saldo_medio_hist_away", "diff_saldo_medio_hist",
        "partidas_hist_home", "partidas_hist_away", "diff_partidas_hist",
        "ranking_home", "ranking_away", "diff_ranking",
        "ranking_points_home", "ranking_points_away", "diff_ranking_points",
        "ranking_available_home", "ranking_available_away",
        "fase_ordinal", "fase_knockout",
        "pred_home", "pred_away", "resultado_previsto",
        "prob_home_win", "home_win_previsto", "classificacao_prevista",
    ]
    jogos_2026[cols].to_csv(OUTPUT / "predicoes_2026.csv", index=False)

    model_payload = {
        "tipo": "Regressao Linear com ranking FIFA - numpy least squares",
        "features": FEATURES,
        "target": TARGET,
        "media_features": final_model["mean"].tolist(),
        "desvio_features": final_model["std"].tolist(),
        "intercepto": float(final_model["intercept"]),
        "coeficientes": dict(zip(FEATURES, [float(c) for c in final_model["coef"]])),
    }
    with open(OUTPUT / "modelo_copa.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    with open(OUTPUT / "modelo_regressao_linear.json", "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2, ensure_ascii=False)

    logistic_payload = {
        "tipo": "Regressao Logistica - numpy gradient descent",
        "features": FEATURES,
        "target": CLASS_TARGET,
        "classe_positiva": "resultado_num == 1, ou seja, vitoria do mandante",
        "media_features": final_log_model["mean"].tolist(),
        "desvio_features": final_log_model["std"].tolist(),
        "intercepto": float(final_log_model["intercept"]),
        "coeficientes": dict(zip(FEATURES, [float(c) for c in final_log_model["coef"]])),
        "threshold": 0.5,
    }
    with open(OUTPUT / "modelo_regressao_logistica.json", "w", encoding="utf-8") as f:
        json.dump(logistic_payload, f, indent=2, ensure_ascii=False)

    metricas = {
        "features": FEATURES,
        "regressao_linear": {
            "target": TARGET,
            "objetivo": "prever gols do mandante",
            "intercepto": float(final_model["intercept"]),
            "coeficientes": model_payload["coeficientes"],
            "r2_cv_media": float(cv["r2_test"].mean()),
            "r2_cv_desvio": float(cv["r2_test"].std()),
            "rmse_cv_media": float(cv["rmse"].mean()),
            "mae_cv_media": float(cv["mae"].mean()),
            "r2_teste": float(r2_test),
            "rmse_teste": float(rmse_test),
            "mae_teste": float(mae_test),
            "gap_overfitting": gap,
        },
        "regressao_logistica": {
            "target": CLASS_TARGET,
            "objetivo": "prever se o mandante vence ou nao vence",
            "classe_positiva": "home_win = 1 quando resultado_num == 1",
            "threshold": 0.5,
            "accuracy_cv_media": float(cv_log["accuracy"].mean()),
            "accuracy_cv_desvio": float(cv_log["accuracy"].std()),
            "precision_cv_media": float(cv_log["precision"].mean()),
            "recall_cv_media": float(cv_log["recall"].mean()),
            "f1_cv_media": float(cv_log["f1"].mean()),
            "roc_auc_cv_media": float(cv_log["roc_auc"].mean()),
            "brier_cv_media": float(cv_log["brier"].mean()),
            "accuracy_teste": log_test_metrics["accuracy"],
            "precision_teste": log_test_metrics["precision"],
            "recall_teste": log_test_metrics["recall"],
            "f1_teste": log_test_metrics["f1"],
            "roc_auc_teste": log_test_metrics["roc_auc"],
            "brier_teste": log_test_metrics["brier"],
            "matriz_confusao_teste": {
                "tp": log_test_metrics["tp"],
                "tn": log_test_metrics["tn"],
                "fp": log_test_metrics["fp"],
                "fn": log_test_metrics["fn"],
            },
            "intercepto": float(final_log_model["intercept"]),
            "coeficientes": logistic_payload["coeficientes"],
        },
        "target": TARGET,
        "r2_cv_media": float(cv["r2_test"].mean()),
        "rmse_cv_media": float(cv["rmse"].mean()),
        "mae_cv_media": float(cv["mae"].mean()),
        "r2_teste": float(r2_test),
        "rmse_teste": float(rmse_test),
        "mae_teste": float(mae_test),
        "accuracy_teste": log_test_metrics["accuracy"],
        "f1_teste": log_test_metrics["f1"],
        "roc_auc_teste": log_test_metrics["roc_auc"],
        "n_amostras_total": int(len(X)),
        "n_amostras_treino": int(len(train_idx)),
        "n_amostras_teste": int(len(test_idx)),
        "observacao": "A regressao linear estima gols do mandante; a regressao logistica estima home_win, isto e, vitoria do mandante contra nao-vitoria. decidido_penaltis foi removida por ser variavel pos-jogo.",
        "ranking_fifa": {
            "fonte_2026": "https://api.fifa.com/api/v3/rankings/?gender=1&count=300",
            "arquivo_2026": "ranking/fifa_ranking-2026-04-01.csv",
            "observacao": "Para jogos anteriores a 1992, ranking_available=0 e valores neutros sao usados.",
        },
    }
    with open(OUTPUT / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(jogos_2026[["grupo", "home_pt", "away_pt", "pred_home", "pred_away", "resultado_previsto", "prob_home_win", "classificacao_prevista"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
