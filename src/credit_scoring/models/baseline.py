"""Baseline de regressao logistica protegido pelo contrato de dados."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    COEFFICIENTS_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    RANDOM_STATE,
)
from credit_scoring.data.schema import REFERENCE_SCHEMA, SCORING_SCHEMA

TARGET_COLUMN = "defaulted"


def defaulted_target(loan_status: pd.Series) -> pd.Series:
    """Converte o desfecho terminal na classe positiva de inadimplencia.

    Args:
        loan_status: Status original do emprestimo.

    Returns:
        Serie binaria em que 1 representa Charged Off.
    """
    return (loan_status == POSITIVE_CLASS).astype(int)


def build_pipeline() -> Pipeline:
    """Monta o pipeline de preprocessamento e regressao logistica.

    Returns:
        Pipeline ainda nao ajustado.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def kolmogorov_smirnov(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Calcula o KS de credito entre adimplentes e inadimplentes.

    O valor e a maior distancia absoluta entre as taxas acumuladas de
    verdadeiros positivos e falsos positivos. Nao e um p-valor.

    Args:
        y_true: Classe observada.
        y_score: Probabilidade estimada da classe positiva.

    Returns:
        Estatistica KS no intervalo [0, 1].
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(true_positive_rate - false_positive_rate)))


def evaluate_scores(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    """Resume a discriminacao do baseline no conjunto de teste.

    Args:
        y_true: Classe observada.
        y_score: Probabilidade estimada de inadimplencia.

    Returns:
        Metricas de ranking e calibracao, sem limiar de negocio.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "ks": kolmogorov_smirnov(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(np.mean(y_true)),
    }


def _feature_names(pipeline: Pipeline) -> list[str]:
    """Recupera os nomes das features apos o one-hot.

    Args:
        pipeline: Pipeline ja ajustado.

    Returns:
        Nomes na mesma ordem dos coeficientes.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def export_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Exporta coeficientes para auditoria posterior, sem dados de clientes.

    Args:
        pipeline: Pipeline ajustado.

    Returns:
        Tabela ordenada pelo valor absoluto do coeficiente.
    """
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    table = pd.DataFrame(
        {
            "feature": _feature_names(pipeline),
            "coefficient": classifier.coef_[0],
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")


def train_baseline(reference: pd.DataFrame) -> dict[str, Any]:
    """Treina o baseline somente depois da validacao integral do contrato.

    Diferente da preparacao, o treino nao coloca linhas invalidas em
    quarentena. Se o arquivo de referencia foi adulterado, o treino para.

    Args:
        reference: Amostra de referencia.

    Returns:
        Artefato com pipeline, metricas e metadados.
    """
    validated = REFERENCE_SCHEMA.validate(reference, lazy=True)
    features = validated.loc[:, list(FEATURE_COLUMNS)]
    target = defaulted_target(validated["loan_status"])
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    pipeline = build_pipeline()
    pipeline.fit(train_features, train_target)
    scores = pipeline.predict_proba(test_features)[:, 1]
    metrics = evaluate_scores(test_target, scores)
    metrics["n_train"] = int(len(train_target))
    metrics["n_test"] = int(len(test_target))
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "coefficients": export_coefficients(pipeline),
        "metadata": {
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "positive_class": POSITIVE_CLASS,
            "feature_columns": list(FEATURE_COLUMNS),
            "trained_at": datetime.now(UTC).isoformat(),
            "decision_threshold": None,
            "threshold_note": (
                "O limiar de concessao e decisao de governanca e nao faz parte do baseline."
            ),
        },
    }


def save_baseline(artifact: dict[str, Any], model_path: Path = MODEL_PATH) -> None:
    """Salva o pipeline e omite a tabela de coeficientes do binario principal.

    Args:
        artifact: Retorno de `train_baseline`.
        model_path: Destino Joblib.
    """
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": artifact["pipeline"],
        "metrics": artifact["metrics"],
        "metadata": artifact["metadata"],
    }
    joblib.dump(payload, destination)


def load_baseline(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Carrega o artefato do baseline.

    Args:
        model_path: Caminho do Joblib.

    Returns:
        Dicionario com pipeline, metricas e metadados.
    """
    return joblib.load(model_path)


def predict_default_probability(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.Series:
    """Recusa a inferencia se a entrada violar o contrato de scoring.

    Args:
        artifact: Artefato carregado ou recem-treinado.
        frame: Lote de solicitacoes, sem identificadores.

    Returns:
        Probabilidade estimada de Charged Off.

    Raises:
        pandera.errors.SchemaError: Se o lote nao cumprir o contrato.
    """
    validated = SCORING_SCHEMA.validate(frame, lazy=True)
    scores = artifact["pipeline"].predict_proba(validated.loc[:, list(FEATURE_COLUMNS)])[:, 1]
    return pd.Series(scores, index=validated.index, name="default_probability")


def save_coefficients(table: pd.DataFrame, path: Path = COEFFICIENTS_PATH) -> None:
    """Persiste coeficientes agregados para a documentacao de governanca.

    Args:
        table: Saida de `export_coefficients`.
        path: Destino CSV.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
