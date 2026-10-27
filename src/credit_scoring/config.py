"""Constantes de dados, contratos e artefatos das etapas 1 e 2."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_LOANS_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
REFERENCE_SAMPLE_PATH = ROOT / "data" / "processed" / "reference_sample.csv"
MODEL_PATH = ROOT / "artifacts" / "models" / "credit_baseline.joblib"
METRICS_PATH = ROOT / "artifacts" / "reports" / "baseline_metrics.json"
COEFFICIENTS_PATH = ROOT / "artifacts" / "reports" / "baseline_coefficients.csv"
PREPARATION_REPORT_PATH = ROOT / "artifacts" / "reports" / "preparation_report.json"
FAILURE_REPORT_PATH = ROOT / "artifacts" / "reports" / "contract_failure_report.json"

RANDOM_STATE = 42
REFERENCE_SAMPLE_SIZE = 80_000
CHUNK_SIZE = 200_000
CHUNK_SAMPLE_FRACTION = 0.2
REFERENCE_START = "2012-01-01"
REFERENCE_END = "2015-12-31"
# Janela posterior a referencia. Simula a passagem do tempo sem reutilizar
# as originacoes que treinaram o baseline.
PRODUCTION_START = "2016-01-01"
PRODUCTION_END = "2018-12-31"
PRODUCTION_SAMPLE_SIZE = 20_000
MIN_REFERENCE_ROWS = 5_000
MIN_PRODUCTION_ROWS = 5_000
PRODUCTION_NATURAL_PATH = ROOT / "data" / "processed" / "production_natural.csv"
PRODUCTION_DRIFTED_PATH = ROOT / "data" / "processed" / "production_drifted.csv"
DRIFT_REPORT_PATH = ROOT / "artifacts" / "reports" / "drift_report.html"
DRIFT_INJECTION_REPORT_PATH = ROOT / "artifacts" / "reports" / "drift_injection_report.html"
DRIFT_SUMMARY_PATH = ROOT / "artifacts" / "reports" / "drift_summary.json"
PRODUCTION_SIMULATION_PATH = ROOT / "artifacts" / "reports" / "production_simulation.json"
# Recorte usado so no HTML do Evidently. PSI e KS usam a amostra inteira.
EVIDENTLY_SAMPLE_SIZE = 10_000
# PSI de credito: abaixo de 0,10 a populacao e considerada estavel;
# entre 0,10 e 0,25 ha deslocamento moderado; a partir de 0,25, relevante.
PSI_BIN_COUNT = 10
PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
# Alfa do teste KS de duas amostras. Com dezenas de milhares de linhas,
# o p-valor fica sensivel demais; o alerta operacional usa o PSI.
KS_ALPHA = 0.05
# Deslocamentos intencionais. Permanecem dentro do contrato apos o clip.
# FICO cai 40 pontos: perfil de credito pior. DTI sobe: mais alavancagem.
DRIFT_FICO_SHIFT = -40.0
DRIFT_DTI_SCALE = 1.35
DRIFT_DTI_SHIFT = 4.0
INTENTIONAL_DRIFT_COLUMNS = ("fico_range_low", "dti")

TERM_VALUES = ("36 months", "60 months")
EMP_LENGTH_VALUES = (
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
    "unknown",
)
HOME_OWNERSHIP_VALUES = ("MORTGAGE", "RENT", "OWN", "ANY", "OTHER", "NONE")
VERIFICATION_VALUES = ("Verified", "Source Verified", "Not Verified")
PURPOSE_VALUES = (
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "moving",
    "vacation",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
)
APPLICATION_TYPE_VALUES = ("Individual", "Joint App")
LOAN_STATUS_VALUES = ("Fully Paid", "Charged Off")
POSITIVE_CLASS = "Charged Off"

NUMERIC_FEATURES = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
)
CATEGORICAL_FEATURES = (
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REFERENCE_COLUMNS = FEATURE_COLUMNS + ("issue_d", "loan_status")
SOURCE_COLUMNS = REFERENCE_COLUMNS

EXCLUDED_IDENTIFIERS = (
    "id",
    "member_id",
    "emp_title",
    "url",
    "desc",
    "title",
    "zip_code",
    "addr_state",
)
EXCLUDED_LEAKAGE = (
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "out_prncp",
    "total_pymnt",
    "recoveries",
    "last_pymnt_amnt",
    "last_fico_range_high",
)
