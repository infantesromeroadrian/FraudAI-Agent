"""Tool definitions for FraudAI Agent specialist agents.

Each tool is declared with the LangChain @tool decorator.  Only signatures
and docstrings are defined here — actual implementation will connect with
the sandboxed execution environment in F4.

Tool registries group tools per agent for binding to the LLM at invocation
time.

Architecture decision (F3 gate):
    ``search_boe`` is the ONLY interface agents use for RAG queries.
    Internally it delegates to ``LegalRetriever`` (rag/retriever.py)
    which handles embedding, hybrid search, reranking, and citation
    formatting.  Agents NEVER call LegalRetriever directly — all RAG
    access goes through this tool so it is visible in tool_results
    and audit logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool  # noqa: TC002 — runtime decorator

if TYPE_CHECKING:
    from fraudai.rag.retriever import LegalRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Harvey Specter — Transaction Fraud Detection tools
# ---------------------------------------------------------------------------


@tool
def analyze_transactions(data_path: str, file_format: str = "csv") -> dict[str, Any]:
    """Analyse a transaction dataset for statistical anomalies.

    Reads the file at *data_path* (CSV or JSON), computes distribution
    summaries, Z-scores, and flags transactions that deviate significantly
    from normal patterns.

    Args:
        data_path: Path to the transaction file inside the sandbox.
        file_format: File format — "csv" or "json".

    Returns:
        Dict with keys: summary (distribution stats), anomalies (list of
        flagged rows with Z-scores), total_transactions, anomaly_rate.
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


@tool
def detect_patterns(
    data_path: str,
    algorithms: list[str] | None = None,
) -> dict[str, Any]:
    """Run ML pattern detection on transactions.

    Applies Isolation Forest and DBSCAN (or user-specified algorithms) to
    identify anomalous clusters and outlier groups in the transaction data.

    Args:
        data_path: Path to the transaction file inside the sandbox.
        algorithms: Algorithms to apply.  Defaults to
            ``["isolation_forest", "dbscan"]``.

    Returns:
        Dict with keys: clusters (list of cluster descriptions),
        outliers (list of outlier transactions), model_params,
        silhouette_score.
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


@tool
def risk_scoring(data_path: str) -> dict[str, Any]:
    """Calculate risk scores (0-100) per transaction and account.

    Aggregates statistical and ML signals into a composite risk score.
    Higher scores indicate greater fraud likelihood.

    Args:
        data_path: Path to the transaction file inside the sandbox.

    Returns:
        Dict with keys: transaction_scores (list of {id, score, factors}),
        account_scores (list of {account_id, score, top_factors}),
        high_risk_count, score_distribution.
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


@tool
def generate_rules(patterns: dict[str, Any]) -> dict[str, Any]:
    """Generate detection rules from detected patterns.

    Translates ML-detected patterns into deterministic rules expressed in
    YAML/JSON, ready for deployment in production rule engines.

    Args:
        patterns: Output from ``detect_patterns`` or analyst-provided
            pattern descriptions.

    Returns:
        Dict with keys: rules (list of rule objects with conditions and
        actions), format ("yaml" | "json"), coverage_estimate,
        false_positive_estimate.
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


# ---------------------------------------------------------------------------
# Louis Litt — AML / KYC / Compliance tools
# ---------------------------------------------------------------------------


def create_search_boe(retriever: LegalRetriever) -> Any:
    """Factory that creates a search_boe tool bound to a LegalRetriever instance.

    The returned tool is an async LangChain ``@tool`` function with the
    retriever captured via closure.  This avoids global state while keeping
    the tool visible in ``tool_results`` and audit logs.

    Args:
        retriever: A fully initialised ``LegalRetriever`` instance.

    Returns:
        An async LangChain tool function ``search_boe``.
    """

    @tool
    async def search_boe(
        query: str,
        k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search BOE and EU legislation via RAG.

        Performs hybrid search (dense + BM25) against the BOE legislation
        vector store.  Returns articles with exact citations and metadata.

        Args:
            query: Natural-language query describing the regulatory question.
            k: Maximum number of results to return.
            filters: Optional metadata filters (e.g. ``{"materia_codigo":
                "derecho financiero"}``, ``{"rango": "Ley"}``).

        Returns:
            List of dicts, each with keys: texto_relevante, boe_id,
            norma_titulo, articulo, score, fecha_publicacion,
            estado_consolidacion, collection.
        """
        # The query is user text: DEBUG only, so INFO logs never carry it.
        logger.debug("search_boe called: query=%r, k=%d, filters=%s", query, k, filters)
        results = await retriever.retrieve(query, k=k, filters=filters)
        citations = retriever.format_citations(results)
        logger.info("search_boe returning %d citations", len(citations))
        return citations

    return search_boe


@tool
def search_boe(
    query: str,
    k: int = 20,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search BOE and EU legislation via RAG.

    Performs hybrid search (dense + BM25) against the BOE legislation
    vector store.  Returns articles with exact citations and metadata.

    **Stub**: This fallback is used when no ``LegalRetriever`` is available.
    Use ``create_search_boe()`` or ``ToolRegistry`` to get a working version.

    Args:
        query: Natural-language query describing the regulatory question.
        k: Maximum number of results to return.
        filters: Optional metadata filters (e.g. ``{"materia_codigo":
            "derecho financiero"}``, ``{"rango": "Ley"}``).

    Returns:
        List of dicts, each with keys: text (article content), boe_id,
        norma_titulo, articulo, seccion, rango, score.
    """
    return [
        {
            "status": "unavailable",
            "message": "RAG not connected. Run: python -m fraudai.ingestion --mode full",
        }
    ]


@tool
def generate_sar_report(
    case_description: str,
    subject_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a SAR/STR draft report in SEPBLAC format.

    Produces a structured Suspicious Activity Report following the format
    required by SEPBLAC (Servicio Ejecutivo de la Comision de Prevencion
    del Blanqueo de Capitales).

    Args:
        case_description: Free-text narrative of the suspicious activity.
        subject_data: Optional dict with subject details (name, id_number,
            account_numbers, addresses, etc.).

    Returns:
        Dict with keys: report (structured SEPBLAC report as dict),
        missing_fields (list of fields that need human completion),
        risk_indicators (list of FATF red flags matched),
        regulatory_basis (list of cited articles).
    """
    return {
        "status": "unavailable",
        "message": "Report generation not connected. Deploy with ./scripts/deploy.sh",
    }


@tool
def compliance_checklist(regulation: str) -> dict[str, Any]:
    """Generate a compliance checklist for the specified regulation.

    Builds an actionable checklist of requirements for the given regulatory
    framework, with status tracking support.

    Args:
        regulation: Target regulation — one of "AML" (Ley 10/2010 +
            EU AMLR), "PSD2" (RDL 19/2018), "RGPD" (LOPDGDD + GDPR),
            "AI_ACT" (EU AI Act 2024/1689).

    Returns:
        Dict with keys: checklist (list of {requirement, article,
        description, status}), total_items, regulation_full_name,
        last_updated.
    """
    return {
        "status": "unavailable",
        "message": "Checklist generation not connected. Deploy with ./scripts/deploy.sh",
    }


# ---------------------------------------------------------------------------
# Jessica Pearson — Fraud Intelligence tools
# ---------------------------------------------------------------------------


@tool
def graph_analysis(data_path: str) -> dict[str, Any]:
    """Build and analyse a transaction graph.

    Constructs a directed graph from transaction data using NetworkX.
    Runs community detection (Louvain), centrality analysis (degree,
    betweenness, eigenvector), and identifies bridge nodes and anomalous
    subgraphs.

    Args:
        data_path: Path to the transaction file inside the sandbox.

    Returns:
        Dict with keys: communities (list of community descriptions),
        key_nodes (list of {node_id, centrality_scores, role}),
        bridges (list of bridge node IDs), graph_stats (nodes, edges,
        density, modularity), visualization_data (D3.js-compatible JSON).
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


# ---------------------------------------------------------------------------
# Mike Ross — AI Red Teaming tools
# ---------------------------------------------------------------------------


@tool
def adversarial_evasion(
    target_endpoint: str,
    attack_type: str = "sign_noise",
) -> dict[str, Any]:
    """Probe a model endpoint's robustness to random input perturbations.

    Black-box check: adds random noise to sample inputs and measures how
    often the prediction changes. It uses no gradients, so it is not an
    implementation of FGSM, PGD, C&W or DeepFool.

    Args:
        target_endpoint: Public http(s) URL of the model API endpoint.
            Private, loopback and link-local addresses are rejected.
        attack_type: Perturbation — "sign_noise", "iterative_sign_noise",
            "gaussian_noise_small" or "gaussian_noise".

    Returns:
        Dict with keys: evasion_rate (float), original_predictions (list),
        adversarial_predictions (list), perturbation_stats,
        severity (Critical/High/Medium/Low), mitigations (list).
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


@tool
def prompt_injection_suite(
    target_endpoint: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Test an LLM endpoint for prompt injection vulnerabilities.

    Executes a curated suite of prompt injection, jailbreaking, and data
    exfiltration payloads against the target LLM endpoint.

    Args:
        target_endpoint: URL of the LLM API endpoint to test.
        system_prompt: Optional known system prompt for white-box testing.

    Returns:
        Dict with keys: vulnerabilities (list of {payload, response,
        severity, category}), success_rate, total_payloads_tested,
        categories_tested (list), mitigations (list).
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


# ---------------------------------------------------------------------------
# Rachel Zane — Data Engineering tools
# ---------------------------------------------------------------------------


@tool
def generate_pipeline(
    requirements: str,
    output_format: str = "python",
) -> dict[str, Any]:
    """Generate a Python ETL pipeline for fraud detection.

    Produces production-ready pipeline code based on the provided
    requirements.  Includes data validation, transformation, and output
    stages with full type hints and error handling.

    Args:
        requirements: Natural-language description of the pipeline
            requirements (data sources, transformations, output format).
        output_format: Output format — "python" (default) or "yaml"
            (for pipeline definition files).

    Returns:
        Dict with keys: code (str — generated Python source), stages
        (list of pipeline stage descriptions), dependencies (list of
        required packages), validation_rules (list of included checks).
    """
    return {
        "status": "unavailable",
        "message": "Code generation not connected. Deploy with ./scripts/deploy.sh",
    }


@tool
def data_quality_check(data_path: str) -> dict[str, Any]:
    """Run data quality validation on a dataset.

    Analyses completeness, validity, consistency, and distribution
    characteristics of the dataset.  Flags issues that could compromise
    downstream fraud analysis.

    Args:
        data_path: Path to the data file inside the sandbox.

    Returns:
        Dict with keys: completeness (per-column null rates),
        validity (schema violations, type mismatches), consistency
        (duplicate rows, referential integrity), distributions
        (per-column summary stats), issues (prioritised list of
        problems), overall_quality_score (0-100).
    """
    return {
        "status": "unavailable",
        "message": "Sandbox not connected. Deploy with ./scripts/deploy.sh",
    }


# ---------------------------------------------------------------------------
# Tool registries per agent
# ---------------------------------------------------------------------------

HARVEY_TOOLS: list[Any] = [
    analyze_transactions,
    detect_patterns,
    risk_scoring,
    generate_rules,
    search_boe,
]

LOUIS_TOOLS: list[Any] = [
    search_boe,
    generate_sar_report,
    compliance_checklist,
]

JESSICA_TOOLS: list[Any] = [
    graph_analysis,
    search_boe,
]

MIKE_TOOLS: list[Any] = [
    adversarial_evasion,
    prompt_injection_suite,
    search_boe,
]

RACHEL_TOOLS: list[Any] = [
    generate_pipeline,
    data_quality_check,
    search_boe,
]

AGENT_TOOLS: dict[str, list[Any]] = {
    "harvey": HARVEY_TOOLS,
    "louis": LOUIS_TOOLS,
    "jessica": JESSICA_TOOLS,
    "mike": MIKE_TOOLS,
    "rachel": RACHEL_TOOLS,
}
