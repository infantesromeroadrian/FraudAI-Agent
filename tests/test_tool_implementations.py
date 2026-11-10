"""Tests for ToolImplementations -- all sandbox calls are mocked."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from fraudai.core.session_context import bind_session
from fraudai.tools.implementations import ToolImplementations
from fraudai.tools.sandbox import SandboxResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sandbox_result(
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    output_json: dict | None = None,
    timed_out: bool = False,
    duration: float = 1.0,
) -> SandboxResult:
    """Build a SandboxResult, optionally with a result.json in output_files."""
    output_files: dict[str, bytes] = {}
    if output_json is not None:
        output_files["result.json"] = json.dumps(output_json).encode()
    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        output_files=output_files,
        duration_seconds=duration,
        timed_out=timed_out,
    )


def _make_csv_bytes(rows: list[list[str]], header: list[str] | None = None) -> bytes:
    """Build minimal CSV content as bytes."""
    lines = []
    if header:
        lines.append(",".join(header))
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines).encode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sandbox() -> MagicMock:
    """Return a mocked SandboxEngine with async execute_python."""
    sandbox = MagicMock()
    sandbox.execute_python = AsyncMock()
    sandbox._client = MagicMock()
    sandbox._image = "fraudai-sandbox:latest"
    sandbox._timeout = 600
    sandbox._mem_limit = "8g"
    sandbox._cpu_count = 4
    sandbox._network_disabled = True
    return sandbox


@pytest.fixture()
def mock_sandbox_networked() -> MagicMock:
    """Return a mocked SandboxEngine with network enabled."""
    sandbox = MagicMock()
    sandbox.execute_python = AsyncMock()
    sandbox._client = MagicMock()
    sandbox._image = "fraudai-sandbox:latest"
    sandbox._timeout = 600
    sandbox._mem_limit = "8g"
    sandbox._cpu_count = 4
    sandbox._network_disabled = False
    return sandbox


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every test hostname to a public address (no real DNS in tests)."""
    monkeypatch.setattr(
        ToolImplementations, "_resolve_host", staticmethod(lambda _host: ["93.184.216.34"])
    )


_ACTIVE_SESSION = "44444444-4444-4444-8444-444444444444"


@pytest.fixture(autouse=True)
def _session_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test inside a valid session whose upload folder is ``tmp_path``.

    Tools only read files of the active session (M6); the tests keep their
    fixtures in ``tmp_path`` by making it the session's upload folder. Tests
    that need other bindings (none, or a different folder) use their own scope.
    """
    monkeypatch.setattr("fraudai.tools.implementations.upload_dir", lambda _sid: tmp_path)
    with bind_session(_ACTIVE_SESSION):
        yield


@pytest.fixture()
def tool_impls(mock_sandbox: MagicMock, mock_sandbox_networked: MagicMock) -> ToolImplementations:
    """Return a ToolImplementations with mocked sandboxes."""
    return ToolImplementations(
        sandbox=mock_sandbox,
        sandbox_networked=mock_sandbox_networked,
    )


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write a sample CSV and return its path."""
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_bytes(
        _make_csv_bytes(
            header=["id", "amount", "account_id", "source", "target"],
            rows=[
                ["1", "100.0", "A001", "Alice", "Bob"],
                ["2", "250.5", "A002", "Bob", "Carol"],
                ["3", "50.0", "A001", "Alice", "Dave"],
                ["4", "9999.9", "A003", "Eve", "Alice"],
                ["5", "75.0", "A002", "Carol", "Eve"],
            ],
        )
    )
    return csv_path


# ---------------------------------------------------------------------------
# Tests: analyze_transactions (Harvey)
# ---------------------------------------------------------------------------


class TestAnalyzeTransactions:
    async def test_success(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """analyze_transactions should return parsed sandbox output on success."""
        expected = {
            "row_count": 5,
            "columns": ["id", "amount", "account_id"],
            "anomalies": [],
            "anomaly_rate": 0.0,
        }
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json=expected,
        )

        result = await tool_impls.analyze_transactions(str(sample_csv))

        assert result["status"] == "success"
        assert result["row_count"] == 5
        mock_sandbox.execute_python.assert_called_once()
        call_kwargs = mock_sandbox.execute_python.call_args
        assert "data.csv" in call_kwargs.kwargs["input_files"]
        assert call_kwargs.kwargs["timeout"] == 120

    async def test_json_format(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        tmp_path: Path,
    ) -> None:
        """analyze_transactions with file_format='json' should use data.json as input."""
        json_path = tmp_path / "data.json"
        json_path.write_text('[{"a": 1}]')
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"row_count": 1},
        )

        result = await tool_impls.analyze_transactions(str(json_path), file_format="json")

        assert result["status"] == "success"
        call_kwargs = mock_sandbox.execute_python.call_args
        assert "data.json" in call_kwargs.kwargs["input_files"]

    async def test_file_not_found(self, tool_impls: ToolImplementations) -> None:
        """analyze_transactions should return error dict for missing file."""
        result = await tool_impls.analyze_transactions("nonexistent_file.csv")

        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()

    async def test_generated_code_contains_zscore(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """The generated code should contain Z-score anomaly detection logic."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"row_count": 5},
        )

        await tool_impls.analyze_transactions(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "z > 3" in code or "z_score" in code.lower() or "Z-score" in code


# ---------------------------------------------------------------------------
# Tests: detect_patterns (Harvey)
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    async def test_default_algorithms(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """detect_patterns with no algorithms should default to IF + DBSCAN."""
        expected = {
            "clusters": [],
            "outliers": [{"algorithm": "isolation_forest", "count": 2}],
            "algorithms_applied": ["isolation_forest", "dbscan"],
        }
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json=expected,
        )

        result = await tool_impls.detect_patterns(str(sample_csv))

        assert result["status"] == "success"
        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "isolation_forest" in code
        assert "dbscan" in code

    async def test_custom_algorithms(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """detect_patterns should pass custom algorithm list to generated code."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"algorithms_applied": ["isolation_forest"]},
        )

        await tool_impls.detect_patterns(str(sample_csv), algorithms=["isolation_forest"])

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "'isolation_forest'" in code

    async def test_code_uses_isolation_forest(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should import and use IsolationForest."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"outliers": []},
        )

        await tool_impls.detect_patterns(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "IsolationForest" in code

    async def test_code_uses_dbscan(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should import and use DBSCAN."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"clusters": []},
        )

        await tool_impls.detect_patterns(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "DBSCAN" in code


# ---------------------------------------------------------------------------
# Tests: risk_scoring (Harvey)
# ---------------------------------------------------------------------------


class TestRiskScoring:
    async def test_success(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """risk_scoring should return parsed result on success."""
        expected = {
            "transaction_scores": [{"index": 0, "score": 42.0}],
            "high_risk_count": 1,
        }
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json=expected,
        )

        result = await tool_impls.risk_scoring(str(sample_csv))

        assert result["status"] == "success"
        assert result["high_risk_count"] == 1


# ---------------------------------------------------------------------------
# Tests: generate_rules (Harvey -- no sandbox)
# ---------------------------------------------------------------------------


class TestGenerateRules:
    async def test_generates_rules_from_outliers(
        self,
        tool_impls: ToolImplementations,
    ) -> None:
        """generate_rules should create rules from outlier patterns."""
        patterns = {
            "outliers": [
                {"algorithm": "isolation_forest", "count": 15},
                {"algorithm": "dbscan", "count": 5},
            ],
            "clusters": [],
        }

        result = await tool_impls.generate_rules(patterns)

        assert result["status"] == "success"
        assert len(result["rules"]) == 2
        assert result["rules"][0]["algorithm_source"] == "isolation_forest"
        assert result["rules"][0]["severity"] == "high"  # count > 10
        assert result["rules"][1]["severity"] == "medium"  # count <= 10

    async def test_generates_rules_from_clusters(
        self,
        tool_impls: ToolImplementations,
    ) -> None:
        """generate_rules should create rules from cluster patterns."""
        patterns = {
            "outliers": [],
            "clusters": [{"cluster_id": 0, "size": 5}],
        }

        result = await tool_impls.generate_rules(patterns)

        assert result["status"] == "success"
        assert len(result["rules"]) == 1
        assert result["format"] == "json"

    async def test_empty_patterns(self, tool_impls: ToolImplementations) -> None:
        """generate_rules with empty patterns should return empty rules list."""
        result = await tool_impls.generate_rules({"outliers": [], "clusters": []})

        assert result["status"] == "success"
        assert result["rules"] == []


# ---------------------------------------------------------------------------
# Tests: generate_sar_report (Louis -- no sandbox)
# ---------------------------------------------------------------------------


class TestGenerateSarReport:
    async def test_returns_report_structure(
        self,
        tool_impls: ToolImplementations,
    ) -> None:
        """generate_sar_report should return a complete SEPBLAC report template."""
        result = await tool_impls.generate_sar_report(
            case_description="Suspicious wire transfers totalling EUR 500,000",
            subject_data={"name": "John Doe", "id_number": "12345678A"},
        )

        assert result["status"] == "success"
        assert "report" in result
        assert result["report"]["header"]["format"] == "SEPBLAC"
        assert "missing_fields" in result
        assert "risk_indicators" in result
        assert "regulatory_basis" in result

    async def test_missing_fields_detected(
        self,
        tool_impls: ToolImplementations,
    ) -> None:
        """generate_sar_report should detect missing required SEPBLAC fields."""
        result = await tool_impls.generate_sar_report(
            case_description="Test case",
            subject_data={"name": "Jane"},
        )

        assert "id_number" in result["missing_fields"]
        assert "nationality" in result["missing_fields"]
        # name is provided, should NOT be in missing
        assert "name" not in result["missing_fields"]

    async def test_no_subject_data(self, tool_impls: ToolImplementations) -> None:
        """generate_sar_report with no subject_data should mark all fields missing."""
        result = await tool_impls.generate_sar_report(case_description="Test")

        assert len(result["missing_fields"]) == 7  # All required fields

    async def test_regulatory_basis_contains_ley_10_2010(
        self,
        tool_impls: ToolImplementations,
    ) -> None:
        """SAR report should cite Ley 10/2010."""
        result = await tool_impls.generate_sar_report(case_description="Test")

        basis_text = " ".join(result["regulatory_basis"])
        assert "Ley 10/2010" in basis_text


# ---------------------------------------------------------------------------
# Tests: compliance_checklist (Louis -- no sandbox)
# ---------------------------------------------------------------------------


class TestComplianceChecklist:
    async def test_aml_checklist(self, tool_impls: ToolImplementations) -> None:
        """compliance_checklist('AML') should return AML-specific items."""
        result = await tool_impls.compliance_checklist("AML")

        assert result["status"] == "success"
        assert result["total_items"] > 0
        assert "Ley 10/2010" in result["regulation_full_name"]
        requirements = [item["requirement"] for item in result["checklist"]]
        assert "Customer Due Diligence (CDD)" in requirements

    async def test_ai_act_checklist(self, tool_impls: ToolImplementations) -> None:
        """compliance_checklist('AI_ACT') should return AI Act items."""
        result = await tool_impls.compliance_checklist("AI_ACT")

        assert result["status"] == "success"
        assert "2024/1689" in result["regulation_full_name"]

    async def test_unknown_regulation(self, tool_impls: ToolImplementations) -> None:
        """compliance_checklist with unknown regulation should return error."""
        result = await tool_impls.compliance_checklist("UNKNOWN_REG")

        assert result["status"] == "failed"
        assert "Unknown regulation" in result["error"]

    async def test_case_insensitive(self, tool_impls: ToolImplementations) -> None:
        """compliance_checklist should handle case variations."""
        result = await tool_impls.compliance_checklist("psd2")

        assert result["status"] == "success"
        assert result["total_items"] > 0


# ---------------------------------------------------------------------------
# Tests: graph_analysis (Jessica)
# ---------------------------------------------------------------------------


class TestGraphAnalysis:
    async def test_success(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """graph_analysis should return parsed NetworkX results."""
        expected = {
            "communities": [{"community_id": 0, "size": 3}],
            "key_nodes": [{"node_id": "Alice", "role": "hub"}],
            "bridges": ["Bob"],
            "graph_stats": {"nodes": 5, "edges": 5, "density": 0.25},
        }
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json=expected,
        )

        result = await tool_impls.graph_analysis(str(sample_csv))

        assert result["status"] == "success"
        assert result["graph_stats"]["nodes"] == 5

    async def test_code_uses_networkx(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should use NetworkX for graph construction."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"communities": []},
        )

        await tool_impls.graph_analysis(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "import networkx" in code
        assert "nx.DiGraph" in code

    async def test_code_includes_community_detection(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should include Louvain community detection."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"communities": []},
        )

        await tool_impls.graph_analysis(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "louvain_communities" in code

    async def test_code_includes_centrality(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should compute centrality metrics."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"key_nodes": []},
        )

        await tool_impls.graph_analysis(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "degree_centrality" in code
        assert "betweenness_centrality" in code


# ---------------------------------------------------------------------------
# Tests: adversarial_evasion (Mike -- networked sandbox)
# ---------------------------------------------------------------------------


class TestAdversarialEvasion:
    async def test_uses_networked_sandbox(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """adversarial_evasion should use the networked sandbox, not the default."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={"evasion_rate": 0.3, "severity": "Medium"},
        )

        result = await tool_impls.adversarial_evasion(
            target_endpoint="http://model.local/predict",
            attack_type="sign_noise",
        )

        assert result["status"] == "success"
        mock_sandbox_networked.execute_python.assert_called_once()

    async def test_attack_type_in_code(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """The attack_type should appear in the generated code."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={"evasion_rate": 0.0},
        )

        await tool_impls.adversarial_evasion(
            target_endpoint="http://test/predict",
            attack_type="iterative_sign_noise",
        )

        code = mock_sandbox_networked.execute_python.call_args.kwargs["code"]
        assert "iterative_sign_noise" in code

    async def test_endpoint_in_code(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """The target endpoint should appear in the generated code."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={"evasion_rate": 0.0},
        )

        await tool_impls.adversarial_evasion(
            target_endpoint="http://target.example.com/api/v1/predict",
        )

        code = mock_sandbox_networked.execute_python.call_args.kwargs["code"]
        assert "http://target.example.com/api/v1/predict" in code


# ---------------------------------------------------------------------------
# Tests: prompt_injection_suite (Mike -- networked sandbox)
# ---------------------------------------------------------------------------


class TestPromptInjectionSuite:
    async def test_uses_networked_sandbox(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """prompt_injection_suite should use the networked sandbox."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={
                "vulnerabilities": [],
                "success_rate": 0.0,
                "total_payloads_tested": 8,
            },
        )

        result = await tool_impls.prompt_injection_suite(
            target_endpoint="http://llm.local/chat",
        )

        assert result["status"] == "success"
        mock_sandbox_networked.execute_python.assert_called_once()

    async def test_system_prompt_in_code(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """When system_prompt is provided, it should appear in the generated code."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={"vulnerabilities": []},
        )

        await tool_impls.prompt_injection_suite(
            target_endpoint="http://llm.local/chat",
            system_prompt="You are a helpful banking assistant.",
        )

        code = mock_sandbox_networked.execute_python.call_args.kwargs["code"]
        assert "You are a helpful banking assistant." in code

    async def test_no_system_prompt_sets_none(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """When no system_prompt is provided, code should set SYSTEM_PROMPT to None."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={"vulnerabilities": []},
        )

        await tool_impls.prompt_injection_suite(
            target_endpoint="http://llm.local/chat",
        )

        code = mock_sandbox_networked.execute_python.call_args.kwargs["code"]
        assert "SYSTEM_PROMPT = None" in code

    async def test_code_includes_payload_categories(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """Generated code should include diverse injection payload categories."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            output_json={"vulnerabilities": []},
        )

        await tool_impls.prompt_injection_suite(
            target_endpoint="http://llm.local/chat",
        )

        code = mock_sandbox_networked.execute_python.call_args.kwargs["code"]
        assert "direct_injection" in code
        assert "jailbreak" in code
        assert "data_exfiltration" in code


# ---------------------------------------------------------------------------
# Tests: generate_pipeline (Rachel -- no sandbox)
# ---------------------------------------------------------------------------


class TestGeneratePipeline:
    async def test_returns_python_code(self, tool_impls: ToolImplementations) -> None:
        """generate_pipeline should return Python pipeline code by default."""
        result = await tool_impls.generate_pipeline(
            requirements="Load CSV, clean nulls, output Parquet",
        )

        assert result["status"] == "success"
        assert "code" in result
        assert "def extract" in result["code"]
        assert "def transform" in result["code"]
        assert len(result["stages"]) > 0

    async def test_returns_yaml(self, tool_impls: ToolImplementations) -> None:
        """generate_pipeline with output_format='yaml' should return YAML."""
        result = await tool_impls.generate_pipeline(
            requirements="Simple ETL",
            output_format="yaml",
        )

        assert result["status"] == "success"
        assert "pipeline:" in result["code"]
        assert result["output_format"] == "yaml"

    async def test_includes_dependencies(self, tool_impls: ToolImplementations) -> None:
        """generate_pipeline should list required dependencies."""
        result = await tool_impls.generate_pipeline(requirements="Any pipeline")

        assert "pandas" in result["dependencies"]


# ---------------------------------------------------------------------------
# Tests: data_quality_check (Rachel)
# ---------------------------------------------------------------------------


class TestDataQualityCheck:
    async def test_success(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """data_quality_check should return quality analysis results."""
        expected = {
            "completeness": {"id": 0.0, "amount": 0.0},
            "validity": {},
            "consistency": {"duplicate_rows": 0},
            "distributions": {"amount": {"mean": 495.1}},
            "issues": [],
            "overall_quality_score": 100,
        }
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json=expected,
        )

        result = await tool_impls.data_quality_check(str(sample_csv))

        assert result["status"] == "success"
        assert result["overall_quality_score"] == 100

    async def test_code_checks_nulls(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should check for null rates."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"overall_quality_score": 80},
        )

        await tool_impls.data_quality_check(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "isnull" in code

    async def test_code_checks_duplicates(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should check for duplicate rows."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"consistency": {"duplicate_rows": 0}},
        )

        await tool_impls.data_quality_check(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "duplicated" in code

    async def test_code_checks_distributions(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Generated code should compute distribution statistics."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            output_json={"distributions": {}},
        )

        await tool_impls.data_quality_check(str(sample_csv))

        code = mock_sandbox.execute_python.call_args.kwargs["code"]
        assert "skew" in code
        assert "kurtosis" in code


# ---------------------------------------------------------------------------
# Tests: Sandbox failure handling
# ---------------------------------------------------------------------------


class TestSandboxFailureHandling:
    async def test_sandbox_timeout(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """When sandbox times out, should return error dict."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            timed_out=True,
            exit_code=-1,
            stderr="execution timed out",
        )

        result = await tool_impls.analyze_transactions(str(sample_csv))

        assert result["status"] == "failed"
        assert "timed out" in result["error"].lower()

    async def test_sandbox_nonzero_exit(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """When sandbox returns non-zero exit code, should return error dict."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            exit_code=1,
            stderr="ModuleNotFoundError: No module named 'foo'",
        )

        result = await tool_impls.analyze_transactions(str(sample_csv))

        assert result["status"] == "failed"
        assert "exit code 1" in result["error"]

    async def test_sandbox_no_result_json(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """When sandbox produces no result.json, should return error dict."""
        mock_sandbox.execute_python.return_value = SandboxResult(
            exit_code=0,
            stdout="done",
            stderr="",
            output_files={},  # No result.json
            duration_seconds=1.0,
            timed_out=False,
        )

        result = await tool_impls.analyze_transactions(str(sample_csv))

        assert result["status"] == "failed"
        assert "no result.json" in result["error"].lower()

    async def test_sandbox_invalid_json(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """When sandbox produces invalid JSON, should return error dict."""
        mock_sandbox.execute_python.return_value = SandboxResult(
            exit_code=0,
            stdout="",
            stderr="",
            output_files={"result.json": b"not valid json{{{"},
            duration_seconds=1.0,
            timed_out=False,
        )

        result = await tool_impls.analyze_transactions(str(sample_csv))

        assert result["status"] == "failed"
        assert "parse" in result["error"].lower()

    async def test_failure_on_detect_patterns(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Sandbox failure handling should work for all sandbox-backed tools."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            exit_code=2,
            stderr="MemoryError",
        )

        result = await tool_impls.detect_patterns(str(sample_csv))

        assert result["status"] == "failed"

    async def test_failure_on_graph_analysis(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """graph_analysis should handle sandbox failure gracefully."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            timed_out=True,
        )

        result = await tool_impls.graph_analysis(str(sample_csv))

        assert result["status"] == "failed"

    async def test_failure_on_risk_scoring(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """risk_scoring should handle sandbox failure gracefully."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            exit_code=137,
            stderr="OOMKilled",
        )

        result = await tool_impls.risk_scoring(str(sample_csv))

        assert result["status"] == "failed"

    async def test_failure_on_data_quality_check(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox: MagicMock,
        sample_csv: Path,
    ) -> None:
        """data_quality_check should handle sandbox failure gracefully."""
        mock_sandbox.execute_python.return_value = _make_sandbox_result(
            exit_code=1,
            stderr="TypeError: cannot parse column",
        )

        result = await tool_impls.data_quality_check(str(sample_csv))

        assert result["status"] == "failed"

    async def test_networked_sandbox_failure(
        self,
        tool_impls: ToolImplementations,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """Mike's tools should handle networked sandbox failures."""
        mock_sandbox_networked.execute_python.return_value = _make_sandbox_result(
            exit_code=1,
            stderr="ConnectionRefusedError",
        )

        result = await tool_impls.adversarial_evasion(
            target_endpoint="http://unreachable.local/predict",
        )

        assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Tests: ToolImplementations constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_networked_engine_defaults_to_the_same_engine(
        self,
        mock_sandbox: MagicMock,
    ) -> None:
        """No second engine: the engine grants network only for a validated EgressTarget."""
        impls = ToolImplementations(sandbox=mock_sandbox)

        assert impls._sandbox_networked is mock_sandbox

    def test_sandbox_is_required_unless_explicitly_disabled(self) -> None:
        with pytest.raises(ValueError, match="explicitly disabled"):
            ToolImplementations(sandbox=None)

    def test_uses_provided_networked_sandbox(
        self,
        mock_sandbox: MagicMock,
        mock_sandbox_networked: MagicMock,
    ) -> None:
        """When networked sandbox is provided, should use it directly."""
        impls = ToolImplementations(
            sandbox=mock_sandbox,
            sandbox_networked=mock_sandbox_networked,
        )

        assert impls._sandbox_networked is mock_sandbox_networked


# ---------------------------------------------------------------------------
# Tests: input files are confined to the active session (M6)
# ---------------------------------------------------------------------------


class TestInputFileConfinement:
    """The autouse session scope must not weaken the fail-closed rules."""

    async def test_path_outside_the_session_folder_is_denied(
        self, tool_impls: ToolImplementations, tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / "outside.csv"
        outside.write_text("id\n1\n")

        with pytest.raises(PermissionError):
            await tool_impls.analyze_transactions(str(outside))

    async def test_traversal_out_of_the_session_folder_is_denied(
        self, tool_impls: ToolImplementations, tmp_path: Path
    ) -> None:
        (tmp_path.parent / "outside.csv").write_text("id\n1\n")

        with pytest.raises(PermissionError):
            await tool_impls.risk_scoring("../outside.csv")

    async def test_no_active_session_is_denied_even_for_an_existing_file(
        self, tool_impls: ToolImplementations, sample_csv: Path, mock_sandbox: MagicMock
    ) -> None:
        from fraudai.core import session_context

        token = session_context._active_session_id.set(None)
        try:
            with pytest.raises(PermissionError, match="no active session"):
                await tool_impls.analyze_transactions(str(sample_csv))
        finally:
            session_context._active_session_id.reset(token)

        mock_sandbox.execute_python.assert_not_called()
