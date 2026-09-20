"""
Unit tests for Coding Agent Bounded Docker Self-Correction Loop.

Verifies:
1. MAX_CORRECTION_ATTEMPTS = 3 bounded execution.
2. Automatic error detection, diagnosis prompt creation, and re-execution.
3. Accurate attempt storage in state.tool_results with required schema:
   {"attempt": X, "code": "...", "exit_code": ..., "stdout": "...", "stderr": "...", "status": ...}
4. Fail-closed security isolation: Docker unavailable fails closed without host fallback.
5. Zero subprocess / host execution requirement.
6. Response formatting in node_reason() for CODING tasks.
"""

import os
import subprocess
import pytest
from unittest.mock import MagicMock, patch

from backend.agent.graph import (
    MAX_CORRECTION_ATTEMPTS,
    node_tool_call,
    node_reason,
    _extract_code_block,
)
from backend.agent.prompts import CODE_CORRECTION_PROMPT, CODE_GENERATION_PROMPT
from backend.agent.state import AgentState


def test_max_correction_attempts_constant():
    """Verify that MAX_CORRECTION_ATTEMPTS is strictly set to 3."""
    assert MAX_CORRECTION_ATTEMPTS == 3


def test_code_correction_prompt_format():
    """Verify CODE_CORRECTION_PROMPT accepts all required diagnostic parameters."""
    prompt = CODE_CORRECTION_PROMPT.format(
        query="write python code to calculate financial status",
        exit_code=1,
        stderr="NameError: name 'max_balance' is not defined",
        stdout="",
        code="print('test')",
    )
    assert "write python code to calculate financial status" in prompt
    assert "NameError: name 'max_balance' is not defined" in prompt
    assert "Exit code:\n1" in prompt
    assert "print('test')" in prompt
    assert "Preserve the user's requested functionality" in prompt
    assert "expenses and debts reduce net balance/income" in prompt


def test_correction_loop_stops_on_first_attempt_success():
    """When attempt 1 succeeds (exit_code=0), loop must immediately stop with 1 attempt."""
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    mock_llm_code = "```python\nprint('Financial status: OK')\n```"
    mock_sandbox_success = {
        "tool": "sandbox_execute",
        "status": "success",
        "result": {
            "status": "success",
            "stdout": "Financial status: OK",
            "stderr": "",
            "exit_code": 0,
            "execution_time_ms": 45,
        },
    }

    with patch("backend.agent.graph._call_llm", return_value=mock_llm_code) as mock_llm:
        with patch("backend.agent.graph.execute_tool", return_value=mock_sandbox_success) as mock_exec:
            out_state = node_tool_call(state)

            assert mock_llm.call_count == 1
            assert mock_exec.call_count == 1
            assert len(out_state.tool_results) == 1
            first_attempt = out_state.tool_results[0]
            assert first_attempt["attempt"] == 1
            assert first_attempt["exit_code"] == 0
            assert first_attempt["status"] == "verified"
            assert "Financial status: OK" in first_attempt["stdout"]


def test_correction_loop_recovers_on_attempt_two():
    """
    When attempt 1 fails (exit_code=1, NameError) and attempt 2 succeeds (exit_code=0),
    the agent must make 2 attempts and record both in state.tool_results.
    """
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    buggy_code = "```python\nprint(max_balance)\n```"
    fixed_code = "```python\nmax_balance = 500\nprint(max_balance)\n```"

    llm_side_effects = [buggy_code, fixed_code]
    sandbox_side_effects = [
        {
            "tool": "sandbox_execute",
            "status": "success",
            "result": {
                "status": "failure",
                "stdout": "",
                "stderr": "NameError: name 'max_balance' is not defined",
                "exit_code": 1,
                "execution_time_ms": 50,
            },
        },
        {
            "tool": "sandbox_execute",
            "status": "success",
            "result": {
                "status": "success",
                "stdout": "500",
                "stderr": "",
                "exit_code": 0,
                "execution_time_ms": 40,
            },
        },
    ]

    with patch("backend.agent.graph._call_llm", side_effect=llm_side_effects) as mock_llm:
        with patch("backend.agent.graph.execute_tool", side_effect=sandbox_side_effects) as mock_exec:
            out_state = node_tool_call(state)

            assert mock_llm.call_count == 2
            assert mock_exec.call_count == 2
            assert len(out_state.tool_results) == 2

            # Attempt 1: Failed
            att1 = out_state.tool_results[0]
            assert att1["attempt"] == 1
            assert att1["exit_code"] == 1
            assert att1["status"] == "failed"
            assert "NameError: name 'max_balance' is not defined" in att1["stderr"]

            # Attempt 2: Verified
            att2 = out_state.tool_results[1]
            assert att2["attempt"] == 2
            assert att2["exit_code"] == 0
            assert att2["status"] == "verified"
            assert att2["stdout"] == "500"


def test_correction_loop_bounded_to_max_three_attempts():
    """
    When all attempts continuously fail, the loop must execute exactly 3 attempts
    and then stop without looping indefinitely.
    """
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    persistent_bug = "```python\nraise RuntimeError('Persistent failure')\n```"
    failure_result = {
        "tool": "sandbox_execute",
        "status": "success",
        "result": {
            "status": "failure",
            "stdout": "",
            "stderr": "RuntimeError: Persistent failure",
            "exit_code": 1,
            "execution_time_ms": 30,
        },
    }

    with patch("backend.agent.graph._call_llm", return_value=persistent_bug) as mock_llm:
        with patch("backend.agent.graph.execute_tool", return_value=failure_result) as mock_exec:
            out_state = node_tool_call(state)

            # Strictly 3 attempts
            assert mock_llm.call_count == 3
            assert mock_exec.call_count == 3
            assert len(out_state.tool_results) == 3

            for idx, att in enumerate(out_state.tool_results, 1):
                assert att["attempt"] == idx
                assert att["exit_code"] == 1
                assert att["status"] == "failed"


def test_docker_unavailable_fails_closed_immediately():
    """
    If Docker daemon is unavailable, the agent must fail closed immediately on attempt 1,
    must NOT attempt further LLM calls, must NOT execute on host, and exit_code must be None.
    """
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    docker_down_result = {
        "tool": "sandbox_execute",
        "status": "error",
        "result": {
            "status": "error",
            "stdout": "",
            "stderr": "Code sandbox is unavailable: Docker daemon is not running.",
            "exit_code": None,
            "execution_time_ms": 0,
        },
    }

    with patch("backend.agent.graph._call_llm", return_value="```python\nprint(1)\n```") as mock_llm:
        with patch("backend.agent.graph.execute_tool", return_value=docker_down_result) as mock_exec:
            out_state = node_tool_call(state)

            # Must stop immediately after 1 attempt
            assert mock_llm.call_count == 1
            assert mock_exec.call_count == 1
            assert len(out_state.tool_results) == 1

            att = out_state.tool_results[0]
            assert att["attempt"] == 1
            assert att["exit_code"] is None
            assert att["status"] == "error"
            assert "Docker daemon is not running" in att["stderr"]


def test_zero_host_subprocess_execution():
    """
    Ensure that under no circumstances is subprocess.run, os.system, or shell execution
    called on the host during code execution or correction.
    """
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    with patch("subprocess.run") as mock_subproc, patch("os.system") as mock_ossys:
        with patch("backend.agent.graph._call_llm", return_value="```python\nprint('hello')\n```"):
            with patch("backend.agent.graph.execute_tool", return_value={
                "tool": "sandbox_execute",
                "status": "success",
                "result": {"status": "success", "stdout": "hello", "stderr": "", "exit_code": 0},
            }):
                node_tool_call(state)

        # Asserts zero host subprocess execution
        assert mock_subproc.call_count == 0
        assert mock_ossys.call_count == 0


def test_node_reason_verified_response_format():
    """Verify that node_reason() formats successful verified execution as required."""
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        tool_results=[
            {
                "tool": "sandbox_execute",
                "attempt": 1,
                "code": "print(x)",
                "exit_code": 1,
                "stdout": "",
                "stderr": "NameError: name 'x' is not defined",
                "status": "failed",
            },
            {
                "tool": "sandbox_execute",
                "attempt": 2,
                "code": "x = 400\nprint(x)",
                "exit_code": 0,
                "stdout": "400",
                "stderr": "",
                "status": "verified",
            },
        ],
    )

    out_state = node_reason(state)
    resp = out_state.response

    assert "**Task Type:** CODING" in resp
    assert "**Model:** qwen2.5-coder:3b" in resp
    assert "**Execution:** VERIFIED" in resp
    assert "**Correction Attempts:** 2" in resp
    assert "x = 400" in resp
    assert "### Execution Result (Docker Sandbox)" in resp
    assert "400" in resp


def test_node_reason_blocked_response_format():
    """Verify that node_reason() formats Docker-unavailable failure cleanly with security notice."""
    state = AgentState(
        query="write python code to calculate financial status",
        task_type="CODING",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_sandbox=True,
        required_tools=["sandbox_execute"],
        tool_results=[
            {
                "tool": "sandbox_execute",
                "attempt": 1,
                "code": "print('hello')",
                "exit_code": None,
                "stdout": "",
                "stderr": "Docker daemon is not running.",
                "status": "error",
            },
        ],
    )

    out_state = node_reason(state)
    resp = out_state.response

    assert "**Task Type:** CODING" in resp
    assert "**Execution:** BLOCKED (Docker Sandbox Unavailable)" in resp
    assert "⚠️ **Sandbox Execution Notice**" in resp
    assert "failed closed" in resp
