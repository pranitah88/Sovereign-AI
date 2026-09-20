"""
Task Router Local Evaluation Dataset & Benchmark.

Tests actual deterministic routing behavior for representative refinery tasks across:
1. RAG
2. CODING
3. VISION
4. DOCUMENT_ANALYSIS
5. GENERAL
6. HYBRID

Zero fabricated metrics — measures actual accuracy.
"""

import pytest
from backend.services.task_router import route_task


ROUTING_EVALUATION_DATASET = [
    # ── 1. RAG Tasks ────────────────────────────────────────────────────────
    {
        "query": "What was the crude throughput of MRPL during FY 2023-24?",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "RAG",
        "expected_model": "gemma3:4b",
    },
    {
        "query": "Explain the design pressure and temperature for the Hydrocracker Unit (HCU).",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "RAG",
        "expected_model": "gemma3:4b",
    },
    {
        "query": "What are the vigilance complaint procedures outlined by MRPL CVC guidelines?",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "RAG",
        "expected_model": "gemma3:4b",
    },

    # ── 2. Coding & Scripting Tasks ─────────────────────────────────────────
    {
        "query": "Write a Python script to calculate Gross Refining Margin (GRM) from crude feed assay.",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "CODING",
        "expected_model": "qwen2.5-coder:3b",
    },
    {
        "query": "Create a python function to compute heat exchanger log mean temperature difference (LMTD).",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "CODING",
        "expected_model": "qwen2.5-coder:3b",
    },
    {
        "query": "Write python code to parse vibration telemetry from pump 11-P-101A and detect anomalies.",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "CODING",
        "expected_model": "qwen2.5-coder:3b",
    },

    # ── 3. Vision & Drawing Inspection Tasks ────────────────────────────────
    {
        "query": "Inspect the attached P&ID drawing and extract all equipment tags.",
        "has_image": True,
        "has_scanned_pdf": False,
        "expected_task_type": "VISION",
        "expected_model": ["gemma3:4b", "qwen2.5vl:3b", "qwen2.5-vl:3b"],
    },
    {
        "query": "Verify equipment pump tag 11-P-101A against this schematic image.",
        "has_image": True,
        "has_scanned_pdf": False,
        "expected_task_type": "VISION",
        "expected_model": ["gemma3:4b", "qwen2.5vl:3b", "qwen2.5-vl:3b"],
    },

    # ── 4. Document Analysis Tasks ──────────────────────────────────────────
    {
        "query": "Analyze this uploaded operational safety manual.",
        "has_image": False,
        "has_scanned_pdf": True,
        "expected_task_type": "DOCUMENT_ANALYSIS",
        "expected_model": "gemma3:4b",
    },

    # ── 5. General Conversational / Navigational Tasks ───────────────────────
    {
        "query": "Hello! What capabilities does the sovereign workbench support?",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "GENERAL",
        "expected_model": "gemma3:4b",
    },
    {
        "query": "What models are currently active in the registry?",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "GENERAL",
        "expected_model": "gemma3:4b",
    },

    # ── 6. Hybrid Tasks ─────────────────────────────────────────────────────
    {
        "query": "From the MRPL annual report, extract the revenue numbers and write Python code to plot the YoY growth rate.",
        "has_image": False,
        "has_scanned_pdf": False,
        "expected_task_type": "HYBRID",
        "expected_model": "qwen2.5-coder:3b",
    },
]


def test_task_router_local_evaluation_dataset():
    """
    Run evaluation across all test items in ROUTING_EVALUATION_DATASET.
    Verifies that model routing accuracy is measured and exceeds 90% without fabrication.
    """
    total = len(ROUTING_EVALUATION_DATASET)
    task_type_matches = 0
    model_matches = 0
    results = []

    for item in ROUTING_EVALUATION_DATASET:
        decision = route_task(
            user_request=item["query"],
            has_image=item["has_image"],
            has_scanned_pdf=item["has_scanned_pdf"],
        )

        task_match = decision["task_type"] == item["expected_task_type"]
        if isinstance(item["expected_model"], (list, tuple, set)):
            model_match = decision["model"] in item["expected_model"]
        else:
            model_match = decision["model"] == item["expected_model"]

        if task_match:
            task_type_matches += 1
        if model_match:
            model_matches += 1

        results.append({
            "query": item["query"][:40] + "...",
            "expected_task": item["expected_task_type"],
            "actual_task": decision["task_type"],
            "task_match": task_match,
            "expected_model": item["expected_model"],
            "actual_model": decision["model"],
            "model_match": model_match,
        })

    task_accuracy = (task_type_matches / total) * 100
    model_accuracy = (model_matches / total) * 100

    print(f"\n--- Model Router Evaluation Results ---")
    print(f"Total Test Cases: {total}")
    print(f"Task Classification Accuracy: {task_type_matches}/{total} ({task_accuracy:.1f}%)")
    print(f"Model Selection Accuracy: {model_matches}/{total} ({model_accuracy:.1f}%)")

    assert task_type_matches == total, f"Routing discrepancies: {[r for r in results if not r['task_match']]}"
    assert model_matches == total, f"Model discrepancies: {[r for r in results if not r['model_match']]}"
