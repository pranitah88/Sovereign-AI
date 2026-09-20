"""
Verification test runner for the 5 RAG benchmark queries.
Measures retrieval time, generation time, total response time,
and evaluates grounding and sources.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.agent.graph import run_agent

TEST_CASES = [
    {
        "id": "TEST 1",
        "query": "What is MRPL?",
        "notes": "General overview test",
    },
    {
        "id": "TEST 2",
        "query": "What is the purpose of PFCCU at MRPL?",
        "notes": "Technical domain RAG lookup",
    },
    {
        "id": "TEST 3",
        "query": "What was MRPL's turnover in 2019-20? Give only the exact value stated in the source, including the unit and page number.",
        "notes": "Exact numerical grounding test",
    },
    {
        "id": "TEST 4",
        "query": "Summarize MRPL's financial performance from 2016-17 to 2019-20.",
        "notes": "Multi-year financial grounding test",
    },
    {
        "id": "TEST 5",
        "query": "What is MRPL's confidential internal maintenance budget for 2026?",
        "notes": "Refusal / zero-hallucination test (must state not in knowledge base)",
    },
]


async def run_all_tests():
    print("=" * 80)
    print("STARTING RAG VERIFICATION TESTS")
    print("=" * 80)

    results = []

    for test in TEST_CASES:
        test_id = test["id"]
        query = test["query"]
        print(f"\n[{test_id}] Query: {query}")
        print("-" * 60)

        t0 = time.perf_counter()
        agent_res = await run_agent(
            query=query,
            user_id=1,
            session_id=None,
        )
        total_time_ms = int((time.perf_counter() - t0) * 1000)

        retrieval_ms = agent_res.get("retrieval_ms", 0)
        generation_ms = agent_res.get("execution_ms", 0)
        sources = agent_res.get("sources", [])
        response = agent_res.get("response", "")

        doc_names = list({s.get("source", "") for s in sources if s.get("source")})
        page_numbers = sorted(list({s.get("page") for s in sources if s.get("page") is not None}))

        # Grounding check:
        # For TEST 5, check if the response refuses or states information is not available
        refusal_phrases = [
            "don't have sufficient information",
            "not available",
            "not found",
            "does not contain",
            "insufficient information",
            "confidential",
        ]
        is_grounded = True
        warning_present = "⚠️" in response

        if test_id == "TEST 5":
            is_grounded = any(p in response.lower() for p in refusal_phrases) and ("2026" not in response or "not available" in response.lower() or "don't have" in response.lower())
        else:
            is_grounded = not warning_present

        result_item = {
            "test_id": test_id,
            "query": query,
            "retrieval_time_ms": retrieval_ms,
            "generation_time_ms": generation_ms,
            "total_response_time_ms": total_time_ms,
            "retrieved_document_names": doc_names,
            "retrieved_page_numbers": page_numbers,
            "final_answer": response,
            "is_fully_grounded": is_grounded,
            "has_ungrounded_warning": warning_present,
        }
        results.append(result_item)

        # Save to file incrementally
        out_path = Path("outputs/rag_verification_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Retrieval Time: {retrieval_ms} ms ({retrieval_ms/1000:.2f}s)")
        print(f"Generation Time: {generation_ms} ms ({generation_ms/1000:.2f}s)")
        print(f"Total Time: {total_time_ms} ms ({total_time_ms/1000:.2f}s)")
        print(f"Retrieved Documents: {doc_names}")
        print(f"Retrieved Pages: {page_numbers}")
        print(f"Fully Grounded: {is_grounded}")
        print(f"\nFinal Answer:\n{response}")
        print("=" * 80)

    print(f"\nAll results saved to outputs/rag_verification_results.json")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
