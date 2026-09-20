# backend/router.py
"""
Router service for the Sovereign Agentic AI Workbench.

Reads config/model_registry.yaml, asks the primary reasoning model (Gemma)
to classify an incoming request into a task_type, then looks up which
model + tools should handle it -- entirely from the registry (single source
of truth), with modality compatibility enforced at selection time.
Returns structured JSON for n8n to act on.
"""

import json
import yaml
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

REGISTRY_PATH = Path(__file__).parent.parent / "config" / "model_registry.yaml"


def load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


REGISTRY = load_registry()

# The router always uses the primary reasoning model to classify intent.
# Change this if you later add a lighter/faster dedicated classifier model.
ROUTER_MODEL_ID = "gemma3_4b"

VALID_TASK_TYPES = list(REGISTRY["task_type_routing"].keys())

# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="Sovereign AI Router")


class RouteRequest(BaseModel):
    request: str
    has_image: bool = False        # true if an image/photo was attached
    has_scanned_pdf: bool = False  # true if a scanned PDF was attached


class RouteDecision(BaseModel):
    task_type: str
    model: str
    ollama_model_name: str
    model_endpoint: str
    tools: list[str]
    reasoning: str


# --------------------------------------------------------------------------
# Registry helpers
# --------------------------------------------------------------------------

def get_model_entry(model_id: str) -> dict:
    for m in REGISTRY["models"]:
        if m["id"] == model_id:
            return m
    raise ValueError(f"Model id '{model_id}' not found in registry")


# --------------------------------------------------------------------------
# Step 1: classify the request into a task_type using Gemma
# --------------------------------------------------------------------------

def classify_task_type(user_request: str, has_image: bool, has_scanned_pdf: bool) -> dict:
    """
    Calls the router model (Gemma) with a constrained prompt asking it to
    classify the request into one of the known task_types and return JSON.
    """
    router_model = get_model_entry(ROUTER_MODEL_ID)

    task_type_list_str = "\n".join(f"- {t}" for t in VALID_TASK_TYPES)

    system_prompt = f"""You are a task classifier for an industrial AI workbench.
Classify the user's request into EXACTLY ONE of these task_types:

{task_type_list_str}

Context flags:
- has_image: {has_image}
- has_scanned_pdf: {has_scanned_pdf}

Rules:
- If has_image is true and the request is about understanding a photo, drawing,
  or visual content, use "image_or_drawing_understanding".
- If has_scanned_pdf is true and no image understanding is needed, prefer
  "document_summary" or "approval_note" as appropriate (OCR + RAG path).
- Use "calculation" ONLY if the user explicitly wants something computed,
  derived, or verified using math/logic on data THEY provide or reference
  (e.g. "calculate X from this CSV", "verify this value is within limits").
- Use "general_reasoning" for factual lookup questions asking about known
  figures, capacities, specifications, policies, or facts that should be
  retrieved from documents, even if the answer happens to be a number.
  A question asking "what is the capacity/value/rating of X" is a LOOKUP,
  not a calculation, unless it explicitly asks to compute/derive/verify something.
- If the request asks to write/generate code or scripts, use "code_generation".
- Otherwise use "document_summary" if it's about summarizing content.

Examples:
- "What is the refining capacity of MRPL?" -> general_reasoning (factual lookup)
- "Calculate the refinery efficiency from this CSV" -> calculation (explicit compute)
- "Is this pressure reading within the specified limit?" -> calculation (verify/derive)
- "What is the maximum allowed pressure per the SOP?" -> general_reasoning (lookup from docs)

Respond with ONLY valid JSON in this exact shape, nothing else:
{{"task_type": "<one_of_the_types_above>", "reasoning": "<one short sentence>"}}
"""

    payload = {
        "model": router_model["ollama_model_name"],
        "prompt": f"{system_prompt}\n\nUser request: {user_request}",
        "stream": False,
        "format": "json"  # Ollama structured JSON output mode
    }

    try:
        resp = requests.post(router_model["endpoint"], json=payload, timeout=60)
        resp.raise_for_status()
        raw_text = resp.json().get("response", "").strip()
        parsed = json.loads(raw_text)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        # Fallback: default to general_reasoning if classification fails
        return {
            "task_type": "general_reasoning",
            "reasoning": f"Fallback due to classification error: {e}"
        }

    if parsed.get("task_type") not in VALID_TASK_TYPES:
        parsed["task_type"] = "general_reasoning"
        parsed["reasoning"] = parsed.get("reasoning", "") + " (invalid type returned, defaulted)"

    return parsed


# --------------------------------------------------------------------------
# Step 2: select model + tools from the registry, enforcing modality fit
# --------------------------------------------------------------------------

def select_model_and_tools(
    task_type: str,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
) -> tuple[dict, list[str]]:
    """
    Select a model using the registry's preferred tags and priority,
    enforce modality compatibility (vision) when an image is attached,
    and obtain tools directly from the registry's required_tools --
    augmented with ocr_extract when a scanned PDF is attached.
    """

    routing_info = REGISTRY["task_type_routing"].get(task_type)

    if not routing_info:
        task_type = "general_reasoning"
        routing_info = REGISTRY["task_type_routing"][task_type]

    preferred_tags = set(routing_info.get("preferred_tags", []))

    candidates = []
    for model in REGISTRY["models"]:
        if not model.get("enabled", True):
            continue

        model_tags = set(model.get("task_tags", []))

        if preferred_tags & model_tags:
            candidates.append(model)

    if not candidates:
        candidates = [get_model_entry(ROUTER_MODEL_ID)]

    # ----------------------------------------------------------------
    # Enforce modality compatibility: an attached image requires a
    # vision-capable model, regardless of what tag-matching produced.
    # ----------------------------------------------------------------
    if has_image:
        vision_candidates = [
            model for model in candidates
            if "vision" in model.get("modality", [])
        ]

        if vision_candidates:
            candidates = vision_candidates
        else:
            # None of the tag-matched candidates support vision --
            # fall back to searching the whole registry for any
            # enabled vision-capable model instead of failing silently.
            vision_candidates = [
                model for model in REGISTRY["models"]
                if model.get("enabled", True)
                and "vision" in model.get("modality", [])
            ]
            if vision_candidates:
                candidates = vision_candidates
            # If truly no vision model exists in the registry, candidates
            # stays as the original (non-vision) list -- this is a real
            # capability gap that should surface as a routing decision
            # a human would notice, not be silently masked.

    chosen_model = sorted(candidates, key=lambda m: m.get("priority", 99))[0]

    # ----------------------------------------------------------------
    # Tool selection: base tools from the registry, plus force
    # ocr_extract when a scanned PDF is attached.
    # ----------------------------------------------------------------
    if has_scanned_pdf:
        routing_tools = set(routing_info.get("required_tools", []))
        routing_tools.add("ocr_extract")
        tools = list(routing_tools)
    else:
        tools = routing_info.get("required_tools", [])

    return chosen_model, tools


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------

@app.post("/route", response_model=RouteDecision)
def route(req: RouteRequest):
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="Empty request text")

    classification = classify_task_type(req.request, req.has_image, req.has_scanned_pdf)
    task_type = classification["task_type"]

    chosen_model, tools = select_model_and_tools(
        task_type,
        req.has_image,
        req.has_scanned_pdf,
    )

    return RouteDecision(
        task_type=task_type,
        model=chosen_model["id"],
        ollama_model_name=chosen_model["ollama_model_name"],
        model_endpoint=chosen_model["endpoint"],
        tools=tools,
        reasoning=classification.get("reasoning", "")
    )


@app.get("/registry")
def get_registry():
    """Debug endpoint: view the loaded registry as-is."""
    return REGISTRY


@app.get("/health")
def health():
    """Simple health check for n8n / monitoring."""
    return {"status": "ok", "models_loaded": len(REGISTRY["models"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)