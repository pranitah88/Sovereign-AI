# MRPL Sovereign AI Workbench — Security Architecture & Controls

## 1. Executive Summary & Zero-Trust Architecture

The MRPL Sovereign AI Workbench is an air-gapped, on-premise industrial AI assistance platform designed for sensitive refinery operations (Mangalore Refinery and Petrochemicals Limited). It operates under a strict **Zero-Trust Security Architecture** where:
1. **The LLM is NEVER the security boundary**: No decisions regarding file access, role permissions, tool execution, or system state depend on model self-policing or system prompts alone.
2. **Zero Cloud Telemetry or Inference**: All AI model inference is hosted locally via Ollama. No data or queries ever leave the host machine or refinery intranet.
3. **Fail-Closed Code Execution**: Sandboxed code runs exclusively inside an isolated Docker container with zero network access and restricted resources. If the sandbox environment is not ready, execution immediately halts with a controlled error rather than falling back to host execution.

```
USER QUERY
    ↓
AUTHENTICATION (JWT / Session)
    ↓
RBAC (Admin, Engineer, Reviewer, Viewer)
    ↓
DATA CLASSIFICATION CLEARANCE (PUBLIC, INTERNAL, CONFIDENTIAL, HIGHLY_CONFIDENTIAL)
    ↓
POLICY & AUTHORIZATION ENGINE
    ↓
HYBRID RAG RETRIEVAL (ChromaDB + BM25)
    ↓
PRE-LLM CLEARANCE FILTERING (Drop unauthorized chunks BEFORE model sees them)
    ↓
MODEL ROUTER (Approved Model Registry Only)
    ↓
LOCAL OLLAMA INFERENCE (Gemma 3 4B / Qwen 2.5 Coder 3B)
    ↓
BACKEND TOOL GATEWAY (Parameter Validation + Role Checks)
    ↓
FAIL-CLOSED DOCKER SANDBOX (network_mode="none", read-only root, 256MB RAM)
    ↓
AUDIT LOGGING (SQLite Immutable Event Trail)
    ↓
SANITIZED OUTPUT ("AI-GENERATED — REQUIRES HUMAN REVIEW")
```

---

## 2. Implemented Security Controls

### 1. Local / Air-Gapped Operation
- **Inference**: Handled 100% locally by Ollama on `localhost:11434`.
- **Zero Cloud APIs**: No external LLM keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) are used or required.
- **Embedded Vector Database**: ChromaDB runs locally on SQLite and disk vectors.
- **Zero Runtime Downloads**: Models and Docker sandbox images must be pre-built; no automatic downloads occur during runtime.

### 2. Role-Based Access Control (RBAC)
Enforced purely in backend middleware and API route dependencies:
- `admin`: Full system control, user management, audit logs, model registry configuration.
- `engineer`: Chat, RAG search, document uploads, sandbox coding, report generation.
- `reviewer`: Chat, view documents/results, approve or reject high-risk AI proposals.
- `viewer`: Read-only queries against public/internal docs, no code execution, no administrative routes.
*Unauthorized attempts return HTTP 403 Forbidden with security audit logging.*

### 3. Data Classification
Documents are labeled with four security classifications:
- `PUBLIC` (Accessible to all roles)
- `INTERNAL` (Accessible to Engineer, Reviewer, Admin)
- `CONFIDENTIAL` (Accessible to Engineer, Admin)
- `HIGHLY_CONFIDENTIAL` (Accessible to Admin only)

Metadata includes `classification`, `department`, `owner`, `doc_id`, `version`, and `upload_timestamp`.

### 4. Secure RAG & Pre-LLM Filtering
Chunks are indexed with their parent document's classification. During vector and BM25 hybrid retrieval, chunks that exceed the user's role clearance are filtered out **prior to reranking and prior to prompt generation**. The LLM context never contains unauthorized chunks.

### 5. Approved Model Registry
Local models are governed by `config/model_registry.yaml`. Every task checks that the requested model is explicitly `approved: true` and `enabled: true`. Unapproved models are blocked with an HTTP 400 error.

### 6. Backend Tool Permission Gateway
Tools (`READ_DOCUMENT`, `SEARCH_KB`, `GENERATE_DOCUMENT`, `EXECUTE_CODE`) require backend authorization:
- LLMs request actions via structured JSON.
- Tool Gateway validates parameters, prevents path traversal, and enforces user role permissions before invoking any function.

### 7. Docker Code Execution Sandbox (Fail-Closed)
- **Image**: `mrpl-sandbox:latest` (`python:3.13-slim` base with non-root user `sandboxuser` UID 1000).
- **Isolation**:
  - `network_mode="none"` (zero socket connectivity).
  - `read_only=True` (root filesystem is read-only).
  - Memory limit: `256m`.
  - CPU limit: `1.0` CPU (100,000 quota).
  - Timeout: 30 seconds.
  - Temporary workspace mounted with `noexec` on root.
- **Fail-Closed**: If Docker is unreachable or the image is absent, execution fails immediately. Host execution fallback is strictly prohibited.

### 8. Human-in-the-Loop Approvals for High-Risk Actions
Actions classified as `HIGH` risk (file deletions, system configuration, database modifications) generate a pending proposal in the `action_approvals` table.
- Execution is blocked until a `reviewer` or `admin` explicitly reviews and approves the request.
- The AI is strictly prohibited from approving its own actions.

### 9. Prompt-Injection Defense
- Retrieved knowledge base passages are wrapped with untrusted content delimiters (`<untrusted_document_context>`).
- Regex scanners detect common jailbreaks and instruction-override sequences, logging security warnings.
- Retrieved instructions are never treated as system directives or permitted to trigger tools.

### 10. File Upload Security
- Strict extension whitelist (`.pdf`, `.txt`, `.docx`, `.xlsx`, `.pptx`, `.csv`, `.json`).
- Executable files (`.exe`, `.bat`, `.sh`, `.ps1`, `.cmd`) are rejected with HTTP 400.
- Directory traversal sequences (`../`, `..\\`) are detected, blocked, and logged.
- Maximum upload size capped at 50 MB.

### 11. Comprehensive Audit Logging
All security-sensitive operations are immutably logged to `data/mrpl_sovereign.db`:
- Logins, authentication failures, and authorization rejections.
- Model selections and RAG query retrieval metadata.
- Tool executions and sandboxed code executions.
- Action approval requests and reviewer decisions.
- Admins can query audit records via `/api/audit/logs`.

### 12. Privacy & Chat Isolation
Chat sessions are scoped to user ownership. Users cannot query, view, or append to another user's session. Cross-user access returns HTTP 403 Forbidden.

### 13. Output Security & Watermarking
All generated documents (DOCX, XLSX, PPTX) include a visible warning:
`"AI-GENERATED — REQUIRES HUMAN REVIEW"`
Generated files are restricted to the designated `outputs/` directory.

### 14. Safety & Execution Guards
- Maximum agent iterations capped at 10 to prevent runaway loops.
- Maximum tool invocations per task capped at 15.
- Global request timeout enforced at 60 seconds.

---

## 3. "Nothing Leaves the Premises" Demonstration Procedure

For the SIH live presentation or refinery audit:

1. **System Startup**:
   ```powershell
   # Start local Ollama service
   ollama serve
   # Start backend
   .\venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   # Start frontend
   cd frontend; npm run dev
   ```
2. **Confirm Local Model Availability**:
   Visit `http://localhost:8000/api/status/services` or the UI dashboard to verify Ollama and ChromaDB are healthy.
3. **Physical Air-Gap / Network Disconnect**:
   - Disconnect Wi-Fi and unplug Ethernet cables.
   - Run `ipconfig /all` to demonstrate no external default gateway is reachable.
4. **General Inference Test**:
   - Ask an engineering query: *"Explain the fluid catalytic cracking regeneration loop."*
   - Verify immediate local completion via Gemma 3 4B.
5. **Secure RAG Retrieval Test**:
   - Query refinery operating manuals in the knowledge base.
   - Verify chunks are retrieved from local ChromaDB and cited with page numbers.
6. **Isolated Sandbox Code Execution**:
   - Request Python calculations (e.g., Reynolds number or heat exchanger duty).
   - Verify code runs in `mrpl-sandbox:latest` with network disabled.
7. **Verification Summary**:
   Check the **Air-Gap & Sovereign Security Status** dashboard card to confirm:
   - Local Inference: `PASS`
   - External AI APIs: `BLOCKED / NONE CONFIGURED`
   - Local RAG: `PASS`
   - Internet Dependency: `NONE FOR RUNTIME`

---

## 4. Production Hardening Beyond Prototype

The current implementation is an operational prototype tailored for the Smart India Hackathon (SIH) local demonstration environment. For deployment in a live refinery distributed control system (DCS) or corporate plant network, the following enterprise-grade controls are recommended:

- **Enterprise Directory Integration**: Integrate with corporate Active Directory / LDAP and Kerberos for centralized Single Sign-On (SSO).
- **Multi-Factor Authentication (MFA)**: Enforce hardware FIDO2/WebAuthn or TOTP tokens for all users.
- **Hardware Security Modules (HSM) / KMS**: Protect disk encryption keys and JWT signing secrets using FIPS 140-2 Level 3 hardware security modules.
- **Storage-at-Rest Encryption**: Enforce AES-256-XTS bitlocker/LUKS encryption across all database volumes and document stores.
- **SIEM & SOC Forwarding**: Stream audit logs over TLS-encrypted Syslog/CEF to enterprise SIEM platforms (e.g., Splunk, IBM QRadar, Microsoft Sentinel).
- **Enterprise Data Loss Prevention (DLP)**: Implement egress inspection and file fingerprinting to ensure plant intellectual property cannot be exported.
- **Cryptographic Model Provenance & Signing**: Sign local GGUF/Ollama weights with internal PKI and verify sha256 checksums at every cold boot.
- **Software Bill of Materials (SBOM)**: Generate and track CycloneDX / SPDX SBOMs for all container layers and Python virtual environments.
- **Automated Vulnerability Management**: Daily image scanning with Trivy/Clair and continuous SAST/DAST checks.
- **High-Availability & Disaster Recovery**: Multi-node replication of ChromaDB and SQLite write-ahead logs across redundant refinery control rooms.
- **Network Micro-segmentation & OT DMZ**: Place AI inference servers in an isolated Industrial DMZ (Purdue Model Level 3.5) with unidirectional security gateways (data diodes) protecting Level 3 Supervisory systems.
- **Penetration & Red-Teaming**: Regular third-party black-box penetration testing and automated LLM prompt injection red-teaming.
- **Incident Response & Patching**: Formalized CSIRT operational runbooks and air-gapped offline package mirroring for dependency updates.

*Note: The features in this section represent production hardening recommendations and are not fully implemented in the SIH local prototype.*

---

## 5. Threat Model & Security Boundary Analysis

| Threat Vector | Potential Impact | Mitigating Architectural Boundary |
|---|---|---|
| **Adversarial Prompt Injection** | Model instruction hijacking, data leakage | Delimited untrusted context (`<untrusted_document_context>`), deterministic regex scanner, pre-LLM clearance filtering, LLM is never execution authority. |
| **Malicious Code Execution** | Host compromise, privilege escalation | Isolated Docker container (`network_mode="none"`, `read_only=True`, 256MB RAM cap, 1 CPU quota, non-root user `1000:1000`, 30s timeout). **Fail-Closed**: zero host execution fallback. |
| **Path Traversal File Upload** | Arbitrary file overwrite on host filesystem | Filename sanitization, UUID internal mapping, extension whitelist (`.pdf`, `.txt`, `.csv`, `.docx`, `.xlsx`, `.pptx`, `.json`), executable rejection (`.exe`, `.sh`, `.bat`, `.ps1`), 50MB size limit. |
| **Unauthorized Data Exfiltration** | Plant intellectual property compromise | Air-gapped deployment, local Ollama runtime on `localhost:11434`, zero external cloud inference endpoints, local ChromaDB vector store. |
| **Privilege Escalation** | Unauthorized administrative or destructive actions | Backend RBAC middleware with `require_permission()`, independent user data clearance (`PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `HIGHLY_CONFIDENTIAL`), human approval required for high-risk actions (`DELETE_FILE`, schema changes). |
| **Cross-User Session Hijacking** | Confidential query inspection | Session ownership checks per endpoint; cross-user session retrieval or messaging strictly returns HTTP 403 Forbidden with audit logging. |
| **Runaway Agent Execution** | Local DoS, resource starvation | Execution bounds (`MAX_AGENT_ITERATIONS = 10`, `MAX_TOOL_CALLS_PER_TASK = 15`, `TASK_TIMEOUT_SECONDS = 60`). |

---

## 6. Prototype Limitations & Demonstration Boundaries

1. **Host-Level Egress**: In this SIH laptop prototype, network air-gapping is demonstrated by physically disconnecting Wi-Fi/Ethernet or checking local host adapters. In a plant deployment, hardware data diodes and physical air-gaps enforce egress blocking at the network switch level.
2. **Local Image Repository**: The Docker sandbox image (`mrpl-sandbox:latest`) is built once during setup. In an air-gapped facility, updates are transferred via cryptographic offline USB media rather than registry pulls.
3. **Database Concurrency**: The local SQLite database operates in WAL mode, tailored for single-workstation or control-room appliance usage. Distributed clusters would require redundant replicated storage.
