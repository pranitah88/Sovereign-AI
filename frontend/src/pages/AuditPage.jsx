import { useState, useEffect, useCallback } from 'react';
import { audit } from '../api/client';

const SUPPORTED_ACTIONS = [
  { value: null, label: 'All Action Types' },
  // Authentication & Access
  { value: 'login', label: 'Login (login)' },
  { value: 'logout', label: 'Logout (logout)' },
  { value: 'rbac_access_denied', label: 'RBAC Access Denied' },
  // Chat & Collaboration
  { value: 'chat_message', label: 'Chat Message (chat_message)' },
  { value: 'chat_session_create', label: 'Session Create' },
  { value: 'chat_session_delete', label: 'Session Delete' },
  { value: 'chat_image_analysis', label: 'Image Analysis' },
  { value: 'chat_access_violation', label: 'Chat Access Violation' },
  // Agent & Execution
  { value: 'agent_invoke', label: 'Agent Invoke (agent_invoke)' },
  { value: 'tool_execution', label: 'Tool Execution' },
  { value: 'sandbox_execute', label: 'Sandbox Execution' },
  { value: 'temporal_grounding_verified', label: 'Temporal Grounding' },
  // Document Operations
  { value: 'file_upload', label: 'File Upload (file_upload)' },
  { value: 'file_delete', label: 'File Delete (file_delete)' },
  { value: 'file_reindex', label: 'File Reindex' },
  { value: 'DOCUMENT_GENERATED', label: 'Document Generated' },
  { value: 'DOCUMENT_GENERATION_FAILED', label: 'Document Gen Failed' },
  { value: 'file_upload_security_violation', label: 'Upload Violation' },
  // RAG & Knowledge
  { value: 'rag_query', label: 'RAG Query (rag_query)' },
  // Security Boundaries & Governance
  { value: 'PROMPT_INJECTION_DETECTED', label: 'Prompt Injection Detected' },
  { value: 'SCOPE_GUARD_BLOCKED', label: 'Scope Guard Blocked' },
  { value: 'NETWORK_SEAL_EGRESS_BLOCKED', label: 'Network Seal Blocked' },
  { value: 'SECURITY_CONTROL_VERIFIED', label: 'Security Control Verified' },
  { value: 'VISION_VERIFICATION_APPROVED', label: 'Vision Approved' },
  { value: 'VISION_VERIFICATION_ESCALATED', label: 'Vision Escalated' },
  // Administration & Governance
  { value: 'model_toggle', label: 'Model Toggle (model_toggle)' },
  { value: 'user_create', label: 'User Create' },
  { value: 'user_update', label: 'User Update' },
  { value: 'user_deactivate', label: 'User Deactivate' },
  { value: 'action_proposed', label: 'Approval Proposed' },
  { value: 'action_approval_decided', label: 'Approval Decided' },
  // Voice Assistant
  { value: 'voice_interaction', label: 'Voice Interaction' },
  { value: 'voice_interrupted', label: 'Voice Interrupted' },
  { value: 'VOICE_SESSION_STOPPED', label: 'Voice Session Stopped' },
];

export default function AuditPage() {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ action: null, outcome: null, limit: 50, offset: 0 });

  const loadLogs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const queryParams = {
        limit: filters.limit,
        offset: filters.offset,
        ...(filters.action ? { action: filters.action } : {}),
        ...(filters.outcome ? { outcome: filters.outcome } : {}),
      };
      const data = await audit.getLogs(queryParams);
      setLogs(Array.isArray(data?.logs) ? data.logs : []);
      setTotal(typeof data?.total === 'number' ? data.total : 0);
    } catch (err) {
      const msg = err?.message || 'Failed to load audit logs';
      const isForbidden = msg.includes('403') || msg.toLowerCase().includes('permission') || msg.toLowerCase().includes('forbidden');
      setError({
        isForbidden,
        message: isForbidden
          ? 'You do not have permission to view audit records.'
          : `Unable to load audit records: ${msg}`,
      });
      console.error('Failed to load audit logs:', err);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  const outcomeBadge = (outcome) => {
    const map = {
      success: 'badge-success',
      failure: 'badge-error',
      denied: 'badge-warning',
    };
    return <span className={`badge ${map[outcome] || 'badge-neutral'}`}>{outcome?.toUpperCase()}</span>;
  };

  const parseDetails = (raw) => {
    if (!raw) return {};
    if (typeof raw === 'object') return raw;
    try {
      return JSON.parse(raw);
    } catch {
      return {};
    }
  };

  return (
    <div className="page-content">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Traceability & Audit Log</h1>
          <p className="page-subtitle">
            Append-only on-premise activity record · {total.toLocaleString()} security and agent events logged
          </p>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={loadLogs}
          disabled={loading}
          title="Fetch latest audit events from database"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          Refresh Log
        </button>
      </div>

      {/* Filter Toolbar */}
      <div className="flex gap-12 mb-16" style={{ flexWrap: 'wrap' }}>
        <select
          className="input"
          style={{ width: '240px' }}
          value={filters.action ?? ''}
          onChange={e => {
            const val = e.target.value.trim();
            setFilters(prev => ({ ...prev, action: val ? val : null, offset: 0 }));
          }}
          aria-label="Filter by action type"
        >
          {SUPPORTED_ACTIONS.map(item => (
            <option key={item.value ?? 'all'} value={item.value ?? ''}>
              {item.label}
            </option>
          ))}
        </select>

        <select
          className="input"
          style={{ width: '160px' }}
          value={filters.outcome ?? ''}
          onChange={e => {
            const val = e.target.value.trim();
            setFilters(prev => ({ ...prev, outcome: val ? val : null, offset: 0 }));
          }}
          aria-label="Filter by result outcome"
        >
          <option value="">All Results</option>
          <option value="success">Success</option>
          <option value="failure">Failure</option>
          <option value="denied">Denied</option>
        </select>
      </div>

      {/* State Separation: Loading, Error (Access Denied / Load Failed), Empty, Data */}
      {loading ? (
        <div className="empty-state card">
          <div className="spinner spinner-lg mb-16" />
          <div className="empty-state-title">Loading audit records...</div>
          <div className="empty-state-description">Querying on-premise activity ledger.</div>
        </div>
      ) : error ? (
        <div className="empty-state card" role="alert">
          <div className="empty-state-title" style={{ color: 'var(--color-danger, #ef4444)' }}>
            {error.isForbidden ? 'Access Denied' : 'Load Failed'}
          </div>
          <div className="empty-state-description">
            {error.message}
          </div>
          <button className="btn btn-secondary btn-sm mt-16" onClick={loadLogs}>
            Retry Request
          </button>
        </div>
      ) : logs.length === 0 ? (
        <div className="empty-state card">
          <div className="empty-state-title">No audit records located</div>
          <div className="empty-state-description">
            {filters.action || filters.outcome
              ? 'No events match the selected criteria. Adjust filters.'
              : 'Enterprise system activity will appear here in chronological sequence.'}
          </div>
        </div>
      ) : (
        <>
          <div className="card" style={{ overflow: 'hidden' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>User</th>
                  <th>Action</th>
                  <th>Task / Context</th>
                  <th>Model</th>
                  <th>Resource</th>
                  <th>Result</th>
                  <th>Client IP</th>
                </tr>
              </thead>
              <tbody>
                {logs.map(log => {
                  const details = parseDetails(log.details);
                  const model = details.model || details.model_id || (log.action === 'agent_invoke' ? 'Local LLM' : '—');
                  const task = details.task_type || (log.action === 'agent_invoke' ? 'Agent Pipeline' : (log.target ? log.target.split(':')[0] : '—'));
                  const resource = log.target || details.resource || details.filename || '—';

                  return (
                    <tr key={log.id}>
                      <td className="font-mono text-sm" style={{ whiteSpace: 'nowrap' }}>
                        {new Date(log.created_at).toLocaleString()}
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--brand-olive)' }}>
                        {log.username || (log.user_id ? `user:${log.user_id}` : 'SYSTEM')}
                      </td>
                      <td>
                        <span className="badge badge-info">{log.action}</span>
                      </td>
                      <td className="text-sm">
                        <span className="badge badge-neutral" style={{ textTransform: 'uppercase' }}>
                          {task}
                        </span>
                      </td>
                      <td className="font-mono text-sm" style={{ color: 'var(--text-secondary)' }}>
                        {model}
                      </td>
                      <td
                        className="text-sm"
                        style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={resource}
                      >
                        {resource}
                      </td>
                      <td>{outcomeBadge(log.outcome)}</td>
                      <td className="font-mono text-sm" style={{ color: 'var(--text-tertiary)' }}>
                        {log.ip_address || '127.0.0.1'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          <div className="flex items-center justify-between mt-16">
            <span className="text-sm text-muted">
              Displaying {filters.offset + 1}–{Math.min(filters.offset + filters.limit, total)} of {total.toLocaleString()} verified events
            </span>
            <div className="flex gap-8">
              <button
                className="btn btn-secondary btn-sm"
                disabled={filters.offset === 0}
                onClick={() => setFilters(prev => ({ ...prev, offset: Math.max(0, prev.offset - prev.limit) }))}
              >
                Previous
              </button>
              <button
                className="btn btn-secondary btn-sm"
                disabled={filters.offset + filters.limit >= total}
                onClick={() => setFilters(prev => ({ ...prev, offset: prev.offset + prev.limit }))}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
