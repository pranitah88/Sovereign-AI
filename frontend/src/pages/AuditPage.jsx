import { useState, useEffect } from 'react';
import { audit } from '../api/client';

export default function AuditPage() {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ action: '', outcome: '', limit: 50, offset: 0 });

  useEffect(() => { loadLogs(); }, [filters]);

  const loadLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await audit.getLogs(filters);
      setLogs(Array.isArray(data.logs) ? data.logs : []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message || 'Failed to load audit logs');
      console.error('Failed to load audit logs:', err);
    } finally {
      setLoading(false);
    }
  };

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
            Immutable on-premise activity record · {total} security and agent events logged
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={loadLogs} disabled={loading}>
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
          style={{ width: '220px' }}
          value={filters.action}
          onChange={e => setFilters({ ...filters, action: e.target.value || '', offset: 0 })}
          aria-label="Filter by action type"
        >
          <option value="">All Action Types</option>
          {[
            'login', 'logout', 'chat_message', 'chat_session_create',
            'file_upload', 'file_delete', 'rag_query', 'agent_invoke',
            'sandbox_execute', 'user_create', 'user_update',
            'user_deactivate', 'model_toggle'
          ].map(a => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>

        <select
          className="input"
          style={{ width: '160px' }}
          value={filters.outcome}
          onChange={e => setFilters({ ...filters, outcome: e.target.value || '', offset: 0 })}
          aria-label="Filter by result outcome"
        >
          <option value="">All Results</option>
          <option value="success">Success</option>
          <option value="failure">Failure</option>
          <option value="denied">Denied</option>
        </select>
      </div>

      {error && (
        <div className="alert alert-error mb-16" role="alert">{error}</div>
      )}

      {loading ? (
        <div className="empty-state"><div className="spinner spinner-lg" /></div>
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
                        {log.username || `user:${log.user_id}` || 'SYSTEM'}
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
                      <td className="text-sm" style={{ maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={resource}>
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
              Displaying {filters.offset + 1}–{Math.min(filters.offset + filters.limit, total)} of {total} verified events
            </span>
            <div className="flex gap-8">
              <button
                className="btn btn-secondary btn-sm"
                disabled={filters.offset === 0}
                onClick={() => setFilters({ ...filters, offset: Math.max(0, filters.offset - filters.limit) })}
              >
                Previous
              </button>
              <button
                className="btn btn-secondary btn-sm"
                disabled={filters.offset + filters.limit >= total}
                onClick={() => setFilters({ ...filters, offset: filters.offset + filters.limit })}
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
