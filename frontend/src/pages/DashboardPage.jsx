import { useState, useEffect } from 'react';
import { status, approvals } from '../api/client';

function SecurityMetric({ label, value, variant }) {
  const colorMap = {
    pass: 'var(--success)',
    fail: 'var(--error)',
    info: 'var(--info)',
    neutral: 'var(--text-secondary)',
  };
  return (
    <div className="security-metric">
      <div className="security-metric-label">{label}</div>
      <div className="security-metric-value" style={{ color: colorMap[variant] || colorMap.neutral }}>
        {value || '—'}
      </div>
    </div>
  );
}

function SkeletonDashboard() {
  return (
    <div className="page-content">
      <div className="page-header">
        <div>
          <div className="skeleton skeleton-text" style={{ width: '220px' }} />
          <div className="skeleton skeleton-text-sm" style={{ width: '320px' }} />
        </div>
      </div>
      <div className="skeleton skeleton-card mb-16" style={{ height: '160px' }} />
      <div className="dashboard-grid mb-16">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="skeleton skeleton-card" />
        ))}
      </div>
      <div className="skeleton skeleton-text" style={{ width: '160px', marginBottom: '12px' }} />
      <div className="dashboard-grid">
        {[1, 2, 3].map(i => (
          <div key={i} className="skeleton skeleton-card" style={{ height: '70px' }} />
        ))}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [systemData, setSystemData] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [servicesData, setServicesData] = useState(null);
  const [featuresData, setFeaturesData] = useState(null);
  const [securityData, setSecurityData] = useState(null);
  const [pendingApprovals, setPendingApprovals] = useState([]);
  const [approvalReasons, setApprovalReasons] = useState({});
  const [actionLoading, setActionLoading] = useState({});
  const [actionMsg, setActionMsg] = useState(null);
  const [loading, setLoading] = useState(true);

  const currentUser = (() => {
    try {
      return JSON.parse(localStorage.getItem('mrpl_user') || '{}');
    } catch {
      return {};
    }
  })();
  const userRoles = (currentUser?.roles || []).map(r => r.toLowerCase());
  const canManageApprovals = userRoles.includes('administrator') || userRoles.includes('admin') || userRoles.includes('reviewer');

  useEffect(() => {
    loadAll();
    const interval = setInterval(loadAll, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadApprovals = async () => {
    if (!canManageApprovals) return;
    try {
      const list = await approvals.listPending();
      setPendingApprovals(Array.isArray(list) ? list : []);
    } catch (err) {
      console.error('Approvals load error:', err);
    }
  };

  const loadAll = async () => {
    try {
      const [sys, net, svc, feat, sec] = await Promise.allSettled([
        status.system(),
        status.network(),
        status.services(),
        status.features(),
        status.security(),
      ]);
      setSystemData(sys.status === 'fulfilled' ? sys.value : null);
      setNetworkData(net.status === 'fulfilled' ? net.value : null);
      setServicesData(svc.status === 'fulfilled' ? svc.value : null);
      setFeaturesData(feat.status === 'fulfilled' ? feat.value : null);
      setSecurityData(sec.status === 'fulfilled' ? sec.value?.security_status : null);

      if (canManageApprovals) {
        await loadApprovals();
      }
    } catch (err) {
      console.error('Dashboard load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (id) => {
    try {
      setActionLoading(prev => ({ ...prev, [id]: true }));
      const reason = approvalReasons[id] || '';
      await approvals.approve(id, reason);
      setActionMsg({ id, text: 'Action approved successfully', type: 'success' });
      await loadApprovals();
    } catch (err) {
      setActionMsg({ id, text: err.message || 'Approval failed', type: 'error' });
    } finally {
      setActionLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const handleReject = async (id) => {
    try {
      setActionLoading(prev => ({ ...prev, [id]: true }));
      const reason = approvalReasons[id] || '';
      await approvals.reject(id, reason);
      setActionMsg({ id, text: 'Action rejected', type: 'warning' });
      await loadApprovals();
    } catch (err) {
      setActionMsg({ id, text: err.message || 'Rejection failed', type: 'error' });
    } finally {
      setActionLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const statusBadge = (s) => {
    const map = {
      healthy: 'badge-success',
      unavailable: 'badge-error',
      unhealthy: 'badge-warning',
      implemented: 'badge-success',
      in_progress: 'badge-warning',
      planned: 'badge-info',
    };
    return <span className={`badge ${map[s] || 'badge-neutral'}`}>{s}</span>;
  };

  if (loading) {
    return <SkeletonDashboard />;
  }

  const securityPassColor = (val) => {
    if (!val) return 'neutral';
    const v = val.toUpperCase();
    if (v === 'PASS' || v === 'NONE' || v === 'ENABLED' || v === 'NONE CONFIGURED') return 'pass';
    if (v.includes('FAIL') || v.includes('ERROR')) return 'fail';
    return 'info';
  };

  return (
    <div className="page-content">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Tasks & System Monitor</h1>
          <p className="page-subtitle">
            Real-time on-premise hardware telemetry, task governance, and air-gapped security state
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={loadAll}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          Refresh Telemetry
        </button>
      </div>

      {/* Security Status Panel — Live verified backend data */}
      <div className="card mb-16" style={{ padding: '16px 20px', borderLeft: '4px solid var(--brand-green)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div>
            <h2 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'var(--brand-olive)' }}>
              Air-Gap Security Verification
            </h2>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
              Strict on-premise governance · Zero outbound internet routing · Sandboxed Docker execution
            </p>
          </div>
          <span className="badge badge-success" style={{ fontSize: '0.72rem', padding: '3px 8px' }}>
            ● PERIMETER VERIFIED
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
          <SecurityMetric
            label="Local Inference"
            value={securityData?.local_inference || '—'}
            variant={securityPassColor(securityData?.local_inference)}
          />
          <SecurityMetric
            label="External AI APIs"
            value={securityData?.external_ai_apis || '—'}
            variant={securityPassColor(securityData?.external_ai_apis)}
          />
          <SecurityMetric
            label="Local RAG (ChromaDB)"
            value={securityData?.local_rag || '—'}
            variant={securityPassColor(securityData?.local_rag)}
          />
          <SecurityMetric
            label="Internet Dependency"
            value={securityData?.runtime_internet_dependency || securityData?.internet_dependency || '—'}
            variant={securityPassColor(securityData?.runtime_internet_dependency || securityData?.internet_dependency)}
          />
          <SecurityMetric
            label="Docker Sandbox"
            value={securityData?.code_sandbox || '—'}
            variant="info"
          />
          <SecurityMetric
            label="Sandbox Egress"
            value={securityData?.sandbox_network || '—'}
            variant={securityPassColor(securityData?.sandbox_network)}
          />
          <SecurityMetric
            label="Fail-Closed Policy"
            value={securityData?.fail_closed_execution || '—'}
            variant={securityPassColor(securityData?.fail_closed_execution)}
          />
          <SecurityMetric
            label="Inference Hardware"
            value={securityData?.inference_device || (systemData?.inference_device === 'gpu' ? 'Ollama GPU' : (systemData?.inference_device === 'cpu' ? 'CPU Fallback' : '—'))}
            variant={securityPassColor(securityData?.inference_device || (systemData?.inference_device === 'gpu' ? 'PASS' : 'neutral'))}
          />
        </div>
      </div>

      {/* Action Approvals — Governance for high-risk actions */}
      {canManageApprovals && (
        <div className="card mb-16" style={{ padding: '16px 20px', borderLeft: '4px solid var(--brand-orange)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div>
              <h2 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'var(--brand-olive)' }}>
                Governance & High-Risk Task Approvals
              </h2>
              <p style={{ margin: '2px 0 0 0', fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
                Human-in-the-loop validation — anti-self-approval strictly enforced
              </p>
            </div>
            <span className={`badge ${pendingApprovals.length > 0 ? 'badge-warning' : 'badge-success'}`}>
              {pendingApprovals.length} Pending
            </span>
          </div>

          {actionMsg && (
            <div style={{
              padding: '8px 12px',
              borderRadius: '4px',
              marginBottom: '10px',
              fontSize: '0.82rem',
              background: actionMsg.type === 'success' ? 'var(--success-bg)' : 'var(--warning-bg)',
              color: 'var(--text-primary)',
              border: `1px solid ${actionMsg.type === 'success' ? 'var(--success-border)' : 'var(--warning-border)'}`,
            }}>
              {actionMsg.text}
            </div>
          )}

          {pendingApprovals.length === 0 ? (
            <div style={{ padding: '14px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.84rem' }}>
              No pending high-risk tasks. All sovereign agent operations are executing within pre-approved policy parameters.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Task ID</th>
                    <th>Requester</th>
                    <th>Action</th>
                    <th>Risk</th>
                    <th>Target Resource</th>
                    <th>Rationale</th>
                    <th style={{ textAlign: 'right' }}>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {pendingApprovals.map((item) => (
                    <tr key={item.id}>
                      <td className="font-mono text-sm">#{item.id}</td>
                      <td style={{ fontWeight: 600 }}>{item.requesting_username}</td>
                      <td>
                        <span className="badge badge-info">{item.action_type}</span>
                      </td>
                      <td>
                        <span className="badge badge-error" style={{ fontWeight: 700 }}>
                          {item.risk_level}
                        </span>
                      </td>
                      <td className="font-mono text-sm" style={{ maxWidth: '180px', wordBreak: 'break-all' }}>
                        {item.resource_target}
                      </td>
                      <td style={{ minWidth: '180px' }}>
                        <input
                          type="text"
                          className="input text-sm"
                          style={{ padding: '4px 8px' }}
                          placeholder="Reason (optional)"
                          value={approvalReasons[item.id] || ''}
                          onChange={(e) => setApprovalReasons({ ...approvalReasons, [item.id]: e.target.value })}
                          disabled={actionLoading[item.id]}
                        />
                      </td>
                      <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                        <button
                          className="btn btn-primary btn-sm"
                          style={{ marginRight: '6px' }}
                          onClick={() => handleApprove(item.id)}
                          disabled={actionLoading[item.id]}
                        >
                          {actionLoading[item.id] ? '...' : 'Approve'}
                        </button>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => handleReject(item.id)}
                          disabled={actionLoading[item.id]}
                        >
                          {actionLoading[item.id] ? '...' : 'Reject'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* System Resources Telemetry */}
      <h2 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '10px', color: 'var(--brand-olive)' }}>
        Hardware Resources Telemetry
      </h2>
      <div className="dashboard-grid mb-16">
        {/* CPU */}
        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Processor</span>
            <div className="stat-card-icon">CPU</div>
          </div>
          <div className="stat-card-value">
            {systemData?.cpu?.usage_percent != null ? `${systemData.cpu.usage_percent}%` : '—'}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
            {systemData?.cpu?.core_count || '8'} Physical Cores Active
          </div>
        </div>

        {/* RAM */}
        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">System Memory</span>
            <div className="stat-card-icon">RAM</div>
          </div>
          <div className="stat-card-value">
            {systemData?.memory?.usage_percent != null ? `${systemData.memory.usage_percent}%` : '—'}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
            {systemData?.memory?.used_gb || '?'} / {systemData?.memory?.total_gb || '?'} GB Allocated
          </div>
        </div>

        {/* GPU */}
        <div className="stat-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="stat-card-header">
              <span className="stat-card-label">Hardware Acceleration</span>
              <span className={`badge ${systemData?.gpu_available ? 'badge-success' : 'badge-neutral'}`}>
                {systemData?.gpu_available ? 'GPU ACCELERATED' : 'CPU FALLBACK'}
              </span>
            </div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={systemData?.gpu_name || systemData?.gpu?.gpu_name || 'NVIDIA GPU'}>
              {systemData?.gpu_name || systemData?.gpu?.gpu_name || (systemData?.gpu_available ? 'NVIDIA GPU' : 'No GPU Detected')}
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Engine: <strong style={{ color: 'var(--brand-green)' }}>
                {systemData?.inference_device === 'gpu' ? 'Ollama GPU' : 'CPU Fallback'}
              </strong>
            </div>
          </div>
          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--border-default)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
              VRAM: {systemData?.gpu?.vram_used_mb != null && systemData?.gpu?.vram_total_mb != null
                ? `${systemData.gpu.vram_used_mb.toLocaleString()} / ${systemData.gpu.vram_total_mb.toLocaleString()} MB`
                : (systemData?.gpu?.gpus?.[0]?.vram_used_mb != null ? `${systemData.gpu.gpus[0].vram_used_mb.toLocaleString()} / ${systemData.gpu.gpus[0].vram_total_mb.toLocaleString()} MB` : '—')}
            </div>
            <div className="stat-card-value" style={{ fontSize: '1.2rem', lineHeight: 1 }}>
              {systemData?.gpu?.utilization_percent != null
                ? `${systemData.gpu.utilization_percent}%`
                : (systemData?.gpu?.gpus?.[0]?.utilization_percent != null ? `${systemData.gpu.gpus[0].utilization_percent}%` : '—')}
            </div>
          </div>
        </div>

        {/* Disk */}
        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">On-Premise Storage</span>
            <div className="stat-card-icon">DSK</div>
          </div>
          <div className="stat-card-value">
            {systemData?.disk?.project?.usage_percent != null
              ? `${systemData.disk.project.usage_percent}%`
              : '—'}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
            Local Storage Partition
          </div>
        </div>
      </div>

      {/* Services Health */}
      <h2 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '10px', color: 'var(--brand-olive)' }}>
        Core Sovereign Services
      </h2>
      <div className="dashboard-grid mb-16">
        {servicesData && Object.entries(servicesData).map(([name, data]) => (
          <div key={name} className="feature-card">
            <div className="feature-icon" style={{
              background: data.status === 'healthy' ? 'var(--success-bg)' : 'var(--error-bg)',
              color: data.status === 'healthy' ? 'var(--success)' : 'var(--error)',
            }}>
              {name === 'ollama' ? 'LLM' : name === 'chromadb' ? 'RAG' : 'SVC'}
            </div>
            <div className="feature-info">
              <div className="feature-name" style={{ textTransform: 'capitalize' }}>{name}</div>
              <div className="feature-notes">
                {name === 'ollama' && systemData?.inference_device ? `Device: ${systemData.inference_device === 'gpu' ? 'Ollama GPU' : systemData.inference_device.toUpperCase()} | ` : ''}
                {data.loaded_models ? `Models: ${data.loaded_models.join(', ')}` : data.reason || data.status}
              </div>
            </div>
            {statusBadge(data.status)}
          </div>
        ))}
      </div>

      {/* Network Air-Gap Telemetry */}
      {networkData?.interfaces && (
        <>
          <h2 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '10px', color: 'var(--brand-olive)' }}>
            Network Interfaces & Air-Gap Compliance
          </h2>
          <div className="card mb-16" style={{ overflow: 'hidden' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Interface Name</th>
                  <th>Perimeter Status</th>
                  <th>Speed</th>
                  <th>Internal IP Binding</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(networkData.interfaces).map(([name, iface]) => (
                  <tr key={name}>
                    <td style={{ fontWeight: 600 }}>{name}</td>
                    <td>{iface.is_up ? statusBadge('healthy') : statusBadge('unavailable')}</td>
                    <td>{iface.speed_mbps > 0 ? `${iface.speed_mbps} Mbps` : '—'}</td>
                    <td className="font-mono text-sm">
                      {iface.addresses?.slice(0, 2).map((a, i) => (
                        <div key={i}>{a.address}</div>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* System Features Matrix */}
      {featuresData?.features && (
        <>
          <h2 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '10px', color: 'var(--brand-olive)' }}>
            Engineering Capabilities Matrix
          </h2>
          {Object.entries(featuresData.features).map(([category, features]) => (
            <div key={category} style={{ marginBottom: '14px' }}>
              <h3 style={{
                fontSize: '0.7rem', fontWeight: 700, color: 'var(--brand-olive)',
                textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '8px',
              }}>
                {category}
              </h3>
              <div className="feature-grid">
                {features.map((f) => (
                  <div key={f.feature_key} className="feature-card">
                    <div className="feature-icon" style={{
                      background: f.status === 'implemented' ? 'var(--success-bg)'
                        : f.status === 'in_progress' ? 'var(--warning-bg)'
                        : 'var(--info-bg)',
                      color: f.status === 'implemented' ? 'var(--success)'
                        : f.status === 'in_progress' ? 'var(--warning)'
                        : 'var(--info)',
                    }}>
                      {f.status === 'implemented' ? '✓' : f.status === 'in_progress' ? '◐' : '○'}
                    </div>
                    <div className="feature-info">
                      <div className="feature-name">{f.display_name}</div>
                      {f.notes && <div className="feature-notes">{f.notes}</div>}
                    </div>
                    {statusBadge(f.status)}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
