import { useState, useEffect } from 'react';
import { admin } from '../api/client';

const ROLES = ['administrator', 'engineer', 'reviewer', 'document_manager', 'auditor', 'viewer'];
const CLEARANCE_LEVELS = ['CONFIDENTIAL', 'RESTRICTED', 'INTERNAL', 'PUBLIC'];

export default function AdminPage() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({
    username: '',
    display_name: '',
    password: '',
    roles: ['engineer'],
    clearance: 'CONFIDENTIAL',
  });
  const [creating, setCreating] = useState(false);

  useEffect(() => { loadUsers(); }, []);

  const loadUsers = async () => {
    try {
      const data = await admin.listUsers();
      setUsers(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Failed to load users:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    setCreating(true);
    try {
      await admin.createUser(form);
      setForm({ username: '', display_name: '', password: '', roles: ['engineer'], clearance: 'CONFIDENTIAL' });
      setShowCreate(false);
      await loadUsers();
    } catch (err) {
      alert('Create user failed: ' + err.message);
    } finally {
      setCreating(false);
    }
  };

  const toggleActive = async (userId, isActive) => {
    try {
      if (isActive) {
        await admin.deactivateUser(userId);
      } else {
        await admin.updateUser(userId, { is_active: true });
      }
      await loadUsers();
    } catch (err) {
      alert('Update failed: ' + err.message);
    }
  };

  const toggleRole = (role) => {
    setForm(f => ({
      ...f,
      roles: f.roles.includes(role)
        ? f.roles.filter(r => r !== role)
        : [...f.roles, role],
    }));
  };

  const clearanceBadge = (cls, roles = []) => {
    const effective = cls || (roles.includes('administrator') || roles.includes('engineer') ? 'CONFIDENTIAL' : (roles.includes('document_manager') ? 'RESTRICTED' : 'INTERNAL'));
    if (effective === 'CONFIDENTIAL') return <span className="badge badge-clearance-confidential">CONFIDENTIAL</span>;
    if (effective === 'RESTRICTED') return <span className="badge badge-clearance-restricted">RESTRICTED</span>;
    if (effective === 'PUBLIC') return <span className="badge badge-clearance-public">PUBLIC</span>;
    return <span className="badge badge-clearance-internal">INTERNAL</span>;
  };

  return (
    <div className="page-content">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Access Control & User Management</h1>
          <p className="page-subtitle">
            Configure enterprise operator credentials, role-based access control (RBAC), and security clearance
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => setShowCreate(!showCreate)}
          title="Add a new authorized operator"
        >
          {showCreate ? 'Cancel' : '+ Provision User'}
        </button>
      </div>

      {/* User Creation Panel */}
      {showCreate && (
        <div className="card mb-16 animate-slide-up" style={{ padding: '20px', borderLeft: '4px solid var(--brand-green)' }}>
          <h2 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '14px', color: 'var(--brand-olive)' }}>
            Provision New Sovereign Operator
          </h2>
          <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxWidth: '520px' }}>
            <div className="form-group">
              <label htmlFor="form-username">Username</label>
              <input
                id="form-username"
                className="input"
                placeholder="e.g. j.doe"
                value={form.username}
                onChange={e => setForm({ ...form, username: e.target.value })}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="form-display-name">Full Display Name & Department</label>
              <input
                id="form-display-name"
                className="input"
                placeholder="e.g. John Doe (Refinery Operations)"
                value={form.display_name}
                onChange={e => setForm({ ...form, display_name: e.target.value })}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="form-password">Initial Password</label>
              <input
                id="form-password"
                className="input"
                type="password"
                placeholder="Temporary secure password"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
                required
              />
            </div>

            <div>
              <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                Security Clearance Level
              </label>
              <div className="flex gap-8" style={{ flexWrap: 'wrap' }}>
                {CLEARANCE_LEVELS.map(cl => (
                  <button
                    key={cl}
                    type="button"
                    className={`btn btn-sm ${form.clearance === cl ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setForm({ ...form, clearance: cl })}
                  >
                    {cl}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                Assigned RBAC Roles
              </label>
              <div className="flex gap-8" style={{ flexWrap: 'wrap' }}>
                {ROLES.map(role => (
                  <button
                    key={role}
                    type="button"
                    className={`btn btn-sm ${form.roles.includes(role) ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => toggleRole(role)}
                  >
                    {role.replace('_', ' ')}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={creating}
              style={{ alignSelf: 'flex-start', marginTop: '4px' }}
            >
              {creating ? 'Provisioning...' : 'Complete Provisioning'}
            </button>
          </form>
        </div>
      )}

      {/* Users Data Grid */}
      {loading ? (
        <div className="empty-state"><div className="spinner spinner-lg" /></div>
      ) : (
        <div className="card" style={{ overflow: 'hidden' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Operator</th>
                <th>Assigned Roles</th>
                <th>Clearance</th>
                <th>Status</th>
                <th>Created</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map(u => (
                <tr key={u.id}>
                  <td>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{u.display_name}</div>
                    <div className="text-sm text-muted">@{u.username} · ID #{u.id}</div>
                  </td>
                  <td>
                    <div className="flex gap-6" style={{ flexWrap: 'wrap' }}>
                      {u.roles?.map(r => (
                        <span key={r} className="badge badge-info">{r.replace('_', ' ')}</span>
                      ))}
                    </div>
                  </td>
                  <td>{clearanceBadge(u.clearance, u.roles)}</td>
                  <td>
                    <span className={`badge ${u.is_active ? 'badge-success' : 'badge-error'}`}>
                      {u.is_active ? 'ACTIVE' : 'DEACTIVATED'}
                    </span>
                  </td>
                  <td className="text-sm" style={{ whiteSpace: 'nowrap' }}>
                    {new Date(u.created_at).toLocaleDateString()}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className={`btn btn-sm ${u.is_active ? 'btn-secondary' : 'btn-primary'}`}
                      onClick={() => toggleActive(u.id, u.is_active)}
                    >
                      {u.is_active ? 'Deactivate' : 'Activate'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
