import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, NavLink, useNavigate, useLocation, Link } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import ChatPage from './pages/ChatPage';
import DashboardPage from './pages/DashboardPage';
import DocumentsPage from './pages/DocumentsPage';
import AdminPage from './pages/AdminPage';
import AuditPage from './pages/AuditPage';
import FilesPage from './pages/FilesPage';
import { auth } from './api/client';

export function MrplLogo({ size = 36 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="MRPL ONGC Emblem">
      <rect width="100" height="100" rx="8" fill="#234d20" />
      <text x="50" y="22" fill="#ffffff" fontSize="14" fontWeight="900" fontFamily="sans-serif" textAnchor="middle" letterSpacing="2">ONGC</text>
      {/* Refinery chimney & derrick */}
      <path d="M49 28 L49 68 M51 28 L51 68" stroke="#ffffff" strokeWidth="2" />
      <path d="M42 70 L50 32 L58 70" stroke="#ffffff" strokeWidth="2.5" fill="none" strokeLinecap="round" />
      <path d="M45 54 L55 54" stroke="#ffffff" strokeWidth="2" />
      {/* Industrial flame in MRPL orange */}
      <path d="M50 25 C47 30 50 34 50 34 C50 34 53 30 50 25 Z" fill="#d96b27" />
      <circle cx="50" cy="51" r="16" stroke="#ffffff" strokeWidth="2" fill="none" opacity="0.85" />
      <text x="50" y="88" fill="#ffffff" fontSize="14" fontWeight="900" fontFamily="sans-serif" textAnchor="middle" letterSpacing="1.5">MRPL</text>
    </svg>
  );
}

const PAGE_TITLES = {
  '/': 'MRPL Sovereign AI | Engineering Chat',
  '/dashboard': 'MRPL Sovereign AI | Tasks & System Monitor',
  '/documents': 'MRPL Sovereign AI | Knowledge Base & Collections',
  '/files': 'MRPL Sovereign AI | Documents & Outputs',
  '/admin': 'MRPL Sovereign AI | Access Control & Users',
  '/audit': 'MRPL Sovereign AI | Traceability & Audit Log',
  '/login': 'MRPL Sovereign AI | Sign In',
};

const PAGE_NAMES = {
  '/': 'Engineering Chat',
  '/dashboard': 'Tasks & System Monitor',
  '/documents': 'Knowledge Base & Collections',
  '/files': 'Documents & Outputs',
  '/admin': 'Access Control & Users',
  '/audit': 'Traceability & Audit Log',
};

function usePageTitle() {
  const location = useLocation();
  useEffect(() => {
    const title = PAGE_TITLES[location.pathname] || 'MRPL Sovereign AI | Page Not Found';
    document.title = title;

    const descriptions = {
      '/': 'Secure on-premise AI chat with local model inference, RAG retrieval, and Docker-sandboxed code execution.',
      '/dashboard': 'Real-time tasks monitoring, security status, and service health for the MRPL Sovereign AI Workbench.',
      '/documents': 'Upload, index, and manage knowledge base documents for on-premise RAG retrieval.',
      '/files': 'Browse generated reports, exports, and uploaded repository documents.',
      '/admin': 'Manage users, roles, clearance levels, and access control.',
      '/audit': 'View audit logs for all system actions, security events, and access attempts.',
      '/login': 'Sign in to the MRPL Sovereign AI Workbench.',
    };
    const meta = document.querySelector('meta[name="description"]');
    if (meta) {
      meta.setAttribute(
        'content',
        descriptions[location.pathname] || 'MRPL Sovereign AI Workbench — On-premise agentic AI system for Mangalore Refinery and Petrochemicals Limited.'
      );
    }
  }, [location.pathname]);
}

function RouteWatcher() {
  usePageTitle();
  return null;
}

function NotFoundPage() {
  return (
    <div className="not-found-page">
      <div className="not-found-code">404</div>
      <h1 className="not-found-title">Route Not Located</h1>
      <p className="not-found-description">
        The requested resource path does not exist in this Sovereign AI Workbench. Return to the active workspace.
      </p>
      <Link to="/" className="btn btn-primary">Return to Chat</Link>
    </div>
  );
}

function getClearanceLevel(roles = []) {
  if (roles.includes('administrator') || roles.includes('auditor') || roles.includes('engineer')) {
    return { label: 'CONFIDENTIAL', className: 'badge-clearance-confidential' };
  }
  if (roles.includes('document_manager') || roles.includes('reviewer')) {
    return { label: 'RESTRICTED', className: 'badge-clearance-restricted' };
  }
  return { label: 'INTERNAL', className: 'badge-clearance-internal' };
}

function Layout({ user, onLogout, children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const roles = user?.roles || [];
  const isAdmin = roles.includes('administrator');
  const isAuditor = roles.includes('auditor') || isAdmin;
  const canManageDocs = roles.includes('document_manager') || roles.includes('engineer') || isAdmin || roles.includes('reviewer');

  const clearance = getClearanceLevel(roles);
  const currentPageName = PAGE_NAMES[location.pathname] || 'Workbench';

  const handleLogout = async () => {
    try { await auth.logout(); } catch (_e) { /* session may already be expired */ }
    localStorage.removeItem('mrpl_token');
    localStorage.removeItem('mrpl_user');
    onLogout();
    navigate('/login');
  };

  return (
    <div className="app-layout">
      {/* ── Top Corporate Header ────────────────────────────────────────── */}
      <header className="app-top-header" role="banner">
        <div className="header-brand">
          <div className="header-brand-logo">
            <MrplLogo size={36} />
            <div className="header-brand-text">
              <span className="header-brand-title">MRPL Sovereign AI Workbench</span>
              <span className="header-brand-subtitle">
                Mangalore Refinery and Petrochemicals Limited · A Subsidiary of ONGC
              </span>
            </div>
          </div>

          <div className="header-separator" aria-hidden="true" />

          <div className="header-context">
            <span className="header-context-badge">{currentPageName}</span>
          </div>
        </div>

        <div className="header-actions">
          {/* Security & Runtime status indicators */}
          <div className="header-status-pills" role="status" aria-label="Security Status">
            <span className="status-indicator-pill live-green" title="Operating exclusively within MRPL network perimeter">
              <span className="status-dot" aria-hidden="true" /> ON-PREM
            </span>
            <span className="status-indicator-pill air-gapped" title="Zero outbound internet connections permitted">
              NO EXTERNAL EGRESS
            </span>
            <span className="status-indicator-pill" title="Local model inference via Ollama">
              LOCAL MODEL
            </span>
          </div>

          {/* User Profile & Clearance */}
          <div className="header-user-pill">
            <div className="header-user-info">
              <span className="header-user-name">{user?.display_name || user?.username || 'User'}</span>
              <span className={`badge ${clearance.className}`} style={{ fontSize: '0.62rem', padding: '1px 5px' }}>
                {clearance.label}
              </span>
            </div>

            <button
              className="btn btn-secondary btn-sm"
              onClick={handleLogout}
              title="Sign out of MRPL Sovereign AI Workbench"
              aria-label="Sign out"
            >
              Sign Out
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Application Workspace Body ─────────────────────────────── */}
      <div className="app-body">
        {/* Left Sidebar Navigation */}
        <aside className="sidebar" aria-label="Workbench navigation">
          <nav className="sidebar-nav">
            <div className="nav-section-header">Workspace</div>

            {/* Chat */}
            <NavLink
              to="/"
              end
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              title="Interactive Engineering Chat"
            >
              <span className="nav-item-icon" aria-hidden="true">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </span>
              <span>Chat</span>
            </NavLink>

            {/* Knowledge Base */}
            {canManageDocs && (
              <NavLink
                to="/documents"
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                title="Knowledge Base & Document Collections"
              >
                <span className="nav-item-icon" aria-hidden="true">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                  </svg>
                </span>
                <span>Knowledge Base</span>
              </NavLink>
            )}

            {/* Documents (Repository files) */}
            {canManageDocs && (
              <NavLink
                to="/files?tab=uploaded"
                className={({ isActive }) => `nav-item ${isActive && location.search.includes('tab=uploaded') ? 'active' : (!location.search && isActive ? 'active' : '')}`}
                title="Uploaded Repository Documents"
              >
                <span className="nav-item-icon" aria-hidden="true">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                  </svg>
                </span>
                <span>Documents</span>
              </NavLink>
            )}

            {/* Tasks (System & Approvals) */}
            <NavLink
              to="/dashboard"
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              title="Tasks, Governance & System Resources"
            >
              <span className="nav-item-icon" aria-hidden="true">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7" />
                  <rect x="14" y="3" width="7" height="7" />
                  <rect x="14" y="14" width="7" height="7" />
                  <rect x="3" y="14" width="7" height="7" />
                </svg>
              </span>
              <span>Tasks</span>
            </NavLink>

            {/* Outputs (Deliverables, Generated DOCX/PDF/XLSX) */}
            {canManageDocs && (
              <NavLink
                to="/files?tab=generated"
                className={() => `nav-item ${location.search.includes('tab=generated') ? 'active' : ''}`}
                title="Generated Reports & Deliverable Outputs"
              >
                <span className="nav-item-icon" aria-hidden="true">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </span>
                <span>Outputs</span>
              </NavLink>
            )}

            {/* Administration / Governance */}
            {(isAdmin || isAuditor) && (
              <>
                <div className="nav-section-header" style={{ marginTop: '12px' }}>Governance</div>

                {isAuditor && (
                  <NavLink
                    to="/audit"
                    className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    title="Audit Log & Traceability"
                  >
                    <span className="nav-item-icon" aria-hidden="true">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                      </svg>
                    </span>
                    <span>Audit Log</span>
                  </NavLink>
                )}

                {isAdmin && (
                  <NavLink
                    to="/admin"
                    className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    title="User Administration & Roles"
                  >
                    <span className="nav-item-icon" aria-hidden="true">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                        <circle cx="9" cy="7" r="4" />
                        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                      </svg>
                    </span>
                    <span>Admin</span>
                  </NavLink>
                )}
              </>
            )}
          </nav>

          {/* Sidebar Footer: System Summary */}
          <div className="sidebar-footer">
            <div className="sidebar-system-summary">
              <div className="system-summary-row">
                <span>Inference:</span>
                <strong>Local GPU</strong>
              </div>
              <div className="system-summary-row">
                <span>Sandbox:</span>
                <strong>Docker Isolated</strong>
              </div>
              <div className="system-summary-row">
                <span>Network:</span>
                <strong>Air-Gapped</strong>
              </div>
            </div>
          </div>
        </aside>

        {/* Content Viewport */}
        <main className="main-content" role="main">
          {children}
        </main>
      </div>
    </div>
  );
}

function ProtectedRoute({ user, onLogout, children }) {
  if (!user) return <Navigate to="/login" replace />;
  return <Layout user={user} onLogout={onLogout}>{children}</Layout>;
}

export default function App() {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem('mrpl_user');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  return (
    <BrowserRouter>
      <RouteWatcher />
      <Routes>
        <Route path="/login" element={
          user ? <Navigate to="/" replace /> : <LoginPage onLogin={handleLogin} />
        } />
        <Route path="/" element={
          <ProtectedRoute user={user} onLogout={handleLogout}>
            <ChatPage user={user} />
          </ProtectedRoute>
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute user={user} onLogout={handleLogout}>
            <DashboardPage />
          </ProtectedRoute>
        } />
        <Route path="/documents" element={
          <ProtectedRoute user={user} onLogout={handleLogout}>
            <DocumentsPage />
          </ProtectedRoute>
        } />
        <Route path="/files" element={
          <ProtectedRoute user={user} onLogout={handleLogout}>
            <FilesPage />
          </ProtectedRoute>
        } />
        <Route path="/admin" element={
          <ProtectedRoute user={user} onLogout={handleLogout}>
            <AdminPage />
          </ProtectedRoute>
        } />
        <Route path="/audit" element={
          <ProtectedRoute user={user} onLogout={handleLogout}>
            <AuditPage />
          </ProtectedRoute>
        } />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </BrowserRouter>
  );
}
