import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from '../api/client';
import { MrplLogo } from '../App';
import HeroSection, { MRPL_BRAND_COLORS } from '../components/HeroSection';

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'MRPL Sovereign AI | Sign In';
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const data = await auth.login(username, password);
      localStorage.setItem('mrpl_token', data.token);
      localStorage.setItem('mrpl_user', JSON.stringify(data.user));
      onLogin(data.user);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Login authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <HeroSection
        colors={MRPL_BRAND_COLORS}
        distortion={0.7}
        swirl={0.5}
        speed={0.9}
      >
        <div className="login-hero-layout">
          {/* Left Hero Overview */}
          <div className="login-hero-info">
            <div className="login-hero-badge">
              <span className="sparkle">✦</span>
              <span>MRPL · ONGC GROUP ENTERPRISE</span>
            </div>

            <h1 className="login-hero-title">
              Sovereign Industrial AI for <span>MRPL Refinery</span>
            </h1>

            <p className="login-hero-desc">
              Confidential, air-gapped on-premise agentic intelligence platform engineered for refinery operations, engineering calculations, P&amp;ID compliance, and corporate governance.
            </p>

            <div className="login-hero-features">
              <div className="login-hero-feature-item">
                <span className="login-hero-feature-icon">🛡️</span>
                <div className="login-hero-feature-text">
                  <strong>Zero Cloud Egress</strong>
                  Hardened Network Seal with local Ollama inference
                </div>
              </div>

              <div className="login-hero-feature-item">
                <span className="login-hero-feature-icon">⚙️</span>
                <div className="login-hero-feature-text">
                  <strong>Refinery Vector Corpus</strong>
                  21,328 chunks indexed across 185 refinery assets
                </div>
              </div>

              <div className="login-hero-feature-item">
                <span className="login-hero-feature-icon">🔒</span>
                <div className="login-hero-feature-text">
                  <strong>Clearance Authorization</strong>
                  4-Tier RBAC with strict database-level filtering
                </div>
              </div>

              <div className="login-hero-feature-item">
                <span className="login-hero-feature-icon">🐳</span>
                <div className="login-hero-feature-text">
                  <strong>Docker Sandbox</strong>
                  Networkless, isolated Python execution container
                </div>
              </div>
            </div>
          </div>

          {/* Right Login Card */}
          <div className="login-card">
            <div className="login-logo">
              <div style={{ display: 'inline-flex', marginBottom: '12px' }}>
                <MrplLogo size={48} />
              </div>
              <h1>Mangalore Refinery and Petrochemicals Limited</h1>
              <p style={{ fontWeight: 600, color: 'var(--brand-green)', marginTop: '2px' }}>
                Sovereign AI Engineering Workbench
              </p>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                A Subsidiary of Oil and Natural Gas Corporation Limited
              </p>
            </div>

            {/* Demo Role Selection (Requirement 17) */}
            <div style={{ marginBottom: '16px', padding: '10px', background: 'var(--bg-secondary)', borderRadius: '6px', border: '1px solid var(--border-default)' }}>
              <div style={{ fontSize: '0.68rem', fontWeight: 750, color: 'var(--brand-olive)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '3px' }}>
                DEMO ROLE SELECTION
              </div>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: '0 0 8px 0', lineHeight: 1.35 }}>
                Select an authorized persona to test RBAC clearance and capabilities:
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '5px' }}>
                {[
                  { role: 'Engineer', user: 'engineer', clearance: 'CONFIDENTIAL' },
                  { role: 'Reviewer', user: 'reviewer', clearance: 'CONFIDENTIAL' },
                  { role: 'Viewer', user: 'viewer', clearance: 'INTERNAL' },
                  { role: 'Admin', user: 'admin', clearance: 'RESTRICTED' },
                ].map((d) => (
                  <button
                    key={d.user}
                    type="button"
                    className={`btn btn-sm ${username === d.user ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ fontSize: '0.7rem', padding: '5px 3px', textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '2px', alignItems: 'center' }}
                    onClick={() => {
                      setUsername(d.user);
                      setPassword('1234567890');
                    }}
                  >
                    <span style={{ fontWeight: 600 }}>{d.role}</span>
                    <span style={{ fontSize: '0.6rem', opacity: 0.85 }}>{d.clearance}</span>
                  </button>
                ))}
              </div>
            </div>

            <form className="login-form" onSubmit={handleSubmit}>
              {error && (
                <div className="login-error" role="alert">
                  {error}
                </div>
              )}

              <div className="form-group">
                <label htmlFor="username">Operator Username</label>
                <input
                  id="username"
                  className="input"
                  type="text"
                  placeholder="Enter your authorized username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoFocus
                  autoComplete="username"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="password">Security Password</label>
                <input
                  id="password"
                  className="input"
                  type="password"
                  placeholder="Enter account password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-lg w-full"
                disabled={loading}
                style={{ marginTop: '6px' }}
              >
                {loading ? (
                  <>
                    <div className="spinner" /> Authenticating Session...
                  </>
                ) : (
                  'Sign In to Sovereign Workbench'
                )}
              </button>
            </form>

            <div style={{ textAlign: 'center', marginTop: '18px', paddingTop: '12px', borderTop: '1px solid var(--border-default)' }}>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '6px', marginBottom: '4px' }}>
                <span className="badge badge-success" style={{ fontSize: '0.6rem' }}>ON-PREM</span>
                <span className="badge badge-neutral" style={{ fontSize: '0.6rem' }}>ZERO EGRESS</span>
                <span className="badge badge-neutral" style={{ fontSize: '0.6rem' }}>AIR-GAPPED</span>
              </div>
              <p style={{ fontSize: '0.66rem', color: 'var(--text-muted)', margin: 0 }}>
                Strictly for authorized MRPL operational and engineering personnel.
              </p>
            </div>
          </div>
        </div>
      </HeroSection>
    </div>
  );
}

