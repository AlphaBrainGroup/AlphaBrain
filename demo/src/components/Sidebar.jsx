const NAV_ITEMS = [
  { id: 'home',     label: 'Project Hub',        badge: null,
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 6.5L8 2l6 4.5V14H2V6.5z"/></svg> },
  { id: 'overview', label: 'Project Overview',   badge: null,
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/><rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/></svg> },
  { id: 'failure',  label: 'Failure Map',        badge: '4',
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="6"/><path d="M8 5v3.5l2 2"/></svg> },
  { id: 'patch',    label: 'Patch Plan',         badge: null,
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 8h10M8 3v10"/><circle cx="8" cy="8" r="6"/></svg> },
  { id: 'runner',   label: 'Iteration Runner',   badge: null,
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><polygon points="4,3 13,8 4,13"/></svg> },
  { id: 'report',   label: 'Improvement Report', badge: null,
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="2,12 5,7 8,9 11,5 14,8"/></svg> },
  { id: 'memory',   label: 'Platform Memory',    badge: null,
    icon: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><ellipse cx="8" cy="5" rx="6" ry="3"/><path d="M2 5v6a6 3 0 0 0 12 0V5"/><path d="M2 8a6 3 0 0 0 12 0"/></svg> },
];

export default function Sidebar({ active, onNav }) {
  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-mark">N</div>
        <div>
          <div className="logo-text">Nvex</div>
          <div className="logo-sub">Physical AI Orchestration</div>
        </div>
      </div>

      <div className="nav-section-label">Navigation</div>

      {NAV_ITEMS.map(item => (
        <div
          key={item.id}
          className={`nav-item${active === item.id ? ' active' : ''}`}
          onClick={() => onNav(item.id)}
        >
          {item.icon}
          {item.label}
          {item.badge && <span className="nav-badge">{item.badge}</span>}
        </div>
      ))}

      <div className="sidebar-footer">
        <div className="nav-section-label" style={{ paddingTop: 4 }}>Active Project</div>
        <div className="sidebar-project" onClick={() => onNav('overview')}>
          <div className="project-dot" />
          <div>
            <div className="project-name">LIBERO Kitchen</div>
            <div className="project-sub">ckpt_v0.7 · 62% SR</div>
          </div>
        </div>
      </div>
    </nav>
  );
}
