const PAGE_LABELS = {
  home:     'Project Hub',
  overview: 'Project Overview',
  failure:  'Failure Map',
  patch:    'Patch Plan',
  runner:   'Iteration Runner',
  report:   'Improvement Report',
  memory:   'Platform Memory',
};

export default function TopBar({ page, onNav }) {
  return (
    <header className="topbar">
      <div className="topbar-breadcrumb">
        <span>Nvex</span>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-cur">{PAGE_LABELS[page] || page}</span>
      </div>
      <div className="topbar-spacer" />
      <button className="btn btn-ghost" onClick={() => onNav('failure')}>View Failure Map</button>
      <button className="btn btn-primary" onClick={() => onNav('runner')}>▶ Run Iteration</button>
    </header>
  );
}
