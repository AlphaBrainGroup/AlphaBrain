import { useState } from 'react';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Home from './pages/Home';
import ProjectOverview from './pages/ProjectOverview';
import FailureMap from './pages/FailureMap';
import PatchPlan from './pages/PatchPlan';
import IterationRunner from './pages/IterationRunner';
import ImprovementReport from './pages/ImprovementReport';
import PlatformMemory from './pages/PlatformMemory';
import { NvexRuntimeProvider } from './data/NvexRuntimeContext';

const PAGES = {
  home:     Home,
  overview: ProjectOverview,
  failure:  FailureMap,
  patch:    PatchPlan,
  runner:   IterationRunner,
  report:   ImprovementReport,
  memory:   PlatformMemory,
};

export default function App() {
  const [page, setPage] = useState('home');
  const PageComponent = PAGES[page] || Home;

  return (
    <NvexRuntimeProvider>
      <>
        <div className="bg-layer">
          <div className="bg-dot-grid" />
          <div className="bg-glow bg-glow-1" />
          <div className="bg-glow bg-glow-2" />
        </div>
        <div className="app-shell">
          <Sidebar active={page} onNav={setPage} />
          <div className="main">
            <TopBar page={page} onNav={setPage} />
            <div className="content">
              <PageComponent onNav={setPage} />
            </div>
          </div>
        </div>
      </>
    </NvexRuntimeProvider>
  );
}
