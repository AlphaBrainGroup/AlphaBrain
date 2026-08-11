import { useQuery } from '@tanstack/react-query';
import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { ApiError, api } from './api/client';
import { AppShell } from './components/AppShell';
import { AsyncState } from './components/AsyncState';
import { DashboardPage } from './pages/DashboardPage';
import { LoginPage } from './pages/LoginPage';
import { SetupPage } from './pages/SetupPage';

const ExperimentBuilderPage = lazy(() => import('./pages/ExperimentBuilderPage').then((module) => ({ default: module.ExperimentBuilderPage })));
const ExperimentsPage = lazy(() => import('./pages/ExperimentsPage').then((module) => ({ default: module.ExperimentsPage })));
const ExperimentDetailPage = lazy(() => import('./pages/ExperimentDetailPage').then((module) => ({ default: module.ExperimentDetailPage })));
const JobsPage = lazy(() => import('./pages/JobsPage').then((module) => ({ default: module.JobsPage })));
const WorkloadsPage = lazy(() => import('./pages/WorkloadsPage').then((module) => ({ default: module.WorkloadsPage })));
const JobDetailPage = lazy(() => import('./pages/JobDetailPage').then((module) => ({ default: module.JobDetailPage })));
const TemplatesPage = lazy(() => import('./pages/TemplatesPage').then((module) => ({ default: module.TemplatesPage })));
const CheckpointsPage = lazy(() => import('./pages/CheckpointsPage').then((module) => ({ default: module.CheckpointsPage })));
const CheckpointDetailPage = lazy(() => import('./pages/CheckpointDetailPage').then((module) => ({ default: module.CheckpointDetailPage })));
const DeploymentsPage = lazy(() => import('./pages/DeploymentsPage').then((module) => ({ default: module.DeploymentsPage })));
const DeploymentDetailPage = lazy(() => import('./pages/DeploymentDetailPage').then((module) => ({ default: module.DeploymentDetailPage })));
const PlaygroundPage = lazy(() => import('./pages/PlaygroundPage').then((module) => ({ default: module.PlaygroundPage })));
const EvaluationsPage = lazy(() => import('./pages/EvaluationsPage').then((module) => ({ default: module.EvaluationsPage })));
const EvaluationDetailPage = lazy(() => import('./pages/EvaluationDetailPage').then((module) => ({ default: module.EvaluationDetailPage })));
const EvaluationGroupDetailPage = lazy(() => import('./pages/EvaluationGroupDetailPage').then((module) => ({ default: module.EvaluationGroupDetailPage })));
const EvaluationComparePage = lazy(() => import('./pages/EvaluationComparePage').then((module) => ({ default: module.EvaluationComparePage })));
const SettingsPage = lazy(() => import('./pages/SettingsPage').then((module) => ({ default: module.SettingsPage })));
const UsersPage = lazy(() => import('./pages/UsersPage').then((module) => ({ default: module.UsersPage })));
const AuditPage = lazy(() => import('./pages/AuditPage').then((module) => ({ default: module.AuditPage })));
const ResourcesPage = lazy(() => import('./pages/ResourcesPage').then((module) => ({ default: module.ResourcesPage })));
const DatasetsPage = lazy(() => import('./pages/DatasetsPage').then((module) => ({ default: module.DatasetsPage })));
const RegistryPage = lazy(() => import('./pages/RegistryPage').then((module) => ({ default: module.RegistryPage })));
const ReferenceResultsPage = lazy(() => import('./pages/ReferenceResultsPage').then((module) => ({ default: module.ReferenceResultsPage })));

function RequireAuth() {
  const location = useLocation();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me, retry: false });
  if (me.isLoading) return <div className="center-state"><AsyncState loading>{null}</AsyncState></div>;
  if (me.error instanceof ApiError && [401, 403].includes(me.error.status)) {
    return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  }
  if (me.error) return <div className="center-state"><AsyncState error={me.error} onRetry={() => void me.refetch()}>{null}</AsyncState></div>;
  return <AppShell />;
}

export function App() {
  const setup = useQuery({ queryKey: ['setup-status'], queryFn: api.setup.status, retry: false });
  if (setup.isLoading) return <div className="center-state"><AsyncState loading>{null}</AsyncState></div>;
  if (setup.error) return <div className="center-state"><AsyncState error={setup.error} onRetry={() => void setup.refetch()}>{null}</AsyncState></div>;
  if (!setup.data?.configured) return <SetupPage initialMode={setup.data?.mode ?? 'personal'} />;

  return (
    <Suspense fallback={<div className="center-state"><AsyncState loading>{null}</AsyncState></div>}>
      <Routes>
        <Route path="/login" element={setup.data.mode === 'personal' ? <Navigate to="/" replace /> : <LoginPage />} />
        <Route element={<RequireAuth />}>
          <Route index element={<DashboardPage />} />
          <Route path="experiments/new" element={<ExperimentBuilderPage />} />
          <Route path="experiments" element={<ExperimentsPage />} />
          <Route path="experiments/:experimentId" element={<ExperimentDetailPage />} />
          <Route path="jobs" element={<JobsPage />} />
          <Route path="workloads" element={<WorkloadsPage />} />
          <Route path="jobs/:jobId" element={<JobDetailPage />} />
          <Route path="templates" element={<TemplatesPage />} />
          <Route path="checkpoints" element={<CheckpointsPage />} />
          <Route path="checkpoints/:checkpointId" element={<CheckpointDetailPage />} />
          <Route path="resources" element={<ResourcesPage />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="registry" element={<RegistryPage />} />
          <Route path="deployments" element={<DeploymentsPage />} />
          <Route path="deployments/:deploymentId" element={<DeploymentDetailPage />} />
          <Route path="playground" element={<PlaygroundPage />} />
          <Route path="evaluations" element={<EvaluationsPage />} />
          <Route path="evaluations/compare" element={<EvaluationComparePage />} />
          <Route path="evaluations/:evaluationId" element={<EvaluationDetailPage />} />
          <Route path="evaluation-groups/:groupId" element={<EvaluationGroupDetailPage />} />
          <Route path="reference-results" element={<ReferenceResultsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="settings/users" element={<UsersPage />} />
          <Route path="settings/audit" element={<AuditPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
