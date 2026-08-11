import { useEffect, useMemo, useRef, useState } from 'react';
import { api, isRecord } from '../api/client';
import type { DeploymentStatus, DeploymentStreamEvent } from '../api/types';

interface DeploymentStreamState {
  connected: boolean;
  ended: boolean;
  logs: string[];
  status?: DeploymentStatus;
}

function parseEvent(raw: string): DeploymentStreamEvent | undefined {
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRecord(parsed) ? parsed as unknown as DeploymentStreamEvent : undefined;
  } catch {
    return { type: 'log', line: raw };
  }
}

export function useDeploymentStream(deploymentId?: string): DeploymentStreamState {
  const [connected, setConnected] = useState(false);
  const [ended, setEnded] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<DeploymentStatus>();
  const lastEventId = useRef<string | undefined>(undefined);

  useEffect(() => {
    setConnected(false);
    setEnded(false);
    setLogs([]);
    setStatus(undefined);
    lastEventId.current = undefined;
    if (!deploymentId) return;

    const url = new URL(api.deployments.eventUrl(deploymentId), window.location.origin);
    const stream = new EventSource(url.toString(), { withCredentials: true });
    stream.onopen = () => setConnected(true);
    stream.onerror = () => setConnected(false);

    const accept = (event: MessageEvent<string>) => {
      if (event.lastEventId) lastEventId.current = event.lastEventId;
      const parsed = parseEvent(event.data);
      if (!parsed) return;
      const type = parsed.type || event.type;
      if (type === 'log') {
        const line = parsed.line
          ?? (typeof parsed.data === 'string' ? parsed.data : undefined)
          ?? (isRecord(parsed.data) ? String(parsed.data.line ?? parsed.data.message ?? '') : undefined);
        if (line) setLogs((current) => [...current, ...line.split(/\r?\n/)].slice(-2000));
      }
      if (type === 'status') {
        const next = parsed.status
          ?? (typeof parsed.data === 'string' ? parsed.data : undefined)
          ?? (isRecord(parsed.data) ? parsed.data.status : undefined);
        if (typeof next === 'string') setStatus(next as DeploymentStatus);
      }
    };

    const finish = (event: MessageEvent<string>) => {
      accept(event);
      setConnected(false);
      setEnded(true);
      stream.close();
    };

    stream.onmessage = accept;
    ['log', 'status'].forEach((name) => stream.addEventListener(name, accept as EventListener));
    stream.addEventListener('end', finish as EventListener);
    return () => stream.close();
  }, [deploymentId]);

  return useMemo(() => ({ connected, ended, logs, status }), [connected, ended, logs, status]);
}
