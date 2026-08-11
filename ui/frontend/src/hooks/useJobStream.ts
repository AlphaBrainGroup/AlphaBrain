import { useEffect, useMemo, useRef, useState } from 'react';
import { api, isRecord } from '../api/client';
import type { JobStatus, JobStreamEvent, MetricPoint } from '../api/types';

interface StreamState {
  connected: boolean;
  ended: boolean;
  logs: string[];
  metrics: MetricPoint[];
  status?: JobStatus;
}

function parseEvent(raw: string): JobStreamEvent | undefined {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return undefined;
    return parsed as unknown as JobStreamEvent;
  } catch {
    return { type: 'log', line: raw };
  }
}

function normalizeMetric(value: unknown): MetricPoint | undefined {
  if (!isRecord(value)) return undefined;
  const rawMetrics = isRecord(value.metrics) ? value.metrics : value;
  const metrics = Object.fromEntries(
    Object.entries(rawMetrics).filter(([key, metric]) =>
      typeof metric === 'number' && !['step', 'iteration', 'timestamp'].includes(key),
    ),
  ) as Record<string, number>;
  if (!Object.keys(metrics).length) return undefined;
  return {
    timestamp: typeof value.timestamp === 'string' || typeof value.timestamp === 'number' ? value.timestamp : undefined,
    step: typeof value.step === 'number' ? value.step : undefined,
    iteration: typeof value.iteration === 'number' ? value.iteration : undefined,
    phase: typeof value.phase === 'string' ? value.phase : undefined,
    metrics,
  };
}

export function useJobStream(jobId?: string): StreamState {
  const [connected, setConnected] = useState(false);
  const [ended, setEnded] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [status, setStatus] = useState<JobStatus>();
  const lastEventId = useRef<string | undefined>(undefined);

  useEffect(() => {
    setConnected(false);
    setEnded(false);
    setLogs([]);
    setMetrics([]);
    setStatus(undefined);
    lastEventId.current = undefined;
    if (!jobId) return;

    const url = new URL(api.jobs.eventUrl(jobId), window.location.origin);
    if (lastEventId.current) url.searchParams.set('cursor', lastEventId.current);
    const stream = new EventSource(url.toString(), { withCredentials: true });

    stream.onopen = () => setConnected(true);
    stream.onerror = () => setConnected(false);

    const accept = (event: MessageEvent<string>) => {
      if (event.lastEventId) lastEventId.current = event.lastEventId;
      const parsed = parseEvent(event.data);
      if (!parsed) return;
      const eventType = parsed.type || event.type;
      if (eventType === 'log') {
        const line = parsed.line
          ?? (typeof parsed.data === 'string' ? parsed.data : undefined)
          ?? (isRecord(parsed.data) && typeof parsed.data.line === 'string' ? parsed.data.line : undefined)
          ?? (isRecord(parsed) && typeof parsed.text === 'string' ? parsed.text : undefined);
        if (line) {
          const incoming = line.split(/\r?\n/).filter((item, index, values) => item || index < values.length - 1);
          setLogs((current) => [...current, ...incoming].slice(-2000));
        }
      } else if (eventType === 'metric') {
        const point = parsed.metric ?? normalizeMetric(parsed.data) ?? normalizeMetric(parsed);
        if (point) setMetrics((current) => [...current.slice(-1998), point]);
      } else if (eventType === 'status') {
        const next = parsed.status
          ?? (typeof parsed.data === 'string' ? parsed.data : undefined)
          ?? (isRecord(parsed.data) ? parsed.data.status : undefined);
        if (typeof next === 'string') setStatus(next as JobStatus);
      }
    };

    const finish = (event: MessageEvent<string>) => {
      const parsed = parseEvent(event.data);
      const next = parsed?.status
        ?? (isRecord(parsed?.data) ? parsed.data.status : undefined);
      if (typeof next === 'string') setStatus(next as JobStatus);
      setConnected(false);
      setEnded(true);
      stream.close();
    };

    stream.onmessage = accept;
    ['log', 'metric', 'status'].forEach((name) => stream.addEventListener(name, accept as EventListener));
    stream.addEventListener('end', finish as EventListener);
    return () => stream.close();
  }, [jobId]);

  return useMemo(() => ({ connected, ended, logs, metrics, status }), [connected, ended, logs, metrics, status]);
}
