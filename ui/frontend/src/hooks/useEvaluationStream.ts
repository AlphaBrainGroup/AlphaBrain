import { useEffect, useMemo, useState } from 'react';
import { api, isRecord } from '../api/client';
import type { EvaluationStatus, EvaluationStreamEvent } from '../api/types';

interface EvaluationStreamState {
  connected: boolean;
  ended: boolean;
  logs: string[];
  status?: EvaluationStatus;
  progress?: number;
  phase?: string;
}

function parseEvent(raw: string): EvaluationStreamEvent | undefined {
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRecord(parsed) ? parsed as unknown as EvaluationStreamEvent : undefined;
  } catch {
    return { type: 'log', line: raw };
  }
}

export function useEvaluationStream(evaluationId?: string): EvaluationStreamState {
  const [connected, setConnected] = useState(false);
  const [ended, setEnded] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<EvaluationStatus>();
  const [progress, setProgress] = useState<number>();
  const [phase, setPhase] = useState<string>();

  useEffect(() => {
    setConnected(false);
    setEnded(false);
    setLogs([]);
    setStatus(undefined);
    setProgress(undefined);
    setPhase(undefined);
    if (!evaluationId) return;

    const url = new URL(api.evaluations.eventUrl(evaluationId), window.location.origin);
    const stream = new EventSource(url.toString(), { withCredentials: true });
    stream.onopen = () => setConnected(true);
    stream.onerror = () => setConnected(false);

    const accept = (event: MessageEvent<string>) => {
      const parsed = parseEvent(event.data);
      if (!parsed) return;
      const type = parsed.type || event.type;
      const data = isRecord(parsed.data) ? parsed.data : undefined;
      if (type === 'log') {
        const line = parsed.line
          ?? (typeof parsed.data === 'string' ? parsed.data : undefined)
          ?? (data ? String(data.line ?? data.message ?? '') : undefined);
        if (line) setLogs((current) => [...current, ...line.split(/\r?\n/)].slice(-3000));
      }
      if (type === 'status') {
        const next = parsed.status ?? (typeof parsed.data === 'string' ? parsed.data : data?.status);
        if (typeof next === 'string') setStatus(next as EvaluationStatus);
      }
      if (type === 'progress' || parsed.progress != null || data?.progress != null) {
        const next = parsed.progress ?? data?.progress;
        if (typeof next === 'number') setProgress(next);
        const nextPhase = parsed.phase ?? data?.phase;
        if (typeof nextPhase === 'string') setPhase(nextPhase);
      }
    };
    const finish = (event: MessageEvent<string>) => {
      accept(event);
      setConnected(false);
      setEnded(true);
      stream.close();
    };

    stream.onmessage = accept;
    ['log', 'status', 'progress', 'result'].forEach((name) => stream.addEventListener(name, accept as EventListener));
    stream.addEventListener('end', finish as EventListener);
    return () => stream.close();
  }, [evaluationId]);

  return useMemo(
    () => ({ connected, ended, logs, status, progress, phase }),
    [connected, ended, logs, phase, progress, status],
  );
}
