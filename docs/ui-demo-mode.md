# AlphaBrain local end-to-end demo

The demo mode runs the real web UI, SQLite state, FIFO scheduler, logs, metrics,
checkpoint indexing, deployment lifecycle, authenticated WebSocket inference,
and evaluation result handling. It replaces GPU compute, model weights, and
robot simulation with deterministic local workers.

## Start on Windows

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ui_demo.ps1
```

Open <http://127.0.0.1:8100>. The demo state is isolated under
`.alphabrain-demo`; it does not reuse the normal `.alphabrain-ui` database.

The script creates `.venv-ui312` and installs UI dependencies if the environment
does not exist. It also builds the frontend when `ui/frontend/dist` is missing.

Useful options:

```powershell
# Rebuild the frontend before starting
.\scripts\run_ui_demo.ps1 -RebuildFrontend

# Use another local port or state directory
.\scripts\run_ui_demo.ps1 -Port 8200 -StateDir .\.alphabrain-demo-alt
```

## Suggested walkthrough

1. Open **Templates** and choose **Demo · Toy VLA 全流程**.
2. Submit the experiment and open its job to watch logs and metrics.
3. Open **Checkpoints** after the job completes.
4. Create a deployment from `checkpoint-10`.
5. Open **Inference Playground**, select the running deployment, and submit an instruction.
6. Open **Evaluations**, select the same checkpoint, `LIBERO`, and the quick preset.
7. Inspect the result summary and episode rows, then stop the deployment.

The gold **Demo mode** badge is always visible in the header. Demo-generated
artifacts also contain `demo: true` or `simulated: true` metadata.

## Safety boundaries

- Demo mode only binds to loopback (`127.0.0.1`, `localhost`, or `::1`).
- It never downloads or loads model weights.
- It never launches a real training or robot-simulation process.
- It uses a separate database, dataset seed, logs, and result directory.
