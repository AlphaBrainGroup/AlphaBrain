# AlphaBrain Research Console frontend

React + TypeScript frontend for the AlphaBrain local or laboratory research console.

```bash
cd ui/frontend
npm install
npm run dev
```

The development server listens on `0.0.0.0:5173` and proxies `/api` to
`http://127.0.0.1:8000`. Copy `.env.example` to `.env.local` to select a different
backend. The production build is emitted to `ui/frontend/dist`:

```bash
npm test
npm run build
```

All server calls use `/api/v1`, cookies are sent with credentials, mutating calls
forward the `csrf_token` or `alphabrain_csrf` cookie as `X-CSRF-Token`, and job
logs/metrics are consumed from the job SSE endpoint.
