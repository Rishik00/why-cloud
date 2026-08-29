# Folio

Folio is a small personal research library built on Cloudflare Workers and R2. Upload a PDF through the browser or Wrangler, keep the bucket private, and share stable reading links such as `/a/on-policy-distillation`.

[Open the live Folio](https://why-cloud.why-cloud.workers.dev/)

The frontend is deliberately plain HTML, CSS, and JavaScript. The backend is one TypeScript Worker with an R2 binding; there is no database or application server to maintain.

## Requirements

- A Cloudflare account with Workers and R2 enabled
- Node.js 20 or newer and npm
- Wrangler authentication (`npx wrangler login`)
- PowerShell 7 available as `pwsh` for the artifact build script
- Pandoc and Tectonic when compiling Markdown into LaTeX and PDF

Pandoc and Tectonic may be installed on `PATH`. On Windows, the build script also discovers binaries under `.tools/pandoc/` and `.tools/tectonic/tectonic.exe`; `.tools/` is intentionally excluded from Git.

## Project layout

```text
backend/                 Cloudflare Worker and R2 routes
frontend/                Static Folio interface and security headers
artifacts/               Long-form Markdown source documents
latex/                   Shared LaTeX presentation layer
scripts/                 Artifact build tooling
output/                  Generated LaTeX/PDF files (ignored)
wrangler.jsonc           Worker, static assets, and R2 configuration
worker-configuration.d.ts  Generated binding types
SKILL.md                 Instructions for coding agents operating Folio
```

## Setup

1. Install dependencies:

   ```powershell
   npm install
   ```

2. Authenticate Wrangler and create the private bucket:

   ```powershell
   npx wrangler login
   npx wrangler r2 bucket create folio
   ```

   If you choose another bucket name, update `bucket_name` in `wrangler.jsonc`.

3. Generate binding types and check the Worker:

   ```powershell
   npm run types
   npm run typecheck
   ```

4. Configure the browser upload key for local development:

   ```powershell
   Copy-Item .dev.vars.example .dev.vars
   ```

   `.dev.vars` must contain `ADMIN_TOKEN=<long random passphrase>`. This is the same secret name used in production. Never commit `.dev.vars`.

5. Start the local Worker and frontend:

   ```powershell
   npm run dev
   ```

   This serves `http://localhost:8787` against a local simulated R2 bucket under `.wrangler/`. Local uploads do not enter the production `folio` bucket. Use `npx wrangler dev --remote` only when you intentionally want local code to access the real bucket.

6. Deploy the Worker, then store the production upload key:

   ```powershell
   npm run deploy
   npm run secret:admin
   ```

Wrangler prompts for `ADMIN_TOKEN` without writing it to the repository. The deployed Worker receives private R2 access through the `BUCKET` binding; R2 access keys are not required by the app.

## Build an artifact

Place a Markdown document in `artifacts/` with an H1 title and optional bold subtitle, then run:

```powershell
npm run build:artifacts
```

This rebuilds every Markdown document in `artifacts/`. The filename stem becomes the output filename and artifact slug.

To build one document:

```powershell
pwsh -NoProfile -File scripts/build-artifacts.ps1 `
  -Sources artifacts/on-policy-distillation.md
```

Generated files appear under `output/tex/` and `output/pdf/`.

## Upload with Wrangler

Folio discovers PDFs stored with this exact key shape:

```text
artifacts/<slug>/<slug>.pdf
```

`wrangler r2 object put` overwrites an existing object without prompting. Check the PDF key before uploading:

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
npx wrangler r2 object get `
  folio/artifacts/on-policy-distillation/on-policy-distillation.pdf `
  --file tmp/on-policy-distillation-existing.pdf `
  --remote
```

A not-found error means the key is free. If the download succeeds, stop unless replacement was explicitly requested.

Upload the generated source and PDF directly to R2:

```powershell
npx wrangler r2 object put `
  folio/artifacts/on-policy-distillation/on-policy-distillation.tex `
  --file output/tex/on-policy-distillation.tex `
  --content-type "application/x-tex; charset=utf-8" `
  --remote

npx wrangler r2 object put `
  folio/artifacts/on-policy-distillation/on-policy-distillation.pdf `
  --file output/pdf/on-policy-distillation.pdf `
  --content-type application/pdf `
  --content-disposition 'inline; filename="on-policy-distillation.pdf"' `
  --cache-control "public, max-age=3600" `
  --remote
```

The stable reading link is then:

```text
https://<worker-name>.<workers-subdomain>.workers.dev/a/on-policy-distillation
```

The browser upload form offers the same PDF workflow using `ADMIN_TOKEN`. The key is cleared after a successful upload and is not written to browser storage.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | `GET` | Folio interface |
| `/api/artifacts` | `GET` | List published PDF artifacts |
| `/api/artifacts/:slug` | `PUT` | Upload a new PDF with a bearer admin token |
| `/a/:slug` | `GET`, `HEAD` | Stream an inline PDF with range support |
| `/health` | `GET`, `HEAD` | Worker health check |

Only PDF objects matching the expected artifact key are publicly routed. LaTeX sources and every other R2 object remain private.

## Security notes

- Keep the R2 bucket private.
- Store production secrets with `wrangler secret put`, never in `wrangler.jsonc`.
- Treat `ADMIN_TOKEN` as a password and rotate it if exposed.
- PDF uploads are limited to an enforced 95 MB stream length, checked for a PDF signature, and created only when the slug does not already exist.
- Artifact slugs are restricted to lowercase letters, numbers, and internal hyphens, up to 80 characters.
- Upload secrets are compared as fixed-size SHA-256 digests using a timing-safe comparison.
- The UI renders R2 metadata with DOM text nodes rather than HTML injection.
- This is a single-owner tool. Add an identity-aware access layer before using it as a multi-user service.

## Verification

Before deploying changes, run:

```powershell
npm run types
npm run typecheck
npx wrangler deploy --dry-run
npm audit
```

After deployment, verify `/health`, `/api/artifacts`, a full PDF response, and a byte-range response from `/a/<slug>`.
