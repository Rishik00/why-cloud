---
name: why-cloud
description: Operate and maintain the Folio personal research library on Cloudflare Workers and R2. Use when an agent needs to author or compile a Folio artifact, upload LaTeX/PDF files with Wrangler, verify a public reading link, change the Folio frontend or Worker, or deploy the project safely.
---

# Operate Folio

Work from the repository root. Read `README.md`, `wrangler.jsonc`, and the files relevant to the requested change before acting.

## Preserve these invariants

- Keep the R2 bucket private.
- Expose only PDFs matching `artifacts/<slug>/<slug>.pdf` through `/a/<slug>`.
- Restrict slugs to lowercase letters, numbers, and internal hyphens, with at most 80 characters.
- Keep `.env`, `.dev.vars`, `.tools/`, `output/`, and `tmp/` out of Git.
- Never print, read back, or commit secret values.
- Use the `BUCKET` binding inside the Worker; do not call the Cloudflare REST API from Worker code.
- Stream PDF bodies and preserve byte-range support.
- Treat an existing artifact key as immutable unless the user explicitly asks to replace it.

## Build artifacts

1. Create or update `artifacts/<slug>.md` with an H1 title and optional bold subtitle. Treat the filename stem as the slug, output filename, and R2 key; require `^[a-z0-9]+(?:-[a-z0-9]+)*$` and at most 80 characters.
2. Run the focused build:

   ```powershell
   pwsh -NoProfile -File scripts/build-artifacts.ps1 -Sources artifacts/<slug>.md
   ```

3. Inspect `output/pdf/<slug>.pdf` visually and check `output/pdf/<slug>.log` for overflow, missing glyphs, and compilation warnings.
4. Keep generated files out of Git unless the user explicitly requests otherwise.

## Run locally

- Run `npm run dev` to serve `http://localhost:8787` with simulated R2 data under `.wrangler/`.
- Run `npx wrangler dev --remote` only when the task explicitly requires the real bucket.
- Keep `ADMIN_TOKEN` in `.dev.vars` for local browser uploads. Never read the value back into output or logs.

## Upload artifacts with Wrangler

1. Confirm authentication with `npx wrangler whoami`.
2. Check whether the destination PDF key already exists:

   ```powershell
   New-Item -ItemType Directory -Force tmp | Out-Null
   npx wrangler r2 object get folio/artifacts/<slug>/<slug>.pdf --file tmp/<slug>-existing.pdf --remote
   ```

   Treat a not-found error as an available key. Stop if the download succeeds unless replacement is explicit.
3. Upload the LaTeX source:

   ```powershell
   npx wrangler r2 object put folio/artifacts/<slug>/<slug>.tex --file output/tex/<slug>.tex --content-type "application/x-tex; charset=utf-8" --remote
   ```

4. Upload the PDF:

   ```powershell
   npx wrangler r2 object put folio/artifacts/<slug>/<slug>.pdf --file output/pdf/<slug>.pdf --content-type application/pdf --content-disposition 'inline; filename="<slug>.pdf"' --cache-control "public, max-age=3600" --remote
   ```

5. Do not upload Markdown source unless the user requests it.

## Use the browser upload

- Open the Folio root, choose a PDF, enter its title and slug, and provide `ADMIN_TOKEN` in the upload-key field.
- Expect `201` for a new artifact, `409` for an existing slug, `413` above 95 MB, and `415` for a non-PDF body.
- Clear the upload key after success. Never obtain it by reading `.dev.vars` or another secret store.

## Verify every upload

1. Download the remote PDF:

   ```powershell
   npx wrangler r2 object get folio/artifacts/<slug>/<slug>.pdf --file tmp/<slug>.pdf --remote
   ```

2. Compare its SHA-256 hash and byte length with the local PDF:

   ```powershell
   Get-FileHash tmp/<slug>.pdf -Algorithm SHA256
   Get-FileHash output/pdf/<slug>.pdf -Algorithm SHA256
   ```

3. Confirm `/api/artifacts` includes the slug.
4. Set `$base = "https://why-cloud.why-cloud.workers.dev"`, then confirm `/a/<slug>` returns `200`, `Content-Type: application/pdf`, and `Accept-Ranges: bytes`.
5. Send `curl.exe -sI -H "Range: bytes=0-1023" "$base/a/<slug>"` and require `206` with a correct `Content-Range`.
6. Report the stable public URL and verification result.

## Change or deploy the application

1. Keep frontend code in `frontend/` and Worker code in `backend/`.
2. Validate untrusted JSON before using it and render text with `textContent`, not `innerHTML`.
3. Keep secrets in `.dev.vars` locally and set production secrets with `wrangler secret put`.
4. Preserve `onlyIf: { etagDoesNotMatch: "*" }` on R2 uploads to prevent silent replacement.
5. Regenerate binding types after configuration changes:

   ```powershell
   npm run types
   ```

6. Run `npm run typecheck`, `npx wrangler deploy --dry-run`, and `npm audit` before deployment.
7. Deploy only when the user asks for a production change.
