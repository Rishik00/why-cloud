const ARTIFACT_PATH = /^\/a\/([a-z0-9]+(?:-[a-z0-9]+)*)\/?$/;
const API_ARTIFACT_PATH = /^\/api\/artifacts\/([a-z0-9]+(?:-[a-z0-9]+)*)\/?$/;
const MAX_SLUG_LENGTH = 80;
const MAX_UPLOAD_BYTES = 95 * 1024 * 1024;

function jsonError(message: string, status: number, extraHeaders?: HeadersInit): Response {
  return Response.json(
    { error: message },
    {
      status,
      headers: {
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        ...extraHeaders,
      },
    },
  );
}

function artifactTitle(slug: string, storedTitle?: string, titleEncoding?: string): string {
  if (storedTitle?.trim()) {
    if (titleEncoding === "uri") {
      try {
        return decodeURIComponent(storedTitle).trim();
      } catch {
        // Fall through to the slug-derived title when metadata is malformed.
      }
    } else {
      return storedTitle.trim();
    }
  }
  return slug.split("-").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function responseHeaders(object: R2Object, slug: string): Headers {
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("content-type", "application/pdf");
  headers.set("content-disposition", `inline; filename="${slug}.pdf"`);
  headers.set("cache-control", "public, max-age=3600, must-revalidate");
  headers.set("etag", object.httpEtag);
  headers.set("last-modified", object.uploaded.toUTCString());
  headers.set("accept-ranges", "bytes");
  headers.set("content-security-policy", "sandbox; default-src 'none'");
  headers.set("cross-origin-resource-policy", "same-origin");
  headers.set("referrer-policy", "no-referrer");
  headers.set("x-frame-options", "DENY");
  headers.set("x-content-type-options", "nosniff");
  return headers;
}

function rangeNotSatisfiable(object: R2Object, slug: string): Response {
  const headers = responseHeaders(object, slug);
  headers.set("content-range", `bytes */${object.size}`);
  headers.set("content-length", "0");
  return new Response(null, { status: 416, headers });
}

function isSatisfiableRange(value: string, size: number): boolean {
  const match = /^bytes=(\d*)-(\d*)$/.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return false;

  if (!match[1]) {
    const suffix = Number(match[2]);
    return Number.isSafeInteger(suffix) && suffix > 0;
  }

  const start = Number(match[1]);
  if (!Number.isSafeInteger(start) || start >= size) return false;
  if (!match[2]) return true;

  const end = Number(match[2]);
  return Number.isSafeInteger(end) && end >= start;
}

function applyContentRange(headers: Headers, object: R2Object): boolean {
  const range = object.range;
  if (!range) return false;

  let offset: number;
  let length: number;

  if ("suffix" in range && typeof range.suffix === "number") {
    length = Math.min(range.suffix, object.size);
    offset = object.size - length;
  } else {
    offset = "offset" in range && typeof range.offset === "number" ? range.offset : 0;
    length = "length" in range && typeof range.length === "number"
      ? range.length
      : object.size - offset;
  }

  headers.set("content-range", `bytes ${offset}-${offset + length - 1}/${object.size}`);
  headers.set("content-length", String(length));
  return true;
}

function preconditionStatus(request: Request): 304 | 412 {
  return request.headers.has("if-match") || request.headers.has("if-unmodified-since")
    ? 412
    : 304;
}

async function inspectPdfStream(body: ReadableStream<Uint8Array>): Promise<{
  isPdf: boolean;
  uploadBody: ReadableStream<Uint8Array>;
}> {
  const [inspectionBody, uploadBody] = body.tee();
  const reader = inspectionBody.getReader();
  const signature = new Uint8Array(5);
  let offset = 0;

  try {
    while (offset < signature.byteLength) {
      const { done, value } = await reader.read();
      if (done) break;
      const length = Math.min(value.byteLength, signature.byteLength - offset);
      signature.set(value.subarray(0, length), offset);
      offset += length;
    }
  } finally {
    await reader.cancel();
  }

  const isPdf = offset === signature.byteLength
    && signature[0] === 0x25
    && signature[1] === 0x50
    && signature[2] === 0x44
    && signature[3] === 0x46
    && signature[4] === 0x2d;
  return { isPdf, uploadBody };
}

async function serveArtifact(request: Request, env: Env, slug: string): Promise<Response> {
  const key = `artifacts/${slug}/${slug}.pdf`;

  if (request.method === "HEAD") {
    const object = await env.BUCKET.head(key);
    if (!object) return jsonError("Artifact not found", 404);

    const headers = responseHeaders(object, slug);
    headers.set("content-length", String(object.size));
    return new Response(null, { status: 200, headers });
  }

  const rangeHeader = request.headers.get("range");
  if (rangeHeader) {
    const metadata = await env.BUCKET.head(key);
    if (!metadata) return jsonError("Artifact not found", 404);
    if (!isSatisfiableRange(rangeHeader, metadata.size)) {
      return rangeNotSatisfiable(metadata, slug);
    }
  }

  const object = await env.BUCKET.get(key, {
    onlyIf: request.headers,
    range: request.headers,
  });

  if (!object) return jsonError("Artifact not found", 404);

  const headers = responseHeaders(object, slug);
  if (!("body" in object)) {
    return new Response(null, { status: preconditionStatus(request), headers });
  }

  const partial = request.headers.has("range") && applyContentRange(headers, object);
  if (!partial) headers.set("content-length", String(object.size));

  return new Response(object.body, {
    status: partial ? 206 : 200,
    headers,
  });
}

async function listArtifacts(env: Env): Promise<Response> {
  const objects: R2Object[] = [];
  let cursor: string | undefined;
  do {
    const result = await env.BUCKET.list({
      prefix: "artifacts/",
      limit: 1000,
      include: ["customMetadata"],
      ...(cursor ? { cursor } : {}),
    });
    objects.push(...result.objects);
    cursor = result.truncated ? result.cursor : undefined;
  } while (cursor);

  const artifacts = objects.flatMap((object) => {
    const match = /^artifacts\/([a-z0-9]+(?:-[a-z0-9]+)*)\/\1\.pdf$/.exec(object.key);
    if (!match) return [];
    const slug = match[1];
    if (slug.length > MAX_SLUG_LENGTH) return [];
    return [{
      slug,
      title: artifactTitle(
        slug,
        object.customMetadata?.title,
        object.customMetadata?.["title-encoding"],
      ),
      size: object.size,
      uploaded: object.uploaded.toISOString(),
      url: `/a/${slug}`,
    }];
  }).sort((a, b) => b.uploaded.localeCompare(a.uploaded));

  return Response.json({ artifacts }, {
    headers: {
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
  });
}

async function uploadArtifact(request: Request, env: Env, slug: string): Promise<Response> {
  const contentType = request.headers.get("content-type")?.split(";", 1)[0].toLowerCase();
  if (contentType !== "application/pdf") {
    return jsonError("Choose a PDF file", 415);
  }

  const contentLength = Number(request.headers.get("content-length"));
  if (!Number.isSafeInteger(contentLength) || contentLength <= 0) {
    return jsonError("The PDF is empty or its size is unavailable", 400);
  }
  if (contentLength > MAX_UPLOAD_BYTES) {
    return jsonError("PDFs must be smaller than 95 MB", 413);
  }
  if (!request.body) return jsonError("The PDF is empty", 400);

  const { isPdf, uploadBody } = await inspectPdfStream(request.body);
  if (!isPdf) {
    await uploadBody.cancel();
    return jsonError("The uploaded file does not have a valid PDF signature", 415);
  }

  const encodedTitle = request.headers.get("x-artifact-title");
  let decodedTitle = "";
  if (encodedTitle) {
    try {
      decodedTitle = decodeURIComponent(encodedTitle);
    } catch {
      await uploadBody.cancel();
      return jsonError("The artifact title is malformed", 400);
    }
  }
  const safeTitle = decodedTitle.replace(/[\u0000-\u001f\u007f]/g, " ").trim();
  const title = safeTitle.slice(0, 160) || artifactTitle(slug);
  const key = `artifacts/${slug}/${slug}.pdf`;
  const fixedLengthBody = new FixedLengthStream(contentLength);
  const pipePromise = uploadBody.pipeTo(fixedLengthBody.writable);
  const objectPromise = env.BUCKET.put(key, fixedLengthBody.readable, {
    onlyIf: { etagDoesNotMatch: "*" },
    httpMetadata: {
      contentType: "application/pdf",
      contentDisposition: `inline; filename="${slug}.pdf"`,
      cacheControl: "public, max-age=3600, must-revalidate",
    },
    customMetadata: {
      title: encodeURIComponent(title),
      "title-encoding": "uri",
    },
  });
  const [object] = await Promise.all([objectPromise, pipePromise]);

  if (!object) return jsonError("That artifact slug already exists", 409);

  const url = new URL(`/a/${slug}`, request.url).toString();
  console.log(JSON.stringify({ event: "artifact_uploaded", key, size: object.size }));
  return Response.json({ artifact: { slug, title, size: object.size, url } }, {
    status: 201,
    headers: {
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
  });
}

export default {
  async fetch(request, env): Promise<Response> {
    try {
      const url = new URL(request.url);

      if (url.pathname === "/health") {
        if (request.method !== "GET" && request.method !== "HEAD") {
          return jsonError("Method not allowed", 405, { allow: "GET, HEAD" });
        }
        const headers = {
          "cache-control": "no-store",
          "x-content-type-options": "nosniff",
        };
        return request.method === "HEAD"
          ? new Response(null, { status: 200, headers })
          : Response.json({ ok: true }, { headers });
      }

      if (url.pathname === "/api/artifacts") {
        if (request.method !== "GET") {
          return jsonError("Method not allowed", 405, { allow: "GET" });
        }
        return await listArtifacts(env);
      }

      const apiMatch = API_ARTIFACT_PATH.exec(url.pathname);
      if (apiMatch && apiMatch[1].length <= MAX_SLUG_LENGTH) {
        if (request.method !== "PUT") {
          return jsonError("Method not allowed", 405, { allow: "PUT" });
        }
        return await uploadArtifact(request, env, apiMatch[1]);
      }

      const artifactMatch = ARTIFACT_PATH.exec(url.pathname);
      if (!artifactMatch || artifactMatch[1].length > MAX_SLUG_LENGTH) {
        return jsonError("Not found", 404);
      }

      if (request.method !== "GET" && request.method !== "HEAD") {
        return jsonError("Method not allowed", 405, { allow: "GET, HEAD" });
      }

      return await serveArtifact(request, env, artifactMatch[1]);
    } catch (error) {
      console.error(JSON.stringify({
        event: "request_failed",
        method: request.method,
        path: new URL(request.url).pathname,
        error: error instanceof Error ? error.message : String(error),
      }));
      return jsonError("Internal server error", 500);
    }
  },
} satisfies ExportedHandler<Env>;
