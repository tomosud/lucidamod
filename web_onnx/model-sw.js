"use strict";
const MODEL_PATH = "/tomohisa/lucida-web/resolve/main/lucida-web-1024-fp16.onnx";
const MODEL_CACHE = "lucida-model-v1";
const EXPECTED_MODEL_BYTES = 472615213;

async function broadcast(message) {
  const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
  for (const client of windows) client.postMessage(message);
}

function progressResponse(response, source) {
  if (!response.body) return response;
  const total = Number(response.headers.get("content-length")) || EXPECTED_MODEL_BYTES;
  const reader = response.body.getReader();
  let received = 0;
  const started = performance.now();
  const stream = new ReadableStream({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          await broadcast({ type: "lucida-model-progress", source, done: true, received, total,
            elapsedSeconds: (performance.now() - started) / 1000 });
          controller.close();
          return;
        }
        received += value.byteLength;
        controller.enqueue(value);
        await broadcast({ type: "lucida-model-progress", source, done: false, received, total,
          elapsedSeconds: (performance.now() - started) / 1000 });
      } catch (error) {
        controller.error(error);
      }
    },
    cancel(reason) { reader.cancel(reason); },
  });
  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.hostname !== "huggingface.co" || !url.pathname.includes(MODEL_PATH)) return;

  let cacheWrite = Promise.resolve();
  const responsePromise = (async () => {
    const cache = await caches.open(MODEL_CACHE);
    const cached = await cache.match(event.request, { ignoreSearch: true });
    if (cached) {
      await broadcast({ type: "lucida-model-cache", status: "hit" });
      return progressResponse(cached, "cache");
    }

    await broadcast({ type: "lucida-model-cache", status: "miss" });
    const response = await fetch(event.request);
    if (!response.ok || !response.body) return response;
    cacheWrite = cache.put(event.request, response.clone())
      .then(() => broadcast({ type: "lucida-model-cache", status: "stored" }))
      .catch((error) => broadcast({ type: "lucida-model-cache", status: "error", message: String(error) }));
    return progressResponse(response, "network");
  })();

  event.respondWith(responsePromise);
  event.waitUntil(responsePromise.then(() => cacheWrite));
});