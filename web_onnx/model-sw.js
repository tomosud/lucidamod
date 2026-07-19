"use strict";
const MODEL_PATH = "/tomohisa/lucida-web/resolve/main/lucida-web-1024-fp16.onnx";

async function broadcast(message) {
  const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
  for (const client of windows) client.postMessage(message);
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.hostname !== "huggingface.co" || !url.pathname.includes(MODEL_PATH)) return;
  event.respondWith((async () => {
    const response = await fetch(event.request);
    if (!response.ok || !response.body) return response;
    const total = Number(response.headers.get("content-length")) || 472615213;
    const reader = response.body.getReader();
    let received = 0;
    const started = performance.now();
    const stream = new ReadableStream({
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) {
            await broadcast({ type: "lucida-model-progress", done: true, received, total,
              elapsedSeconds: (performance.now() - started) / 1000 });
            controller.close();
            return;
          }
          received += value.byteLength;
          controller.enqueue(value);
          await broadcast({ type: "lucida-model-progress", done: false, received, total,
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
  })());
});
