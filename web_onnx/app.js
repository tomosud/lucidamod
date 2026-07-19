(function () {
  "use strict";

  const MODEL = "https://huggingface.co/tomohisa/lucida-web/resolve/main/lucida-web-1024-fp16.onnx?download=true";
  const EXPECTED_MODEL_BYTES = 472615213;
  const SIZE = 1024;
  const OUTPUT_PREFIX = "lucida-";

  let session = null;
  let sessionPromise = null;
  let serviceWorkerPromise = null;
  let busy = false;
  let logSequence = 0;
  let lastModelProgressLog = -5;
  let currentSourceName = "image";
  let pendingPasteFile = null;
  let pendingPasteUrl = null;

  const $ = (id) => document.getElementById(id);
  const status = $("status");
  const title = $("statusTitle");
  const detail = $("statusDetail");
  const inputViewer = $("inputViewer");
  const outputViewer = $("outputViewer");
  const inputOverlay = $("inputOverlay");
  const outputPlaceholder = $("outputPlaceholder");
  const fileInput = $("file");
  const pasteDialog = $("pasteDialog");
  const pastePreview = $("pastePreview");
  const pasteCancel = $("pasteCancel");
  const pasteProcess = $("pasteProcess");
  const inputPreview = $("inputPreview");
  const output = $("output");
  const work = $("work");
  const timing = $("timing");
  const save = $("save");
  const logOutput = $("logOutput");
  const modelProgress = $("modelProgress");

  function log(message, data) {
    const now = new Date().toLocaleTimeString("en-US", { hour12: false });
    let suffix = "";
    if (data !== undefined) {
      try { suffix = " " + JSON.stringify(data); }
      catch (_) { suffix = " " + String(data); }
    }
    const line = `[${now}] #${++logSequence} ${message}${suffix}`;
    logOutput.textContent += line + "\n";
    logOutput.scrollTop = logOutput.scrollHeight;
    console.log("[Lucida ONNX]", message, data === undefined ? "" : data);
  }

  function describeError(error) {
    return {
      type: typeof error,
      name: error && error.name ? error.name : null,
      message: error && error.message ? error.message : String(error),
      value: String(error),
      stack: error && error.stack ? error.stack : null,
      possibleCause: typeof error === "number"
        ? "Internal runtime failure. This often means GPU memory pressure or a WebGPU kernel failure."
        : null,
    };
  }

  const floatScratch = new Float32Array(1);
  const uintScratch = new Uint32Array(floatScratch.buffer);

  function float32ToFloat16Bits(value) {
    floatScratch[0] = value;
    const bits = uintScratch[0];
    const sign = (bits >>> 16) & 0x8000;
    let mantissa = (bits >>> 12) & 0x07ff;
    const exponent = (bits >>> 23) & 0xff;
    if (exponent < 103) return sign;
    if (exponent > 142) return sign | 0x7c00 | (exponent === 255 && (bits & 0x007fffff) ? 1 : 0);
    if (exponent < 113) {
      mantissa |= 0x0800;
      return sign | (mantissa >>> (114 - exponent)) + ((mantissa >>> (113 - exponent)) & 1);
    }
    return sign | ((exponent - 112) << 10) | (mantissa >>> 1) | (mantissa & 1);
  }

  function float16BitsToFloat32(bits) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction ? NaN : sign * Infinity;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }

  function memorySnapshot() {
    if (!performance.memory) return { available: false };
    const mib = (value) => Math.round(value / 1024 / 1024 * 10) / 10;
    return {
      available: true,
      usedJsHeapMiB: mib(performance.memory.usedJSHeapSize),
      totalJsHeapMiB: mib(performance.memory.totalJSHeapSize),
      heapLimitMiB: mib(performance.memory.jsHeapSizeLimit),
    };
  }

  function setState(kind, heading, description) {
    status.className = "status " + kind;
    title.textContent = heading;
    detail.textContent = description || "";
    log(`State: ${heading}`, description || "");
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "calculating";
    if (seconds < 60) return `${Math.ceil(seconds)}s`;
    return `${Math.floor(seconds / 60)}m ${Math.ceil(seconds % 60)}s`;
  }

  function baseName(name) {
    const clean = (name || "image").replace(/[/\\?%*:|"<>]/g, "_");
    return clean.replace(/\.[^.]+$/, "") || "image";
  }

  function createViewer(viewer, canvas) {
    const state = { zoom: 1, panX: 0, panY: 0, dragging: false, lastX: 0, lastY: 0 };

    function fitScale() {
      if (!canvas.width || !canvas.height) return 1;
      const rect = viewer.getBoundingClientRect();
      return Math.min(rect.width / canvas.width, rect.height / canvas.height, 1);
    }

    function apply() {
      if (!canvas.width || !canvas.height) return;
      const scale = fitScale();
      canvas.style.width = `${canvas.width * scale}px`;
      canvas.style.height = `${canvas.height * scale}px`;
      if (state.zoom <= 1) {
        state.zoom = 1;
        state.panX = 0;
        state.panY = 0;
      }
      canvas.style.transform = `translate(calc(-50% + ${state.panX}px), calc(-50% + ${state.panY}px)) scale(${state.zoom})`;
      viewer.classList.toggle("zoomed", state.zoom > 1.001);
    }

    function reset() {
      state.zoom = 1;
      state.panX = 0;
      state.panY = 0;
      apply();
    }

    viewer.addEventListener("wheel", (event) => {
      if (!canvas.width || !canvas.height) return;
      event.preventDefault();
      const oldZoom = state.zoom;
      const nextZoom = Math.min(8, Math.max(1, oldZoom * Math.exp(-event.deltaY * 0.0015)));
      const rect = viewer.getBoundingClientRect();
      const cursorX = event.clientX - rect.left - rect.width / 2;
      const cursorY = event.clientY - rect.top - rect.height / 2;
      const ratio = nextZoom / oldZoom;
      state.panX = cursorX - (cursorX - state.panX) * ratio;
      state.panY = cursorY - (cursorY - state.panY) * ratio;
      state.zoom = nextZoom;
      apply();
    }, { passive: false });

    viewer.addEventListener("pointerdown", (event) => {
      if (state.zoom <= 1 || event.button !== 0) return;
      state.dragging = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      viewer.classList.add("panning");
      viewer.setPointerCapture(event.pointerId);
    });

    viewer.addEventListener("pointermove", (event) => {
      if (!state.dragging) return;
      state.panX += event.clientX - state.lastX;
      state.panY += event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      apply();
    });

    function stopPan(event) {
      if (!state.dragging) return;
      state.dragging = false;
      viewer.classList.remove("panning");
      try { viewer.releasePointerCapture(event.pointerId); } catch (_) {}
    }
    viewer.addEventListener("pointerup", stopPan);
    viewer.addEventListener("pointercancel", stopPan);
    new ResizeObserver(apply).observe(viewer);
    return { reset, apply };
  }

  const inputPanZoom = createViewer(inputViewer, inputPreview);
  const outputPanZoom = createViewer(outputViewer, output);

  window.addEventListener("error", (event) => log("window.error", describeError(event.error || event.message)));
  window.addEventListener("unhandledrejection", (event) => log("unhandledrejection", describeError(event.reason)));

  ort.env.logLevel = "info";
  ort.env.debug = true;
  log("UI initialized", { model: MODEL, inputSize: SIZE, userAgent: navigator.userAgent });
  log("Runtime settings", { executionProvider: "webgpu", cpuFallback: false, fp16: true });
  log("WebGPU API", { available: Boolean(navigator.gpu), secureContext: window.isSecureContext });
  log("Browser memory", memorySnapshot());

  fetch(MODEL, { method: "HEAD", cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error("HTTP " + response.status);
    const bytes = Number(response.headers.get("content-length"));
    log("Model HEAD ok", { status: response.status, bytes, contentType: response.headers.get("content-type") });
  }).catch((error) => log("Model HEAD unavailable; download can still be attempted", describeError(error)));

  async function setupModelServiceWorker() {
    if (!("serviceWorker" in navigator)) throw new Error("This browser does not support Service Worker.");
    await navigator.serviceWorker.register("model-sw.js", { scope: "./" });
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) {
      if (!sessionStorage.getItem("lucida-sw-reload")) {
        sessionStorage.setItem("lucida-sw-reload", "1");
        location.reload();
        await new Promise(() => {});
      }
      throw new Error("Reload the page once to activate the model download service worker.");
    }
    sessionStorage.removeItem("lucida-sw-reload");
    log("Model download service worker ready");
  }


  function ensureModelServiceWorker() {
    if (!serviceWorkerPromise) serviceWorkerPromise = setupModelServiceWorker();
    return serviceWorkerPromise;
  }

  navigator.serviceWorker?.addEventListener("message", (event) => {
    const message = event.data;
    if (!message || message.type !== "lucida-model-progress") return;
    const total = message.total || EXPECTED_MODEL_BYTES;
    const elapsed = Math.max(message.elapsedSeconds, 0.001);
    const speed = message.received / elapsed;
    const percent = Math.min(100, message.received / total * 100);
    const eta = total > message.received ? (total - message.received) / speed : 0;
    modelProgress.hidden = false;
    modelProgress.value = percent;
    detail.textContent = `Downloading model: ${(message.received / 1024 ** 2).toFixed(1)} / ${(total / 1024 ** 2).toFixed(1)} MiB  ${percent.toFixed(1)}%  ${(speed / 1024 ** 2).toFixed(1)} MiB/s  ETA ${formatDuration(eta)}`;
    if (message.done || percent >= lastModelProgressLog + 5) {
      lastModelProgressLog = Math.floor(percent / 5) * 5;
      log(message.done ? "Model download complete" : "Model download progress", {
        percent: Number(percent.toFixed(1)), received: message.received, total,
        speedMiBs: Number((speed / 1024 ** 2).toFixed(1)), etaSeconds: Math.ceil(eta),
      });
    }
  });

  async function verifyWebGpu() {
    if (!navigator.gpu) throw new Error("WebGPU is not available. Use a recent Chrome or Edge build.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("Could not get a WebGPU adapter. Check browser and GPU settings.");
    const info = adapter.info || {};
    log("WebGPU adapter ok", {
      vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      adapterMaxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      features: Array.from(adapter.features),
    });
  }

  async function ensureSession() {
    if (session) return session;
    if (sessionPromise) return sessionPromise;
    const started = performance.now();
    sessionPromise = (async () => {
      setState("loading", "Loading model", "First use downloads about 451 MiB from Hugging Face. This may take several minutes.");
      lastModelProgressLog = -5;
      modelProgress.hidden = false;
      modelProgress.value = 0;
      await verifyWebGpu();
      await ensureModelServiceWorker();
      setState("loading", "Preparing WebGPU session", "The model file is available. Compiling GPU kernels now...");
      log("InferenceSession.create start", memorySnapshot());
      const created = await ort.InferenceSession.create(MODEL, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
        logSeverityLevel: 0,
        logVerbosityLevel: 1,
      });
      const seconds = (performance.now() - started) / 1000;
      session = created;
      modelProgress.hidden = true;
      log("InferenceSession.create ok", {
        seconds,
        inputs: session.inputNames,
        outputs: session.outputNames,
        memory: memorySnapshot(),
      });
      setState("ready", "Model ready", `Model load time: ${seconds.toFixed(1)}s. Running inference...`);
      return session;
    })().catch((error) => {
      sessionPromise = null;
      modelProgress.hidden = true;
      throw error;
    });
    return sessionPromise;
  }

  function pickImageFile(fileList) {
    return Array.from(fileList || []).find((file) => file && file.type && file.type.startsWith("image/"));
  }


  function imageFileFromClipboard(event) {
    const items = event.clipboardData && event.clipboardData.items ? Array.from(event.clipboardData.items) : [];
    const item = items.find((entry) => entry.kind === "file" && entry.type && entry.type.startsWith("image/"));
    if (!item) return null;
    const blob = item.getAsFile();
    if (!blob) return null;
    const extension = blob.type.split("/")[1] || "png";
    return new File([blob], `clipboard-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`, { type: blob.type });
  }

  function closePasteDialog(restoreStatus) {
    pasteDialog.hidden = true;
    pendingPasteFile = null;
    if (pendingPasteUrl) URL.revokeObjectURL(pendingPasteUrl);
    pendingPasteUrl = null;
    pastePreview.removeAttribute("src");
    if (restoreStatus) setState("idle", "Ready", "Drop, click, or paste an image on the input panel to begin.");
  }

  function showPasteDialog(file) {
    if (!file || busy) return;
    if (pendingPasteUrl) URL.revokeObjectURL(pendingPasteUrl);
    pendingPasteFile = file;
    pendingPasteUrl = URL.createObjectURL(file);
    pastePreview.src = pendingPasteUrl;
    pasteDialog.hidden = false;
    pasteProcess.focus();
    setState("idle", "Clipboard image ready", "Confirm the preview before processing the pasted image.");
    log("Clipboard image preview", { name: file.name, type: file.type, bytes: file.size });
  }

  document.addEventListener("paste", (event) => {
    if (busy || !pasteDialog.hidden) return;
    const file = imageFileFromClipboard(event);
    if (!file) return;
    event.preventDefault();
    showPasteDialog(file);
  });

  pasteCancel.addEventListener("click", () => closePasteDialog(true));
  pasteDialog.addEventListener("click", (event) => {
    if (event.target === pasteDialog) closePasteDialog(true);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !pasteDialog.hidden) closePasteDialog(true);
  });
  pasteProcess.addEventListener("click", () => {
    const file = pendingPasteFile;
    closePasteDialog();
    if (file) processFile(file);
  });
  inputViewer.addEventListener("click", () => { if (!busy) fileInput.click(); });
  inputViewer.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && !busy) {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => processFile(pickImageFile(fileInput.files)));

  ["dragenter", "dragover"].forEach((name) => inputViewer.addEventListener(name, (event) => {
    event.preventDefault();
    inputViewer.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => inputViewer.addEventListener(name, () => inputViewer.classList.remove("dragging")));
  inputViewer.addEventListener("drop", (event) => {
    event.preventDefault();
    if (!busy) processFile(pickImageFile(event.dataTransfer.files));
  });
  ["dragover", "drop"].forEach((name) => window.addEventListener(name, (event) => event.preventDefault()));

  async function drawInput(image) {
    inputPreview.width = image.width;
    inputPreview.height = image.height;
    inputPreview.getContext("2d").drawImage(image, 0, 0);
    inputViewer.classList.remove("empty");
    inputOverlay.querySelector("strong").textContent = "Drop an image to process";
    inputOverlay.querySelector("span").textContent = "click, drop, or paste to replace the current image";
    inputPanZoom.reset();
  }

  function clearOutput() {
    output.width = 0;
    output.height = 0;
    outputViewer.classList.add("empty");
    outputPlaceholder.hidden = false;
    outputPlaceholder.classList.remove("hidden");
    save.disabled = true;
    timing.textContent = "";
    outputPanZoom.reset();
  }

  async function processFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    if (busy) return;
    busy = true;
    currentSourceName = file.name || "image";
    save.disabled = true;
    clearOutput();
    setState("running", "Reading image", "Decoding the selected file in the browser.");
    log("Image processing start", { name: file.name, type: file.type, bytes: file.size, memory: memorySnapshot() });
    try {
      const image = await createImageBitmap(file);
      const width = image.width;
      const height = image.height;
      log("Image decode ok", { width, height });
      await drawInput(image);
      await ensureSession();

      setState("running", "Running WebGPU inference", `Resizing to ${SIZE}x${SIZE} and running the 16-bit ONNX model.`);
      const context = work.getContext("2d", { willReadFrequently: true });
      context.clearRect(0, 0, SIZE, SIZE);
      context.drawImage(image, 0, 0, SIZE, SIZE);
      const pixels = context.getImageData(0, 0, SIZE, SIZE).data;
      const plane = SIZE * SIZE;
      const data = new Uint16Array(3 * plane);
      const rgb = new Float32Array(3 * plane);
      const mean = [.485, .456, .406];
      const std = [.229, .224, .225];
      for (let i = 0; i < plane; i++) {
        const offset = i * 3;
        rgb[offset] = pixels[i * 4] / 255;
        rgb[offset + 1] = pixels[i * 4 + 1] / 255;
        rgb[offset + 2] = pixels[i * 4 + 2] / 255;
        data[i] = float32ToFloat16Bits((rgb[offset] - mean[0]) / std[0]);
        data[plane + i] = float32ToFloat16Bits((rgb[offset + 1] - mean[1]) / std[1]);
        data[2 * plane + i] = float32ToFloat16Bits((rgb[offset + 2] - mean[2]) / std[2]);
      }
      log("Preprocess complete", { tensorShape: [1, 3, SIZE, SIZE], tensorMiB: data.byteLength / 1024 / 1024 });

      const feeds = { pixel_values: new ort.Tensor("float16", data, [1, 3, SIZE, SIZE]) };
      const inferStarted = performance.now();
      log("session.run start", memorySnapshot());
      const outputs = await session.run(feeds);
      const inferSeconds = (performance.now() - inferStarted) / 1000;
      log("session.run ok", {
        seconds: inferSeconds,
        outputNames: Object.keys(outputs),
        outputDims: outputs.alpha.dims,
        outputType: outputs.alpha.type,
        memory: memorySnapshot(),
      });

      const alpha = outputs.alpha.data;
      const alphaIsBitArray = alpha instanceof Uint16Array;
      const readAlpha = alphaIsBitArray ? (index) => float16BitsToFloat32(alpha[index]) : (index) => Number(alpha[index]);
      const alphaFloat = new Float32Array(alpha.length);
      let alphaMin = Infinity;
      let alphaMax = -Infinity;
      let alphaSum = 0;
      for (let p = 0; p < alpha.length; p++) {
        const value = readAlpha(p);
        alphaFloat[p] = value;
        alphaMin = Math.min(alphaMin, value);
        alphaMax = Math.max(alphaMax, value);
        alphaSum += value;
      }
      log("Alpha stats", {
        arrayType: alpha.constructor ? alpha.constructor.name : typeof alpha,
        interpretedAsBits: alphaIsBitArray,
        min: alphaMin,
        max: alphaMax,
        mean: alphaSum / alpha.length,
        samples: Array.from({ length: 8 }, (_, i) => readAlpha(i)),
      });

      setState("running", "Refining foreground", "Applying browser-side foreground color correction.");
      const postStarted = performance.now();
      log("PhotoRoom foreground correction start", { kernelSizes: [90, 6], pixels: plane });
      const foreground = estimateForegroundPhotoRoom(rgb, alphaFloat, SIZE, SIZE);
      const postSeconds = (performance.now() - postStarted) / 1000;
      log("PhotoRoom foreground correction ok", { seconds: postSeconds, memory: memorySnapshot() });

      const mask = context.createImageData(SIZE, SIZE);
      for (let p = 0; p < plane; p++) {
        mask.data[p * 4] = Math.round(foreground[p * 3] * 255);
        mask.data[p * 4 + 1] = Math.round(foreground[p * 3 + 1] * 255);
        mask.data[p * 4 + 2] = Math.round(foreground[p * 3 + 2] * 255);
        mask.data[p * 4 + 3] = Math.max(0, Math.min(255, Math.round(alphaFloat[p] * 255)));
      }
      context.putImageData(mask, 0, 0);
      output.width = width;
      output.height = height;
      const outputContext = output.getContext("2d");
      outputContext.clearRect(0, 0, width, height);
      outputContext.drawImage(work, 0, 0, width, height);
      outputViewer.classList.remove("empty");
      outputPlaceholder.hidden = true;
      outputPlaceholder.classList.add("hidden");
      outputPanZoom.reset();
      save.disabled = false;
      timing.textContent = `Inference: ${inferSeconds.toFixed(1)}s / correction: ${postSeconds.toFixed(1)}s`;
      setState("done", "Done", "Processing finished locally in this browser.");
    } catch (error) {
      const diagnostic = describeError(error);
      log("Image processing or inference failed", { ...diagnostic, memory: memorySnapshot() });
      setState("error", session ? "Inference error" : "Model load error", diagnostic.message + (diagnostic.possibleCause ? ` / ${diagnostic.possibleCause}` : ""));
    } finally {
      busy = false;
      fileInput.value = "";
      log("Image processing end", memorySnapshot());
    }
  }

  if ("serviceWorker" in navigator) {
    setState("loading", "Preparing browser cache", "The page may refresh once to enable model download progress.");
    ensureModelServiceWorker()
      .then(() => setState("idle", "Ready", "Drop, click, or paste an image on the input panel to begin."))
      .catch((error) => {
        log("Model download service worker setup failed", describeError(error));
        setState("error", "Setup error", describeError(error).message);
      });
  }
  save.addEventListener("click", () => {
    if (save.disabled || !output.width || !output.height) return;
    output.toBlob((blob) => {
      if (!blob) return;
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(blob);
      anchor.download = `${OUTPUT_PREFIX}${baseName(currentSourceName)}.png`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
    }, "image/png");
  });
})();









