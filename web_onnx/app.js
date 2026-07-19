(function () {
  "use strict";
  const MODEL = "https://huggingface.co/tomohisa/lucida-web/resolve/main/lucida-web-1024-fp16.onnx?download=true";
  const EXPECTED_MODEL_BYTES = 472615213;
  const SIZE = 1024;
  let session = null;
  let busy = false;
  let logSequence = 0;

  const $ = (id) => document.getElementById(id);
  const loadBtn = $("loadModel");
  const meta = $("modelMeta");
  const status = $("status");
  const title = $("statusTitle");
  const detail = $("statusDetail");
  const drop = $("drop");
  const fileInput = $("file");
  const result = $("result");
  const inputPreview = $("inputPreview");
  const output = $("output");
  const work = $("work");
  const timing = $("timing");
  const logOutput = $("logOutput");
  const modelProgress = $("modelProgress");

  function log(message, data) {
    const now = new Date().toLocaleTimeString("ja-JP", { hour12: false });
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
        ? "ランタイム内部例外です。GPUメモリ不足またはWebGPUカーネル内部失敗の可能性があります。"
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
    log(`状態: ${heading}`, description || "");
  }

  window.addEventListener("error", (event) => log("window.error", describeError(event.error || event.message)));
  window.addEventListener("unhandledrejection", (event) => log("unhandledrejection", describeError(event.reason)));

  ort.env.logLevel = "info";
  ort.env.debug = true;
  log("画面初期化", { model: MODEL, inputSize: SIZE, userAgent: navigator.userAgent });
  log("実行設定", { executionProvider: "webgpu", cpuFallback: false, fp16: true });
  log("WebGPU API", { available: Boolean(navigator.gpu), secureContext: window.isSecureContext });
  log("ブラウザメモリ", memorySnapshot());

  fetch(MODEL, { method: "HEAD", cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error("HTTP " + response.status);
    const bytes = Number(response.headers.get("content-length"));
    meta.textContent = `lucida-web-1024-fp16.onnx / ${(bytes / 1024 / 1024).toFixed(1)} MiB / WebGPU`;
    log("モデルHEAD成功", { status: response.status, bytes, contentType: response.headers.get("content-type") });
  }).catch((error) => {
    meta.textContent = "lucida-web-1024-fp16.onnx / 約450.7 MiB / Hugging Face / WebGPU";
    log("モデルHEAD取得不可（DLは試行可能）", describeError(error));
  });

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "計算中";
    if (seconds < 60) return `${Math.ceil(seconds)}秒`;
    return `${Math.floor(seconds / 60)}分${Math.ceil(seconds % 60)}秒`;
  }


  let lastModelProgressLog = -5;

  async function setupModelServiceWorker() {
    if (!("serviceWorker" in navigator)) throw new Error("このブラウザはService Workerに対応していません。");
    await navigator.serviceWorker.register("model-sw.js", { scope: "./" });
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) {
      if (!sessionStorage.getItem("lucida-sw-reload")) {
        sessionStorage.setItem("lucida-sw-reload", "1");
        location.reload();
        await new Promise(() => {});
      }
      throw new Error("モデルDL用Service Workerを有効にするためページを再読み込みしてください。");
    }
    sessionStorage.removeItem("lucida-sw-reload");
    log("モデルDL Service Worker準備完了");
  }

  navigator.serviceWorker?.addEventListener("message", (event) => {
    const message = event.data;
    if (!message || message.type !== "lucida-model-progress") return;
    const elapsed = Math.max(message.elapsedSeconds, 0.001);
    const speed = message.received / elapsed;
    const percent = Math.min(100, message.received / message.total * 100);
    const eta = message.total > message.received ? (message.total - message.received) / speed : 0;
    modelProgress.hidden = false;
    modelProgress.value = percent;
    detail.textContent = `${(message.received / 1024 ** 2).toFixed(1)} / ${(message.total / 1024 ** 2).toFixed(1)} MiB  `
      + `${percent.toFixed(1)}%  ${(speed / 1024 ** 2).toFixed(1)} MiB/s  残り ${formatDuration(eta)}`;
    if (message.done || percent >= lastModelProgressLog + 5) {
      lastModelProgressLog = Math.floor(percent / 5) * 5;
      log(message.done ? "Hugging FaceモデルDL完了" : "モデルDL進捗", {
        percent: Number(percent.toFixed(1)), received: message.received, total: message.total,
        speedMiBs: Number((speed / 1024 ** 2).toFixed(1)), etaSeconds: Math.ceil(eta),
      });
    }
  });

  async function verifyWebGpu() {
    if (!navigator.gpu) throw new Error("このブラウザではWebGPU APIを利用できません。最新版のChromeまたはEdgeを使用してください。");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPUアダプターを取得できません。ブラウザまたはGPU設定を確認してください。");
    const info = adapter.info || {};
    log("WebGPUアダプター取得成功", {
      vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      adapterMaxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      features: Array.from(adapter.features),
    });
  }
  loadBtn.addEventListener("click", async () => {
    if (session || busy) return;
    busy = true;
    loadBtn.disabled = true;
    const started = performance.now();
    setState("loading", "Hugging Faceからモデルをダウンロード中", "ダウンロードを開始しています...");
    log("InferenceSession.create開始", memorySnapshot());
    try {
      await verifyWebGpu();
      await setupModelServiceWorker();
      lastModelProgressLog = -5;
      modelProgress.hidden = false;
      modelProgress.value = 0;
      setState("loading", "WebGPUセッションを構築中", "ダウンロード完了。モデルをGPU用に準備しています...");
      log("Hugging Face URLからセッション読込開始", { source: MODEL, memory: memorySnapshot() });
      session = await ort.InferenceSession.create(MODEL, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
        logSeverityLevel: 0,
        logVerbosityLevel: 1,
      });
      modelProgress.hidden = true;
      const seconds = (performance.now() - started) / 1000;
      log("InferenceSession.create成功", {
        seconds,
        inputs: session.inputNames,
        outputs: session.outputNames,
        memory: memorySnapshot(),
      });
      setState("ready", "モデル準備完了", `読込時間: ${seconds.toFixed(1)}秒。画像を選択できます。`);
      drop.classList.remove("disabled");
      loadBtn.textContent = "読込済み";
    } catch (error) {
      log("InferenceSession.create失敗", describeError(error));
      setState("error", "モデル読込エラー", describeError(error).message);
      loadBtn.disabled = false;
    } finally {
      busy = false;
      if (!session) modelProgress.hidden = true;
    }
  });

  drop.addEventListener("click", () => { if (session && !busy) fileInput.click(); });
  fileInput.addEventListener("change", () => processFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => event.preventDefault()));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    if (session && !busy) processFile(event.dataTransfer.files[0]);
  });

  async function processFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    busy = true;
    setState("running", "GPUで推論中", `${SIZE}×${SIZE}へ変換してWebGPUで処理しています。`);
    log("画像処理開始", { name: file.name, type: file.type, bytes: file.size, memory: memorySnapshot() });
    try {
      const image = await createImageBitmap(file);
      const width = image.width;
      const height = image.height;
      log("画像デコード成功", { width, height });
      inputPreview.width = width;
      inputPreview.height = height;
      inputPreview.getContext("2d").drawImage(image, 0, 0);

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
      log("前処理完了", { tensorShape: [1, 3, SIZE, SIZE], tensorMiB: data.byteLength / 1024 / 1024 });

      const feeds = { pixel_values: new ort.Tensor("float16", data, [1, 3, SIZE, SIZE]) };
      const started = performance.now();
      log("session.run開始", memorySnapshot());
      const outputs = await session.run(feeds);
      const seconds = (performance.now() - started) / 1000;
      log("session.run成功", {
        seconds,
        outputNames: Object.keys(outputs),
        outputDims: outputs.alpha.dims,
        outputType: outputs.alpha.type,
        memory: memorySnapshot(),
      });

      const alpha = outputs.alpha.data;
      const alphaIsBitArray = alpha instanceof Uint16Array;
      const readAlpha = alphaIsBitArray
        ? (index) => float16BitsToFloat32(alpha[index])
        : (index) => Number(alpha[index]);
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
      log("alpha統計", {
        arrayType: alpha.constructor ? alpha.constructor.name : typeof alpha,
        interpretedAsBits: alphaIsBitArray, min: alphaMin, max: alphaMax,
        mean: alphaSum / alpha.length, samples: Array.from({ length: 8 }, (_, i) => readAlpha(i)),
      });
      const postStarted = performance.now();
      log("PhotoRoom前景色補正開始", { kernelSizes: [90, 6], pixels: plane });
      const foreground = estimateForegroundPhotoRoom(rgb, alphaFloat, SIZE, SIZE);
      const postSeconds = (performance.now() - postStarted) / 1000;
      log("PhotoRoom前景色補正完了", { seconds: postSeconds, memory: memorySnapshot() });
      const mask = context.createImageData(SIZE, SIZE);
      for (let p = 0; p < plane; p++) {
        const value = Math.max(0, Math.min(255, Math.round(alphaFloat[p] * 255)));
        mask.data[p * 4] = Math.round(foreground[p * 3] * 255);
        mask.data[p * 4 + 1] = Math.round(foreground[p * 3 + 1] * 255);
        mask.data[p * 4 + 2] = Math.round(foreground[p * 3 + 2] * 255);
        mask.data[p * 4 + 3] = value;
      }
      context.putImageData(mask, 0, 0);
      output.width = width;
      output.height = height;
      const outputContext = output.getContext("2d");
      outputContext.clearRect(0, 0, width, height);
      outputContext.drawImage(work, 0, 0, width, height);
      result.style.display = "block";
      timing.textContent = `推論: ${seconds.toFixed(1)}秒 / 色補正: ${postSeconds.toFixed(1)}秒`;
      setState("done", "完了", "処理はブラウザ内だけで完了しました。");
    } catch (error) {
      const diagnostic = describeError(error);
      log("画像処理または推論失敗", { ...diagnostic, memory: memorySnapshot() });
      setState("error", "推論エラー", diagnostic.message + (diagnostic.possibleCause ? ` / ${diagnostic.possibleCause}` : ""));
    } finally {
      busy = false;
      fileInput.value = "";
      log("画像処理終了", memorySnapshot());
    }
  }

  $("save").addEventListener("click", () => output.toBlob((blob) => {
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = "lucida-onnx-1024-fp16-output.png";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }, "image/png"));
})();
