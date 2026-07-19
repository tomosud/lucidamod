(function () {
  "use strict";
  const MODEL = "/models/lucida-web-1024-fp16.onnx";
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
    log("モデルHEAD失敗", describeError(error));
    meta.textContent = "モデルを確認できません: " + String(error);
    loadBtn.disabled = true;
    setState("error", "モデルがありません", "1024 FP16版ONNXを先に生成してください。");
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
    setState("loading", "モデルを読み込み中", "約448 MiBを読み込み、WebGPUセッションを作成しています。");
    log("InferenceSession.create開始", memorySnapshot());
    try {
      await verifyWebGpu();
      session = await ort.InferenceSession.create(MODEL, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
        logSeverityLevel: 0,
        logVerbosityLevel: 1,
      });
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
    } finally { busy = false; }
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
      const mean = [.485, .456, .406];
      const std = [.229, .224, .225];
      for (let i = 0; i < plane; i++) {
        data[i] = float32ToFloat16Bits((pixels[i * 4] / 255 - mean[0]) / std[0]);
        data[plane + i] = float32ToFloat16Bits((pixels[i * 4 + 1] / 255 - mean[1]) / std[1]);
        data[2 * plane + i] = float32ToFloat16Bits((pixels[i * 4 + 2] / 255 - mean[2]) / std[2]);
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
      let alphaMin = Infinity;
      let alphaMax = -Infinity;
      let alphaSum = 0;
      for (let p = 0; p < alpha.length; p++) {
        const value = readAlpha(p);
        alphaMin = Math.min(alphaMin, value);
        alphaMax = Math.max(alphaMax, value);
        alphaSum += value;
      }
      log("alpha統計", {
        arrayType: alpha.constructor ? alpha.constructor.name : typeof alpha,
        interpretedAsBits: alphaIsBitArray, min: alphaMin, max: alphaMax,
        mean: alphaSum / alpha.length, samples: Array.from({ length: 8 }, (_, i) => readAlpha(i)),
      });
      const mask = context.createImageData(SIZE, SIZE);
      for (let p = 0; p < plane; p++) {
        const value = Math.max(0, Math.min(255, Math.round(readAlpha(p) * 255)));
        mask.data[p * 4] = 255;
        mask.data[p * 4 + 1] = 255;
        mask.data[p * 4 + 2] = 255;
        mask.data[p * 4 + 3] = value;
      }
      context.putImageData(mask, 0, 0);
      output.width = width;
      output.height = height;
      const outputContext = output.getContext("2d");
      outputContext.drawImage(image, 0, 0, width, height);
      outputContext.globalCompositeOperation = "destination-in";
      outputContext.drawImage(work, 0, 0, width, height);
      outputContext.globalCompositeOperation = "source-over";
      result.style.display = "block";
      timing.textContent = `推論時間: ${seconds.toFixed(1)}秒`;
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
