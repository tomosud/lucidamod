(function () {
  "use strict";
  const MODEL = "/models/lucida-web-512-fp32.onnx";
  const SIZE = 512;
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
        ? "WASM内部例外です。メモリ不足または実行カーネル内部失敗の可能性があります。"
        : null,
    };
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

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.logLevel = "info";
  ort.env.debug = true;
  log("画面初期化", { model: MODEL, inputSize: SIZE, userAgent: navigator.userAgent });
  log("WASM設定", { numThreads: ort.env.wasm.numThreads, proxy: ort.env.wasm.proxy });
  log("ブラウザメモリ", memorySnapshot());

  fetch(MODEL, { method: "HEAD", cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error("HTTP " + response.status);
    const bytes = Number(response.headers.get("content-length"));
    meta.textContent = `lucida-web-512-fp32.onnx / ${(bytes / 1024 / 1024).toFixed(1)} MiB / WASM CPU`;
    log("モデルHEAD成功", { status: response.status, bytes, contentType: response.headers.get("content-type") });
  }).catch((error) => {
    log("モデルHEAD失敗", describeError(error));
    meta.textContent = "モデルを確認できません: " + String(error);
    loadBtn.disabled = true;
    setState("error", "モデルがありません", "512版ONNXエクスポートを先に実行してください。");
  });

  loadBtn.addEventListener("click", async () => {
    if (session || busy) return;
    busy = true;
    loadBtn.disabled = true;
    const started = performance.now();
    setState("loading", "モデルを読み込み中", "約890 MiBを読み込み、WASMセッションを作成しています。");
    log("InferenceSession.create開始", memorySnapshot());
    try {
      session = await ort.InferenceSession.create(MODEL, {
        executionProviders: ["wasm"],
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
    setState("running", "ブラウザで推論中", `${SIZE}×${SIZE}へ変換してWASM CPUで処理しています。`);
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
      const data = new Float32Array(3 * plane);
      const mean = [.485, .456, .406];
      const std = [.229, .224, .225];
      for (let i = 0; i < plane; i++) {
        data[i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
        data[plane + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
        data[2 * plane + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
      }
      log("前処理完了", { tensorShape: [1, 3, SIZE, SIZE], tensorMiB: data.byteLength / 1024 / 1024 });

      const feeds = { pixel_values: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]) };
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
      const mask = context.createImageData(SIZE, SIZE);
      for (let p = 0; p < plane; p++) {
        const value = Math.max(0, Math.min(255, Math.round(alpha[p] * 255)));
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
    anchor.download = "lucida-onnx-512-output.png";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }, "image/png"));
})();
