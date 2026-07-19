(function (global) {
  "use strict";
  function boxBlur(source, width, height, channels, kernelSize) {
    const left = Math.floor((kernelSize - 1) / 2), right = kernelSize - 1 - left;
    const horizontal = new Float32Array(source.length), output = new Float32Array(source.length);
    for (let y = 0; y < height; y++) for (let c = 0; c < channels; c++) {
      let sum = 0;
      for (let x = -left; x <= right; x++) sum += source[(y * width + Math.max(0, Math.min(width - 1, x))) * channels + c];
      for (let x = 0; x < width; x++) {
        horizontal[(y * width + x) * channels + c] = sum / kernelSize;
        const removeX = Math.max(0, Math.min(width - 1, x - left));
        const addX = Math.max(0, Math.min(width - 1, x + right + 1));
        sum += source[(y * width + addX) * channels + c] - source[(y * width + removeX) * channels + c];
      }
    }
    for (let x = 0; x < width; x++) for (let c = 0; c < channels; c++) {
      let sum = 0;
      for (let y = -left; y <= right; y++) sum += horizontal[(Math.max(0, Math.min(height - 1, y)) * width + x) * channels + c];
      for (let y = 0; y < height; y++) {
        output[(y * width + x) * channels + c] = sum / kernelSize;
        const removeY = Math.max(0, Math.min(height - 1, y - left));
        const addY = Math.max(0, Math.min(height - 1, y + right + 1));
        sum += horizontal[(addY * width + x) * channels + c] - horizontal[(removeY * width + x) * channels + c];
      }
    }
    return output;
  }
  function fusion(image, foreground, background, alpha, width, height, kernelSize) {
    const count = width * height;
    const blurredAlpha = boxBlur(alpha, width, height, 1, kernelSize);
    const weightedForeground = new Float32Array(count * 3), weightedBackground = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) {
      const offset = i * 3 + c;
      weightedForeground[offset] = foreground[offset] * alpha[i];
      weightedBackground[offset] = background[offset] * (1 - alpha[i]);
    }
    const blurredFA = boxBlur(weightedForeground, width, height, 3, kernelSize);
    const blurredB1A = boxBlur(weightedBackground, width, height, 3, kernelSize);
    const estimatedForeground = new Float32Array(count * 3), estimatedBackground = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) {
      const offset = i * 3 + c, a = alpha[i], blurredA = blurredAlpha[i];
      const f = blurredFA[offset] / (blurredA + 1e-5);
      const b = blurredB1A[offset] / (1 - blurredA + 1e-5);
      estimatedBackground[offset] = b;
      estimatedForeground[offset] = Math.max(0, Math.min(1, f + a * (image[offset] - a * f - (1 - a) * b)));
    }
    return [estimatedForeground, estimatedBackground];
  }
  global.estimateForegroundPhotoRoom = function (image, alpha, width, height) {
    const first = fusion(image, image, image, alpha, width, height, 90);
    return fusion(image, first[0], first[1], alpha, width, height, 6)[0];
  };
})(typeof window !== "undefined" ? window : globalThis);
