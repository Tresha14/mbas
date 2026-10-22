// AudioWorklet: mono float input -> 16 kHz Int16 PCM blocks posted to the main thread.
// If the AudioContext already runs at 16 kHz the samples pass straight through;
// otherwise a box-filter + linear-interpolation resampler brings them to 16 kHz.
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.target = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / this.target; // input samples per output sample
    this.pos = 0; // fractional read position into the carried input buffer
    this.carry = new Float32Array(0);
    this.blockSamples = this.target / 10; // post every 100 ms
    this.out = new Int16Array(this.blockSamples);
    this.outLen = 0;
    this.outStartFrame = null; // context frame index of the first sample in `out`
    this.running = true;
    this.port.onmessage = (e) => {
      if (e.data === "stop") {
        this.flush();
        this.running = false;
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  push(sample, frame) {
    if (this.outLen === 0) this.outStartFrame = frame;
    const s = Math.max(-1, Math.min(1, sample));
    this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.outLen === this.blockSamples) this.flush();
  }

  flush() {
    if (this.outLen === 0) return;
    const pcm = this.out.slice(0, this.outLen);
    this.port.postMessage({ type: "pcm", pcm, startFrame: this.outStartFrame, sampleRate }, [pcm.buffer]);
    this.outLen = 0;
  }

  process(inputs) {
    if (!this.running) return false;
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    // downmix to mono
    const n = input[0].length;
    const mono = new Float32Array(n);
    for (let c = 0; c < input.length; c++) {
      const ch = input[c];
      for (let i = 0; i < n; i++) mono[i] += ch[i] / input.length;
    }
    if (this.ratio === 1) {
      for (let i = 0; i < n; i++) this.push(mono[i], currentFrame + i);
      return true;
    }
    // resample: smooth with a box filter of width `ratio`, then interpolate
    const buf = new Float32Array(this.carry.length + n);
    buf.set(this.carry, 0);
    buf.set(mono, this.carry.length);
    const firstFrame = currentFrame - this.carry.length;
    const w = Math.max(1, Math.round(this.ratio));
    while (this.pos + w < buf.length) {
      const i0 = Math.floor(this.pos);
      let acc = 0;
      for (let k = 0; k < w; k++) acc += buf[i0 + k];
      this.push(acc / w, firstFrame + i0);
      this.pos += this.ratio;
    }
    const keep = Math.floor(this.pos);
    this.carry = buf.slice(keep);
    this.pos -= keep;
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
