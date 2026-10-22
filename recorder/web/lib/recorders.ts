import { api, putChunk, type TrackName } from "./api";
import { clock } from "./clock";

/** Runs async jobs strictly in order (start -> chunk 0 -> chunk 1 -> ... -> stop). */
class SerialQueue {
  private tail: Promise<void> = Promise.resolve();
  error: Error | null = null;
  pending = 0;

  constructor(private onError: (e: Error) => void) {}

  push(job: () => Promise<unknown>): void {
    this.pending++;
    this.tail = this.tail.then(async () => {
      try {
        if (!this.error) await job();
      } catch (e) {
        this.error = e as Error;
        this.onError(this.error);
      } finally {
        this.pending--;
      }
    });
  }

  drain(): Promise<void> {
    return this.tail;
  }
}

export interface RecorderStatus {
  track: TrackName;
  state: "idle" | "recording" | "stopping" | "done" | "error";
  chunks: number;
  bytes: number;
  pending: number;
  error?: string;
}

type Listener = (s: RecorderStatus) => void;

abstract class BaseRecorder {
  protected queue: SerialQueue;
  protected seq = 0;
  status: RecorderStatus;

  constructor(
    protected code: string,
    readonly track: TrackName,
    protected onStatus: Listener,
  ) {
    this.status = { track, state: "idle", chunks: 0, bytes: 0, pending: 0 };
    this.queue = new SerialQueue((e) => this.update({ state: "error", error: e.message }));
  }

  protected update(p: Partial<RecorderStatus>) {
    this.status = { ...this.status, ...p, pending: this.queue.pending };
    this.onStatus(this.status);
  }

  protected enqueueChunk(data: Blob | ArrayBuffer, size: number) {
    const seq = this.seq++;
    this.queue.push(async () => {
      await putChunk(this.code, this.track, seq, data);
      this.update({ chunks: seq + 1, bytes: this.status.bytes + size });
    });
    this.update({});
  }

  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

/** Records a video MediaStreamTrack to WebM, uploading 1-second chunks. */
export class VideoTrackRecorder extends BaseRecorder {
  private rec: MediaRecorder | null = null;
  private stopped: Promise<void> | null = null;

  constructor(
    code: string,
    track: TrackName,
    private mediaTrack: MediaStreamTrack,
    onStatus: Listener,
    private bitrate = 2_500_000,
  ) {
    super(code, track, onStatus);
  }

  async start(): Promise<void> {
    const mime = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm"].find((m) =>
      MediaRecorder.isTypeSupported(m),
    );
    if (!mime) throw new Error("This browser cannot record WebM video (use Chrome or Edge).");
    const rec = new MediaRecorder(new MediaStream([this.mediaTrack]), { mimeType: mime, videoBitsPerSecond: this.bitrate });
    this.rec = rec;
    const settings = this.mediaTrack.getSettings();
    this.stopped = new Promise<void>((resolve) => {
      rec.onstop = () => {
        const end = clock.now();
        const total = this.seq;
        this.queue.push(() => api.trackStop(this.code, this.track, end, total));
        this.update({ state: "stopping" });
        this.queue.drain().then(() => {
          if (!this.queue.error) this.update({ state: "done" });
          resolve();
        });
      };
    });
    await new Promise<void>((resolve, reject) => {
      rec.onstart = () => {
        const start = clock.now();
        this.queue.push(() =>
          api.trackStart(this.code, this.track, {
            start_ms: start,
            mime,
            settings: { width: settings.width, height: settings.height, frameRate: settings.frameRate, deviceId: settings.deviceId },
          }),
        );
        this.update({ state: "recording" });
        resolve();
      };
      rec.onerror = (e) => reject(new Error(`MediaRecorder error: ${(e as ErrorEvent).message ?? "unknown"}`));
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) this.enqueueChunk(e.data, e.data.size);
      };
      rec.start(1000);
    });
  }

  /** Resolves when the recorder stopped by itself (e.g. the user ended screen sharing). */
  get finished(): Promise<void> {
    return this.stopped ?? Promise.resolve();
  }

  async stop(): Promise<void> {
    if (this.rec && this.rec.state !== "inactive") this.rec.stop();
    await this.finished;
  }
}

/** Records a microphone track as 16 kHz mono PCM16, uploading 1-second chunks to a WAV on the server. */
export class PcmRecorder extends BaseRecorder {
  static readonly SR = 16000;
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private parts: Int16Array[] = [];
  private partSamples = 0;
  private startMs: number | null = null;
  private lastBlockStartMs = 0;
  private lastBlockLen = 0;
  private startedPromise: Promise<void> | null = null;

  constructor(
    code: string,
    track: TrackName,
    private mediaTrack: MediaStreamTrack,
    onStatus: Listener,
  ) {
    super(code, track, onStatus);
  }

  private async makeContext(): Promise<AudioContext> {
    // Prefer a 16 kHz context (browser resamples with a proper filter); fall back to the
    // device rate and let the worklet resample (needed on Firefox).
    const stream = new MediaStream([this.mediaTrack]);
    for (const rate of [PcmRecorder.SR, undefined]) {
      const ctx = rate ? new AudioContext({ sampleRate: rate }) : new AudioContext();
      try {
        await ctx.audioWorklet.addModule("/pcm-worklet.js");
        this.source = ctx.createMediaStreamSource(stream);
        return ctx;
      } catch (e) {
        await ctx.close();
        if (!rate) throw e;
      }
    }
    throw new Error("unreachable");
  }

  async start(): Promise<void> {
    const ctx = await this.makeContext();
    this.ctx = ctx;
    await ctx.resume();
    const node = new AudioWorkletNode(ctx, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      processorOptions: { targetRate: PcmRecorder.SR },
    });
    this.node = node;
    const mute = ctx.createGain();
    mute.gain.value = 0;
    this.source!.connect(node).connect(mute).connect(ctx.destination);

    this.startedPromise = new Promise<void>((resolve) => {
      node.port.onmessage = (e: MessageEvent) => {
        const msg = e.data;
        if (msg.type !== "pcm") return;
        // Server time of this block's first sample = now - how long ago that frame was (context clock).
        // Start and end both use this mapping, so constant capture latency cancels out and
        // check_session's drift measures real audio-clock drift / dropped samples.
        const blockStart = clock.now() - (ctx.currentTime - msg.startFrame / msg.sampleRate) * 1000;
        this.lastBlockStartMs = blockStart;
        this.lastBlockLen = (msg.pcm as Int16Array).length;
        if (this.startMs === null) {
          this.startMs = blockStart;
          const start = this.startMs;
          this.queue.push(() =>
            api.trackStart(this.code, this.track, {
              start_ms: start,
              mime: "audio/wav;codecs=pcm16",
              sample_rate: PcmRecorder.SR,
              settings: { context_rate: ctx.sampleRate, ...this.mediaTrack.getSettings() },
            }),
          );
          this.update({ state: "recording" });
          resolve();
        }
        this.parts.push(msg.pcm as Int16Array);
        this.partSamples += (msg.pcm as Int16Array).length;
        if (this.partSamples >= PcmRecorder.SR) this.flush();
      };
    });
    await this.startedPromise;
  }

  private flush() {
    if (this.partSamples === 0) return;
    const out = new Int16Array(this.partSamples);
    let o = 0;
    for (const p of this.parts) {
      out.set(p, o);
      o += p.length;
    }
    this.parts = [];
    this.partSamples = 0;
    this.enqueueChunk(out.buffer, out.byteLength);
  }

  async stop(): Promise<void> {
    if (!this.node || !this.ctx) return;
    const node = this.node;
    await new Promise<void>((resolve) => {
      const prev = node.port.onmessage;
      node.port.onmessage = (e: MessageEvent) => {
        if (e.data.type === "stopped") resolve();
        else prev?.call(node.port, e);
      };
      node.port.postMessage("stop");
    });
    this.flush();
    const end = this.lastBlockStartMs + (this.lastBlockLen / PcmRecorder.SR) * 1000;
    const total = this.seq;
    this.queue.push(() => api.trackStop(this.code, this.track, end, total));
    this.update({ state: "stopping" });
    this.source?.disconnect();
    node.disconnect();
    await this.ctx.close();
    this.node = null;
    await this.queue.drain();
    if (!this.queue.error) this.update({ state: "done" });
  }
}
