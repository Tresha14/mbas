/**
 * Shared clock: estimates this browser's offset to the recorder server clock
 * (NTP-style, keeps the sample with the smallest round trip).
 * All timestamps sent to the server use clock.now() (server epoch ms).
 */
const local = () => performance.timeOrigin + performance.now();

class SharedClock {
  offsetMs = 0;
  rttMs = Infinity;

  async sync(samples = 10): Promise<{ offset: number; rtt: number }> {
    let best = { offset: 0, rtt: Infinity };
    for (let i = 0; i < samples; i++) {
      const t0 = local();
      const res = await fetch("/api/time", { cache: "no-store" });
      const t1 = local();
      const { t_ms } = (await res.json()) as { t_ms: number };
      const rtt = t1 - t0;
      if (rtt < best.rtt) best = { offset: t_ms - (t0 + t1) / 2, rtt };
    }
    this.offsetMs = best.offset;
    this.rttMs = best.rtt;
    return best;
  }

  now(): number {
    return local() + this.offsetMs;
  }

  /** Convert a performance.now() value to server time. */
  fromPerf(perfMs: number): number {
    return performance.timeOrigin + perfMs + this.offsetMs;
  }
}

export const clock = new SharedClock();
