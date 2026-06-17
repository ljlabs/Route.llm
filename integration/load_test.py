"""
Load Test Engine

Gradually ramps TPS to the proxy, detects steady state, holds for a configured duration,
and generates a self-contained HTML report in output/.
"""

import asyncio
import aiohttp
import json
import time
import os
import statistics
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


@dataclass
class RequestResult:
    idx: int
    latency_ms: float
    ok: bool
    status: int
    ts: str


@dataclass
class SecondMetrics:
    second_offset: int
    target_tps: float
    actual_tps: float
    requests_completed: int
    errors: int
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return round(statistics.mean(self.latencies), 2) if self.latencies else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return round(statistics.median(self.latencies), 2) if self.latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = min(int(len(sorted_lat) * p / 100), len(sorted_lat) - 1)
        return round(sorted_lat[idx], 2)


async def send_request(session: aiohttp.ClientSession, url: str, idx: int) -> RequestResult:
    payload = {
        "model": "mock-model",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    start = time.perf_counter()
    ts = datetime.now().isoformat()
    try:
        async with session.post(url, json=payload) as resp:
            await resp.read()
            latency = (time.perf_counter() - start) * 1000
            return RequestResult(idx=idx, latency_ms=latency, ok=resp.status < 400, status=resp.status, ts=ts)
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(idx=idx, latency_ms=latency, ok=False, status=0, ts=ts)


def generate_report(
    request_results: List[RequestResult],
    second_metrics: List[SecondMetrics],
    config: dict,
    steady_tps: float,
    reached_steady: bool,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(request_results)
    ok_results = [r for r in request_results if r.ok]
    fail_results = [r for r in request_results if not r.ok]
    latencies = sorted([r.latency_ms for r in ok_results])

    total_ok = len(ok_results)
    total_fail = len(fail_results)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
    max_lat = max(latencies) if latencies else 0
    min_lat = min(latencies) if latencies else 0

    wall_elapsed = sum(1 for _ in second_metrics)
    actual_tps = total / wall_elapsed if wall_elapsed > 0 else 0
    target_tps = config.get("max_tps", 0)

    # TPS over time (per-second)
    tps_labels = json.dumps([m.second_offset for m in second_metrics])
    tps_values = json.dumps([m.actual_tps for m in second_metrics])
    tps_target = json.dumps([m.target_tps for m in second_metrics])

    # Latency per request scatter
    lat_labels = json.dumps([r.idx for r in ok_results])
    lat_values = json.dumps([round(r.latency_ms, 1) for r in ok_results])

    # Latency histogram
    if latencies:
        bucket_size = max(50, int((max_lat - min_lat) / 15)) if max_lat > min_lat else 100
        hist_buckets = {}
        for l in latencies:
            b = int(l // bucket_size) * bucket_size
            hist_buckets[b] = hist_buckets.get(b, 0) + 1
        hist_labels = json.dumps([f"{k}-{k+bucket_size}" for k in sorted(hist_buckets)])
        hist_values = json.dumps([hist_buckets[k] for k in sorted(hist_buckets)])
    else:
        hist_labels = "[]"
        hist_values = "[]"

    # Status counts
    status_counts = {}
    for r in request_results:
        s = str(r.status)
        status_counts[s] = status_counts.get(s, 0) + 1
    status_json = json.dumps(status_counts)

    # Latency over time (per-second avg)
    lat_time_labels = json.dumps([m.second_offset for m in second_metrics])
    lat_time_values = json.dumps([m.avg_latency_ms for m in second_metrics])
    lat_time_p95 = json.dumps([m.p95_latency_ms for m in second_metrics])

    expected_time = total / target_tps if target_tps > 0 else 0
    ratio = wall_elapsed / expected_time if expected_time > 0 else 0
    rate_verdict = "ACTIVE" if (target_tps > 0 and ratio >= 0.9) else "NOT ENFORCED"
    verdict_class = "active" if rate_verdict == "ACTIVE" else "off"
    fail_class = "red" if total_fail else "green"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"loadtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    css = """
  :root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a; --text: #e1e4ed;
           --dim: #8b8fa3; --green: #22c55e; --red: #ef4444; --blue: #3b82f6;
           --yellow: #eab308; --purple: #a855f7; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Cascadia Code', monospace;
          padding: 24px; line-height: 1.5; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: var(--dim); font-size: 0.85rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .stat .label { color: var(--dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
  .stat .value.green { color: var(--green); }
  .stat .value.red { color: var(--red); }
  .stat .value.blue { color: var(--blue); }
  .stat .value.yellow { color: var(--yellow); }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .chart-card h3 { font-size: 0.85rem; color: var(--dim); margin-bottom: 12px; }
  canvas { width: 100% !important; }
  .verdict { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px;
              text-align: center; margin-bottom: 24px; }
  .verdict .status { font-size: 1.8rem; font-weight: 700; }
  .verdict .status.active { color: var(--green); }
  .verdict .status.off { color: var(--red); }
  .config { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px;
            margin-bottom: 24px; font-size: 0.85rem; color: var(--dim); }
  .config b { color: var(--text); }
  @media (max-width: 768px) { .charts { grid-template-columns: 1fr; } }
"""

    js_template = """
const chartOpts = {
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: '#8b8fa3', maxTicksLimit: 15 }, grid: { color: '#2a2d3a' } },
    y: { ticks: { color: '#8b8fa3' }, grid: { color: '#2a2d3a' } }
  }
};

new Chart(document.getElementById('tpsChart'), {
  type: 'line',
  data: {
    labels: TPS_LABELS,
    datasets: [
      { label: 'Actual', data: TPS_VALUES, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.3, pointRadius: 1 },
      { label: 'Target', data: TPS_TARGET, borderColor: '#eab308', borderDash: [5,5], pointRadius: 0, borderWidth: 1 }
    ]
  },
  options: { ...chartOpts, plugins: { legend: { display: true, labels: { color: '#8b8fa3' } } },
    scales: { ...chartOpts.scales, x: { ...chartOpts.scales.x, title: { display: true, text: 'Seconds', color: '#8b8fa3' } },
              y: { ...chartOpts.scales.y, title: { display: true, text: 'TPS', color: '#8b8fa3' } } } }
});

new Chart(document.getElementById('latTimeChart'), {
  type: 'line',
  data: {
    labels: LAT_TIME_LABELS,
    datasets: [
      { label: 'Avg', data: LAT_TIME_VALUES, borderColor: '#a855f7', tension: 0.3, pointRadius: 1 },
      { label: 'P95', data: LAT_TIME_P95, borderColor: '#ef4444', borderDash: [3,3], pointRadius: 0, borderWidth: 1 }
    ]
  },
  options: { ...chartOpts, plugins: { legend: { display: true, labels: { color: '#8b8fa3' } } },
    scales: { ...chartOpts.scales, x: { ...chartOpts.scales.x, title: { display: true, text: 'Seconds', color: '#8b8fa3' } },
              y: { ...chartOpts.scales.y, title: { display: true, text: 'ms', color: '#8b8fa3' } } } }
});

new Chart(document.getElementById('latChart'), {
  type: 'scatter',
  data: {
    datasets: [{ data: LAT_LABELS.map((v,i) => ({ x: v, y: LAT_VALUES[i] })),
                  backgroundColor: '#a855f7', pointRadius: 2 }]
  },
  options: { ...chartOpts, scales: {
    x: { ...chartOpts.scales.x, title: { display: true, text: 'Request #', color: '#8b8fa3' } },
    y: { ...chartOpts.scales.y, title: { display: true, text: 'ms', color: '#8b8fa3' } }
  } }
});

new Chart(document.getElementById('histChart'), {
  type: 'bar',
  data: {
    labels: HIST_LABELS,
    datasets: [{ data: HIST_VALUES, backgroundColor: '#22c55e', borderRadius: 2 }]
  },
  options: { ...chartOpts, scales: { ...chartOpts.scales, x: { ...chartOpts.scales.x, title: { display: true, text: 'Latency (ms)', color: '#8b8fa3' } } } }
});

const statusData = STATUS_JSON;
new Chart(document.getElementById('statusChart'), {
  type: 'doughnut',
  data: {
    labels: Object.keys(statusData),
    datasets: [{ data: Object.values(statusData),
                  backgroundColor: Object.keys(statusData).map(s => s === '200' ? '#22c55e' : '#ef4444') }]
  },
  options: { responsive: true, plugins: { legend: { labels: { color: '#e1e4ed' } } } }
});
"""
    js = js_template
    js = js.replace("TPS_LABELS", tps_labels)
    js = js.replace("TPS_VALUES", tps_values)
    js = js.replace("TPS_TARGET", tps_target)
    js = js.replace("LAT_TIME_LABELS", lat_time_labels)
    js = js.replace("LAT_TIME_VALUES", lat_time_values)
    js = js.replace("LAT_TIME_P95", lat_time_p95)
    js = js.replace("LAT_LABELS", lat_labels)
    js = js.replace("LAT_VALUES", lat_values)
    js = js.replace("HIST_LABELS", hist_labels)
    js = js.replace("HIST_VALUES", hist_values)
    js = js.replace("STATUS_JSON", status_json)

    ramp_step = config.get("ramp_step", "?")
    ramp_interval = config.get("ramp_interval", "?")
    hold_duration = config.get("hold_duration", "?")
    max_tps = config.get("max_tps", "?")
    concurrency = config.get("concurrency", "?")
    proxy_url = config.get("proxy_url", "?")

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'<title>Load Test Report — {title_time}</title>\n'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>\n'
        '<style>\n' + css + '</style>\n'
        '</head>\n<body>\n'
        '<h1>Load Test Report</h1>\n'
        f'<div class="subtitle">Generated {now} &middot; {total} requests &middot; concurrency {concurrency}</div>\n'

        '<div class="config">\n'
        f'  <b>Ramp:</b> +{ramp_step} TPS every {ramp_interval}s &middot; '
        f'<b>Max:</b> {max_tps} TPS &middot; '
        f'<b>Hold:</b> {hold_duration}s &middot; '
        f'<b>Proxy:</b> {proxy_url}\n'
        '</div>\n'

        '<div class="verdict">\n'
        '  <div class="label" style="color:var(--dim);font-size:0.8rem;text-transform:uppercase;margin-bottom:8px">Rate Limiter Status</div>\n'
        f'  <div class="status {verdict_class}">{rate_verdict}</div>\n'
        f'  <div style="color:var(--dim);font-size:0.85rem;margin-top:8px">'
        f'Target: {target_tps} TPS &middot; Actual: {actual_tps:.2f} TPS &middot; Ratio: {ratio:.2f}x'
        f'{" &middot; Steady state: " + str(steady_tps) + " TPS" if reached_steady else ""}</div>\n'
        '</div>\n'

        '<div class="grid">\n'
        f'  <div class="stat"><div class="label">Total Requests</div><div class="value blue">{total}</div></div>\n'
        f'  <div class="stat"><div class="label">Successful</div><div class="value green">{total_ok}</div></div>\n'
        f'  <div class="stat"><div class="label">Failed</div><div class="value {fail_class}">{total_fail}</div></div>\n'
        f'  <div class="stat"><div class="label">Duration</div><div class="value">{wall_elapsed}s</div></div>\n'
        f'  <div class="stat"><div class="label">Actual TPS</div><div class="value yellow">{actual_tps:.2f}</div></div>\n'
        f'  <div class="stat"><div class="label">Latency p50</div><div class="value">{p50:.0f}ms</div></div>\n'
        f'  <div class="stat"><div class="label">Latency p95</div><div class="value">{p95:.0f}ms</div></div>\n'
        f'  <div class="stat"><div class="label">Latency p99</div><div class="value">{p99:.0f}ms</div></div>\n'
        f'  <div class="stat"><div class="label">Latency Max</div><div class="value red">{max_lat:.0f}ms</div></div>\n'
        '</div>\n'

        '<div class="charts">\n'
        '  <div class="chart-card"><h3>TPS Over Time</h3><canvas id="tpsChart"></canvas></div>\n'
        '  <div class="chart-card"><h3>Latency Over Time (Avg / P95)</h3><canvas id="latTimeChart"></canvas></div>\n'
        '  <div class="chart-card"><h3>Latency Per Request</h3><canvas id="latChart"></canvas></div>\n'
        '  <div class="chart-card"><h3>Latency Distribution</h3><canvas id="histChart"></canvas></div>\n'
        '  <div class="chart-card"><h3>Status Codes</h3><canvas id="statusChart"></canvas></div>\n'
        '</div>\n'
        '<script>\n' + js + '</script>\n'
        '</body>\n</html>'
    )

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Report saved: {filepath}")
    return filepath


class LoadTester:
    def __init__(
        self,
        proxy_url: str,
        ramp_step: float = 1.0,
        ramp_interval: float = 2.0,
        hold_duration: float = 60.0,
        max_tps: float = 100.0,
        concurrency: int = 50,
        num_requests: int = 0,
        target_tps: float = 0,
    ):
        self.proxy_url = proxy_url.rstrip("/") + "/v1/messages"
        self.ramp_step = ramp_step
        self.ramp_interval = ramp_interval
        self.hold_duration = hold_duration
        self.max_tps = max_tps
        self.concurrency = concurrency
        self.num_requests = num_requests
        self.target_tps = target_tps

        self.request_results: List[RequestResult] = []
        self.metrics_history: List[SecondMetrics] = []
        self._stop = False
        self._req_idx = 0

    async def run(self):
        if self.num_requests > 0:
            await self._run_simple()
        else:
            await self._run_ramp()

    async def _run_simple(self):
        """Fire N requests concurrently, report results."""
        print(f"\n  Proxy: {self.proxy_url}")
        print(f"  Target TPS: {self.target_tps if self.target_tps > 0 else 'unlimited'}")
        print(f"  Concurrency: {self.concurrency}\n")

        semaphore = asyncio.Semaphore(self.concurrency)
        connector = aiohttp.TCPConnector(limit=self.concurrency, limit_per_host=self.concurrency)
        wall_start = time.perf_counter()
        completed = 0
        failed = 0

        bar = tqdm(total=self.num_requests, desc="Requests", unit="req",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        async with aiohttp.ClientSession(connector=connector) as session:
            async def bounded_send(idx):
                nonlocal completed, failed
                async with semaphore:
                    result = await send_request(session, self.proxy_url, idx)
                elapsed = time.perf_counter() - wall_start
                cur_tps = (completed + 1) / elapsed if elapsed > 0 else 0
                completed += 1
                if not result.ok:
                    failed += 1
                bar.set_postfix({"ok": completed - failed, "fail": failed, "tps": f"{cur_tps:.1f}"})
                bar.update(1)
                return result

            tasks = [bounded_send(i + 1) for i in range(self.num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        bar.close()

        for r in results:
            if isinstance(r, Exception):
                self._req_idx += 1
                self.request_results.append(RequestResult(
                    idx=self._req_idx, latency_ms=0, ok=False, status=0,
                    ts=datetime.now().isoformat()
                ))
            else:
                self.request_results.append(r)

        wall_elapsed = time.perf_counter() - wall_start
        self._print_simple_summary(wall_elapsed)

        config = {
            "mode": "simple",
            "num_requests": self.num_requests,
            "concurrency": self.concurrency,
            "target_tps": self.target_tps,
            "proxy_url": self.proxy_url,
        }
        # Build per-second metrics for the report
        self._build_simple_metrics(wall_elapsed)
        generate_report(self.request_results, self.metrics_history, config, 0, False)

    def _build_simple_metrics(self, wall_elapsed: float):
        """Convert request results into per-second metrics for the report."""
        if not self.request_results:
            return
        t0 = datetime.fromisoformat(self.request_results[0].ts)
        buckets = {}
        for r in self.request_results:
            dt = (datetime.fromisoformat(r.ts) - t0).total_seconds()
            bucket = int(dt)
            if bucket not in buckets:
                buckets[bucket] = SecondMetrics(
                    second_offset=bucket, target_tps=self.target_tps,
                    actual_tps=0, requests_completed=0, errors=0
                )
            buckets[bucket].latencies.append(r.latency_ms)
            if r.ok:
                buckets[bucket].requests_completed += 1
            else:
                buckets[bucket].errors += 1

        for b in sorted(buckets):
            m = buckets[b]
            m.actual_tps = round(m.requests_completed, 2)
            self.metrics_history.append(m)

    def _print_simple_summary(self, wall_elapsed: float):
        total = len(self.request_results)
        ok_results = [r for r in self.request_results if r.ok]
        fail_results = [r for r in self.request_results if not r.ok]
        latencies = sorted([r.latency_ms for r in ok_results])
        actual_tps = total / wall_elapsed if wall_elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"  --- Results ---")
        print(f"  Total requests:  {total}")
        print(f"  Successful:      {len(ok_results)}")
        print(f"  Failed:          {len(fail_results)}")
        print(f"  Wall time:       {wall_elapsed:.2f}s")
        print(f"  Actual TPS:      {actual_tps:.2f}")

        if self.target_tps > 0:
            expected_time = total / self.target_tps
            ratio = wall_elapsed / expected_time if expected_time > 0 else 0
            print(f"\n  Expected min time (at {self.target_tps} TPS): {expected_time:.2f}s")
            print(f"  Time ratio (actual/expected): {ratio:.2f}x")
            if ratio >= 0.9:
                print(f"  Rate limiting:   ACTIVE (throughput throttled)")
            else:
                print(f"  Rate limiting:   Possibly not enforced")

        if latencies:
            print()
            print(f"  Latency (p50):   {latencies[len(latencies)//2]:.0f}ms")
            print(f"  Latency (p95):   {latencies[int(len(latencies)*0.95)]:.0f}ms")
            print(f"  Latency (p99):   {latencies[int(len(latencies)*0.99)]:.0f}ms")
            print(f"  Latency (max):   {max(latencies):.0f}ms")

        if fail_results:
            print(f"\n  Failed requests:")
            for f in fail_results[:5]:
                print(f"    #{f.idx}: HTTP {f.status}")

        print(f"{'='*60}\n")

    async def _run_ramp(self):
        """Ramp TPS up, detect steady state, hold."""
        print(f"Load test started")
        print(f"  Target: ramp {self.ramp_step} TPS every {self.ramp_interval}s to {self.max_tps}, hold {self.hold_duration}s")
        print(f"  Proxy: {self.proxy_url}")
        print()

        semaphore = asyncio.Semaphore(self.concurrency)
        connector = aiohttp.TCPConnector(limit=self.concurrency, limit_per_host=self.concurrency)
        steady_tps = 0.0
        reached_steady = False

        async with aiohttp.ClientSession(connector=connector) as session:
            phase = "ramp"
            target_tps = self.ramp_step
            steady_count = 0
            hold_start: Optional[float] = None
            second_offset = 0
            ramp_elapsed = 0.0

            while not self._stop:
                sec_start = time.perf_counter()

                ramp_elapsed += 1.0
                if ramp_elapsed >= self.ramp_interval and target_tps < self.max_tps:
                    target_tps = min(target_tps + self.ramp_step, self.max_tps)
                    ramp_elapsed = 0.0

                m = SecondMetrics(
                    second_offset=second_offset,
                    target_tps=target_tps,
                    actual_tps=0.0,
                    requests_completed=0,
                    errors=0,
                )

                interval = 1.0 / target_tps if target_tps > 0 else 999
                tasks = []
                t = sec_start
                while t < sec_start + 1.0 and not self._stop:
                    async def _guarded():
                        async with semaphore:
                            self._req_idx += 1
                            return await send_request(session, self.proxy_url, self._req_idx)
                    tasks.append(asyncio.create_task(_guarded()))
                    t += interval

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, Exception):
                        m.errors += 1
                        self._req_idx += 1
                        self.request_results.append(RequestResult(
                            idx=self._req_idx, latency_ms=0, ok=False, status=0,
                            ts=datetime.now().isoformat()
                        ))
                    else:
                        self.request_results.append(r)
                        m.latencies.append(r.latency_ms)
                        if r.ok:
                            m.requests_completed += 1
                        else:
                            m.errors += 1

                elapsed_sec = time.perf_counter() - sec_start
                actual = m.requests_completed / max(elapsed_sec, 0.001)
                m.actual_tps = round(actual, 2)

                self.metrics_history.append(m)

                print(f"  [{second_offset:3d}s] phase={phase:6s} target={target_tps:5.1f} "
                      f"actual={m.actual_tps:5.1f} reqs={m.requests_completed:4d} "
                      f"err={m.errors:3d} avg={m.avg_latency_ms:6.1f}ms "
                      f"p95={m.p95_latency_ms:6.1f}ms")

                if phase == "ramp":
                    if target_tps > 0 and abs(actual - target_tps) / target_tps <= 0.10:
                        steady_count += 1
                    else:
                        steady_count = 0

                    if steady_count >= 5:
                        phase = "hold"
                        hold_start = time.perf_counter()
                        steady_tps = target_tps
                        reached_steady = True
                        print(f"\n  >>> Steady state reached at {target_tps:.1f} TPS — holding for {self.hold_duration}s <<<\n")

                elif phase == "hold":
                    if hold_start and (time.perf_counter() - hold_start) >= self.hold_duration:
                        phase = "done"
                        print(f"\n  >>> Hold complete. Test finished. <<<\n")
                        break

                second_offset += 1
                wait_time = 1.0 - (time.perf_counter() - sec_start)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

        # Summary
        total = len(self.request_results)
        errs = sum(1 for r in self.request_results if not r.ok)
        all_lat = [r.latency_ms for r in self.request_results if r.ok]

        print("=" * 60)
        print(f"  SUMMARY")
        print(f"  Duration:      {second_offset}s")
        print(f"  Total requests: {total}")
        print(f"  Errors:        {errs}")
        if all_lat:
            print(f"  Avg latency:   {statistics.mean(all_lat):.1f}ms")
            print(f"  P50 latency:   {statistics.median(all_lat):.1f}ms")
            sorted_lat = sorted(all_lat)
            print(f"  P95 latency:   {sorted_lat[int(len(sorted_lat)*0.95)]:.1f}ms")
            print(f"  P99 latency:   {sorted_lat[int(len(sorted_lat)*0.99)]:.1f}ms")
        if reached_steady:
            print(f"  Steady state:  {steady_tps:.1f} TPS")
        else:
            print(f"  Did not reach steady state")
        print("=" * 60)

        # Generate HTML report
        config = {
            "mode": "ramp",
            "ramp_step": self.ramp_step,
            "ramp_interval": self.ramp_interval,
            "hold_duration": self.hold_duration,
            "max_tps": self.max_tps,
            "concurrency": self.concurrency,
            "proxy_url": self.proxy_url,
        }
        generate_report(self.request_results, self.metrics_history, config, steady_tps, reached_steady)

    def stop(self):
        self._stop = True


def main():
    parser = argparse.ArgumentParser(description="Load Test Engine for LLM Proxy")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8000", help="Proxy server URL")
    parser.add_argument("--requests", type=int, default=0, help="Total requests to send (simple mode)")
    parser.add_argument("--tps", type=float, default=0, help="Target TPS limit to verify (simple mode)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent connections")
    parser.add_argument("--ramp-step", type=float, default=1.0, help="TPS increase per ramp interval (ramp mode)")
    parser.add_argument("--ramp-interval", type=float, default=2.0, help="Seconds between ramp steps (ramp mode)")
    parser.add_argument("--hold-duration", type=float, default=60.0, help="Seconds to hold steady state (ramp mode)")
    parser.add_argument("--max-tps", type=float, default=100.0, help="Maximum TPS target (ramp mode)")
    args = parser.parse_args()

    tester = LoadTester(
        proxy_url=args.proxy_url,
        ramp_step=args.ramp_step,
        ramp_interval=args.ramp_interval,
        hold_duration=args.hold_duration,
        max_tps=args.max_tps,
        concurrency=args.concurrency,
        num_requests=args.requests,
        target_tps=args.tps,
    )
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
