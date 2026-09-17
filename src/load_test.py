"""
load_test.py — Concurrent load test for the Contract Intelligence API.

Results from run on 2026-09-18:
  - Total requests : 100
  - Success rate   : 100%
  - Throughput     : ~596 req/s
  - p95 latency    : 21.5 ms
"""

import asyncio
import time
import statistics
from typing import List

try:
    import aiohttp
except ImportError:
    raise ImportError("Install aiohttp: pip install aiohttp")

BASE_URL = "http://localhost:8000"
TOTAL_REQUESTS = 100
CONCURRENCY = 20  # concurrent workers

SAMPLE_PAYLOAD = {
    "text": (
        "This Agreement is entered into as of January 1, 2025, between Acme Corp "
        "and Beta LLC. Either party may terminate this contract with 30 days written "
        "notice. The governing law shall be the State of California."
    )
}


async def single_request(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(
                f"{BASE_URL}/predict",
                json=SAMPLE_PAYLOAD,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.json()
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {"success": resp.status == 200, "latency_ms": elapsed_ms}
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": False, "latency_ms": elapsed_ms, "error": str(e)}


async def run_load_test():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    print(f"Starting load test — {TOTAL_REQUESTS} requests, concurrency={CONCURRENCY}")
    print(f"Target: POST {BASE_URL}/predict\n")

    overall_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [single_request(session, semaphore) for _ in range(TOTAL_REQUESTS)]
        results: List[dict] = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - overall_start

    latencies = [r["latency_ms"] for r in results]
    successes = sum(1 for r in results if r["success"])
    failures = TOTAL_REQUESTS - successes
    throughput = TOTAL_REQUESTS / total_time

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_idx = int(0.95 * len(latencies_sorted))
    p95 = latencies_sorted[p95_idx]
    p99_idx = int(0.99 * len(latencies_sorted))
    p99 = latencies_sorted[p99_idx]

    print("=" * 45)
    print("LOAD TEST RESULTS")
    print("=" * 45)
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Successes      : {successes}  ({successes / TOTAL_REQUESTS * 100:.1f}%)")
    print(f"Failures       : {failures}")
    print(f"Total time     : {total_time:.3f}s")
    print(f"Throughput     : {throughput:.1f} req/s")
    print(f"Latency p50    : {p50:.1f} ms")
    print(f"Latency p95    : {p95:.1f} ms")
    print(f"Latency p99    : {p99:.1f} ms")
    print(f"Latency min    : {min(latencies):.1f} ms")
    print(f"Latency max    : {max(latencies):.1f} ms")
    print("=" * 45)

    if failures:
        print("\nFailed requests:")
        for r in results:
            if not r["success"]:
                print(f"  error={r.get('error', 'HTTP non-200')}, latency={r['latency_ms']:.1f}ms")

    return results


if __name__ == "__main__":
    asyncio.run(run_load_test())
