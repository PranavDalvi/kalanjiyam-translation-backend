#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Sample texts of varying lengths for translation load testing
SAMPLE_TEXTS = {
    "short": "Artificial intelligence is transforming the world.",
    "medium": "The development of natural language processing systems has enabled computers to understand, interpret, and generate human languages. Machine translation models have particularly bridged the communication gap between different cultures and regions across the globe.",
    "long": (
        "Education is a powerful catalyst for social and economic development. By providing individuals "
        "with knowledge, critical thinking skills, and opportunities, it empowers them to break the cycle "
        "of poverty and contribute meaningfully to their communities. Accessible and quality education fosters "
        "innovation, builds stronger institutions, and promotes understanding across cultures. Investing in "
        "education is not just a moral obligation, but a strategic decision that benefits society as a whole. "
        "When every child has the chance to learn, we build a foundation for a more prosperous, equitable, "
        "and sustainable future. Therefore, governments and organizations worldwide must prioritize educational "
        "reforms and ensure that learning resources are available to all, regardless of background or socioeconomic status."
    )
}

def send_translation_request(url, text, model_name, source_lang, target_lang, batch_size):
    payload = {
        "text": text,
        "model_name": model_name,
        "source_language": source_lang,
        "target_language": target_lang,
        "batch_size": batch_size,
        "gpu_id": 0
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/translate/text",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            translated_text = res_json.get("text", "")
            end_time = time.perf_counter()
            return {
                "success": True,
                "latency": end_time - start_time,
                "input_char_len": len(text),
                "output_char_len": len(translated_text),
                "error": None
            }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "success": False,
            "latency": end_time - start_time,
            "input_char_len": len(text),
            "output_char_len": 0,
            "error": str(e)
        }

def run_load_test(url, concurrency, num_requests, text_key, model_name, source_lang, target_lang, batch_size):
    text = SAMPLE_TEXTS.get(text_key, SAMPLE_TEXTS["medium"])
    print("=" * 60)
    print("                 TRANSLATION LOAD TESTER")
    print("=" * 60)
    print(f"Target URL:         {url}")
    print(f"Endpoint:           /translate/text")
    print(f"Model Name:         {model_name}")
    print(f"Source Language:    {source_lang}")
    print(f"Target Language:    {target_lang}")
    print(f"Concurrency level:  {concurrency} threads")
    print(f"Total Requests:     {num_requests}")
    print(f"Text Payload Type:  {text_key} ({len(text)} chars)")
    print(f"Batch Size:         {batch_size}")
    print("-" * 60)
    print("Starting load test...")
    
    start_test_time = time.perf_counter()
    
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_translation_request, url, text, model_name, source_lang, target_lang, batch_size)
            for _ in range(num_requests)
        ]
        
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            if completed % max(1, num_requests // 10) == 0 or completed == num_requests:
                success_count = sum(1 for r in results if r["success"])
                print(f"Progress: {completed}/{num_requests} requests finished (Success: {success_count}, Failed: {completed - success_count})")

    end_test_time = time.perf_counter()
    total_duration = end_test_time - start_test_time
    
    # Calculate statistics
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    success_rate = (len(successes) / num_requests) * 100
    latencies = [r["latency"] for r in successes]
    
    print("\n" + "=" * 60)
    print("                      RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Duration:         {total_duration:.2f} seconds")
    print(f"Total Requests:         {num_requests}")
    print(f"Successful Requests:    {len(successes)}")
    print(f"Failed Requests:        {len(failures)}")
    print(f"Success Rate:           {success_rate:.2f}%")
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        # Percentiles
        latencies_sorted = sorted(latencies)
        p90 = percentiles(latencies_sorted, 0.90)
        p95 = percentiles(latencies_sorted, 0.95)
        p99 = percentiles(latencies_sorted, 0.99)
        
        # Throughput
        req_per_sec = len(successes) / total_duration
        total_input_chars = sum(r["input_char_len"] for r in successes)
        chars_per_sec = total_input_chars / total_duration
        
        print(f"\nThroughput Statistics:")
        print(f"  Requests/sec:         {req_per_sec:.2f}")
        print(f"  Input Characters/sec: {chars_per_sec:.2f}")
        print(f"  Total Input Chars:    {total_input_chars}")
        
        print(f"\nLatency Statistics (seconds):")
        print(f"  Min Latency:          {min_latency:.4f}s")
        print(f"  Average Latency:      {avg_latency:.4f}s")
        print(f"  Median (P50) Latency: {median_latency:.4f}s")
        print(f"  Max Latency:          {max_latency:.4f}s")
        print(f"  Std Dev:              {std_dev:.4f}s")
        print(f"  90th Percentile (P90): {p90:.4f}s")
        print(f"  95th Percentile (P95): {p95:.4f}s")
        print(f"  99th Percentile (P99): {p99:.4f}s")
    else:
        print("\nNo successful requests to calculate latency stats.")
        
    if failures:
        print(f"\nFailure Examples:")
        err_counts = {}
        for f in failures:
            err_counts[f["error"]] = err_counts.get(f["error"], 0) + 1
        for err, count in list(err_counts.items())[:5]:
            print(f"  - Count: {count}, Error: {err}")
            
    print("=" * 60)

def percentiles(sorted_list, pct):
    if not sorted_list:
        return 0
    idx = int(round(pct * (len(sorted_list) - 1)))
    return sorted_list[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation Service Load Tester")
    parser.add_argument("--url", default="http://10.129.6.170:8888", help="Target API Base URL")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests (threads)")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests to execute")
    parser.add_argument("--text", choices=["short", "medium", "long"], default="medium", help="Payload text size")
    parser.add_argument("--model", default="ai4bharat/indictrans2-en-indic-1B", help="Model to use for translation")
    parser.add_argument("--src", default="English", help="Source language")
    parser.add_argument("--tgt", default="Hindi", help="Target language")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size parameter for translation")
    
    args = parser.parse_args()
    run_load_test(
        url=args.url,
        concurrency=args.concurrency,
        num_requests=args.requests,
        text_key=args.text,
        model_name=args.model,
        source_lang=args.src,
        target_lang=args.tgt,
        batch_size=args.batch_size
    )
