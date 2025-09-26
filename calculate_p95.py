import json
import numpy as np
import os
import re
from pathlib import Path

# --- Configuration ---
LOG_FILE_PATH = "demo/app.log"
# The path to your log file, relative to the root of your project
LATENCY_REGEX = r"RAG pipeline completed\. Latency: (\d+\.\d+) ms"
ERROR_REGEX = r"RAG pipeline failed with error"
#SUCCESS_REGEX = r"Message generated successfully" # No longer needed

def calculate_metrics_from_logs(log_file_path: str) -> (float, float):
    """
    Calculates the 95th percentile of latency and error rate from the log file.
    """
    if not Path(log_file_path).exists():
        print(f"Error: Log file not found at {log_file_path}")
        return float("inf"), float("inf")

    latencies = []
    error_count = 0
    total_requests = 0

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if re.search(ERROR_REGEX, line):
                total_requests += 1
                error_count += 1
            else:
                match = re.search(LATENCY_REGEX, line)
                if match:
                    total_requests += 1
                    latencies.append(float(match.group(1)))
    
    if total_requests == 0:
        print("No requests found in the logs.")
        return float("inf"), float("inf")

    # Calculate the 95th percentile
    if latencies:
        p95_latency = np.percentile(latencies, 95)
    else:
        p95_latency = float("inf")

    error_rate = error_count / total_requests
    
    return p95_latency, error_rate

if __name__ == "__main__":
    p95_val, error_rate_val = calculate_metrics_from_logs(LOG_FILE_PATH)
    
    print("\n================= QUALITY BARS ==================")
    print(f"P95 RAG Latency: {p95_val:.2f} ms")
    print(f"Total Error Rate: {error_rate_val * 100:.2f}%")
    print("=================================================")
