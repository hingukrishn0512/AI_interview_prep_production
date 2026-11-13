"""
Golden Router Evaluation Suite
-------------------------------
Runs 20 ground-truth questions against classifier_node to measure
routing accuracy and prevent silent misrouting regressions across deploys.
"""

import sys
import os
import time

# Ensure Agent root is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project_raw import classifier_node

GOLDEN_ROUTER_DATASET = [
    # --- DSA Practice (Specific Topic Specified) ---
    {
        "query": "Give me a hard dynamic programming question on the knapsack pattern.",
        "expected": "dsa",
    },
    {
        "query": "Can you give me an array problem with two pointers or sliding window?",
        "expected": "dsa",
    },
    {
        "query": "Ask me a tree traversal question using recursion.",
        "expected": "dsa",
    },
    {
        "query": "Give me a graph problem involving Dijkstra's or BFS shortest path.",
        "expected": "dsa",
    },

    # --- Behavioral STAR Coaching (Theme Specified) ---
    {
        "query": "Ask me a behavioral question about handling conflict with a senior teammate.",
        "expected": "behavioral",
    },
    {
        "query": "Give me a situational question about leadership and taking ownership of a failing project.",
        "expected": "behavioral",
    },
    {
        "query": "Ask me a behavioral prompt focusing on prioritizing conflicting deadlines under pressure.",
        "expected": "behavioral",
    },
    {
        "query": "Give me an HR question about a time I failed and what I learned from it.",
        "expected": "behavioral",
    },

    # --- Company Intel & Process ---
    {
        "query": "What is Google's interview process like for L4 Software Engineers?",
        "expected": "company_research",
    },
    {
        "query": "Tell me about Amazon's 16 Leadership Principles and how they test them in interviews.",
        "expected": "company_research",
    },
    {
        "query": "What rounds can I expect in Meta's technical interview loop?",
        "expected": "company_research",
    },
    {
        "query": "What is the engineering culture and tech stack at Uber?",
        "expected": "company_research",
    },

    # --- Resume Grounding & Gap Analysis ---
    {
        "query": "Based on my resume, what technical skills or projects should I brush up on?",
        "expected": "resume_gap",
    },
    {
        "query": "Can you give me a 60-second elevator pitch / 'tell me about yourself' based on my background?",
        "expected": "resume_gap",
    },
    {
        "query": "Which of my past experiences should I highlight most for this role?",
        "expected": "resume_gap",
    },
    {
        "query": "How does my resume compare to the expectations for a Senior Backend Engineer?",
        "expected": "resume_gap",
    },

    # --- General Chat & Clarifications ---
    {
        "query": "Hello! How can you help me prepare today?",
        "expected": "general_chat",
    },
    {
        "query": "Thanks for the feedback, that was really helpful!",
        "expected": "general_chat",
    },
    {
        "query": "Ask me a behavioral question.",  # Bare request without theme -> coach clarifies
        "expected": "general_chat",
    },
    {
        "query": "What are some good resources or books to learn system design?",
        "expected": "general_chat",
    },
]


def run_router_eval():
    print("=" * 70)
    print("🚀 RUNNING MULTI-AGENT ROUTER EVALUATION (20 Ground-Truth Samples)")
    print("=" * 70)

    passed = 0
    total = len(GOLDEN_ROUTER_DATASET)
    latencies = []

    for idx, item in enumerate(GOLDEN_ROUTER_DATASET, 1):
        state = {
            "user_input": item["query"],
            "messages": [],
            "history": [],
            "company_name": "Google",
            "role": "Software Engineer",
            "resume": "Sample resume with Python and distributed systems experience.",
        }

        t0 = time.perf_counter()
        res = classifier_node(state)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(elapsed_ms)

        predicted = res.get("classifier", "").strip().lower()
        is_ok = predicted == item["expected"]

        if is_ok:
            passed += 1
            status_tag = "✅ PASS"
        else:
            status_tag = f"❌ FAIL (Expected: {item['expected']}, Got: {predicted})"

        print(f"[{idx:02d}/{total:02d}] {status_tag} | {elapsed_ms}ms")
        print(f"     Query: \"{item['query'][:60]}...\"")

    accuracy = (passed / total) * 100
    avg_latency = sum(latencies) // len(latencies)

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY: {passed}/{total} Passed ({accuracy:.1f}% Accuracy)")
    print(f"Average Classifier Latency: {avg_latency} ms")
    print("=" * 70)

    if accuracy < 90.0:
        print("❌ ROUTER REGRESSION DETECTED: Accuracy is below 90% threshold.")
        sys.exit(1)
    else:
        print("✅ ROUTER EVALUATION PASSED. System is safe to deploy.")
        sys.exit(0)


if __name__ == "__main__":
    run_router_eval()
