import time
import os
import sys


# Get the directory where benchmark_tokenizer.py physically lives
# Change this section in your file (lines 7-8)
current_dir = os.path.dirname(os.path.abspath(__file__))
tokenization_dir = os.path.join(current_dir, "tokenization")
sys.path.append(tokenization_dir)


try:
    from byte_bpe_tokenizer import ByteLevelBPETokenizer
    from production_tokenizer import ProductionTokenizer
except ImportError as e:
    print(f"[ERROR] Import sequence failed: {str(e)}")
    print(f"Current System Paths searched: {sys.path}")
    sys.exit(1)

# ==============================================================================
# 2. BENCHMARK EXECUTION ENGINE
# ==============================================================================
def run_benchmark():
    # Multi-language test dataset mimicking your project inputs
    base_corpus = [
        "Artificial intelligence is changing the way software is built.",
        "Machine learning allows computers to learn patterns from data.",
        "Deep learning uses neural networks to solve complex problems.",
        "Large language models process and generate human language.",
        "A transformer uses attention to understand relationships between tokens.",
        "Hello, world! def fibonacci(n): return n + 1",
        "🚀 AI is amazing! AI 2026 🚀 भारत",
        "नमस्ते दुनिया  こんにちは世界  你好世界"
    ]

    print("Scaling dataset to simulate real-world production workload...")
    # Scale up text data to create a sufficient execution run (roughly ~3-5 Megabytes)
    scaled_corpus = base_corpus * 8000  
    combined_text = "\n".join(scaled_corpus)
    
    # Calculate dataset metrics
    dataset_bytes = len(combined_text.encode('utf-8'))
    dataset_mb = dataset_bytes / (1024 * 1024)
    print(f"Dataset compiled: {len(scaled_corpus):,} lines | {dataset_mb:.2f} MB\n")

    # Initialize and pre-train both systems on the base corpus
    print("Training production tokenizer pipeline...")
    hf_tokenizer = ProductionTokenizer(vocab_size=1500)
    hf_tokenizer.train_on_corpus(base_corpus)
    
    # Look for this block inside benchmark_tokenizer.py and change it:
    print("Initializing your custom ByteLevelBPETokenizer...")
    scratch_tokenizer = ByteLevelBPETokenizer()

    if hasattr(scratch_tokenizer, 'train'):
        try:
            print("Pre-training your custom tokenizer configuration...")
            # FIX: Join the base_corpus list into a single clean string payload
            training_string_payload = "\n".join(base_corpus)
            scratch_tokenizer.train(training_string_payload,num_merges=1244) 
        except Exception as e:
            print(f"[Warning] Custom training bypassed due to an inner class requirement: {e}")


    # --------------------------------------------------------------------------
    # Test 1: Benchmark Your Custom Scratch Implementation
    # --------------------------------------------------------------------------
    print("\n[Executing] Running Your Custom ByteLevelBPETokenizer...")
    start_time = time.perf_counter()
    
    scratch_token_count = 0
    scratch_failed = False
    
    for line in scaled_corpus:
        try:
            tokens = scratch_tokenizer.encode(line)
            scratch_token_count += len(tokens)
        except Exception as e:
            scratch_failed = True
            error_msg = str(e)
            break
            
    scratch_duration = time.perf_counter() - start_time
    scratch_throughput = dataset_mb / max(scratch_duration, 0.001)

    # --------------------------------------------------------------------------
    # Test 2: Benchmark Production-Grade Implementation (Hugging Face)
    # --------------------------------------------------------------------------
    print("[Executing] Running Production Hugging Face Tokenizer...")
    start_time = time.perf_counter()
    
    hf_token_count = 0
    for line in scaled_corpus:
        tokens = hf_tokenizer.encode(line)
        hf_token_count += len(tokens)
        
    hf_duration = time.perf_counter() - start_time
    hf_throughput = dataset_mb / hf_duration

    # ==============================================================================
    # 3. PERFORMANCE METRICS REPORT
    # ==============================================================================
    print("\n" + "="*80)
    print("                      TOKENIZATION PERFORMANCE REPORT           ")
    print("="*80)
    
    scratch_time_display = f"{scratch_duration:.4f} seconds" if not scratch_failed else "CRASH/FAIL"
    scratch_tp_display = f"{scratch_throughput:.2f} MB/s" if not scratch_failed else "N/A"
    scratch_count_display = f"{scratch_token_count:,}" if not scratch_failed else "N/A"
    
    print(f"{'Metric':<25} | {'Your ByteLevelBPE':<25} | {'Production (Hugging Face)':<25}")
    print("-"*80)
    print(f"{'Total Execution Time':<25} | {scratch_time_display:<25} | {hf_duration:.4f} seconds")
    print(f"{'System Throughput':<25} | {scratch_tp_display:<25} | {hf_throughput:.2f} MB/s")
    print(f"{'Total Tokens Output':<25} | {scratch_count_display:<25} | {hf_token_count:,}")
    print("="*80)
    
    if not scratch_failed:

        speedup = (
            scratch_throughput /
            max(hf_throughput, 0.001)
        )

        print(
            f"\nProduction/Scratch throughput ratio: "
            f"{hf_throughput / max(scratch_throughput, 0.001):.2f}x"
        )

        print(
            f"Scratch/HF throughput ratio: "
            f"{speedup:.2f}x"
        )

        print(
            "\nNote:"
        )

        print(
            "This benchmark compares two Byte-Level BPE "
            "implementations under similar vocabulary-size settings."
        )

        print(
            "The Hugging Face implementation uses an optimized "
            "native tokenizer backend, while the scratch implementation "
            "uses Python-level loops."
        )
    else:
        speedup = hf_throughput / max(scratch_throughput, 0.001)
        print(f"\n🚀 System Verdict: Production Tokenizer is {speedup:.1f}x FASTER than your native loops.")

if __name__ == "__main__":
    run_benchmark()
