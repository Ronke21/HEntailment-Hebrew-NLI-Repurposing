#!/usr/bin/env python3
"""
vllm_worker.py — standalone subprocess entry point for vLLM-based generation.

Runs under a separate, dedicated venv (heb_nli_vllm_venv) so its vLLM/torch
version pairing never has to match whatever the main training venv/conda env
uses. Takes a batch of already-built prompt lists (one per named set — e.g.
"dev", "winograd", "heq", ...), builds ONE vLLM engine, and generates for all
of them in that single engine lifetime (a cold engine start costs real
wall-clock time, so batching avoids paying it once per test set).

Input/output are plain JSON files so the caller (decoder_pipeline.py, running
under a different interpreter) never needs to import vllm or torch itself.
"""

import argparse
import json


def _gpu_mem_fraction(headroom: float = 0.95, floor: float = 0.30,
                      ceiling: float = 0.95) -> float:
    """vLLM's gpu_memory_utilization as a fraction of TOTAL memory, derived
    from what is actually FREE on this device right now.

    Called once, before LLM() is constructed — never in a retry loop (see the
    note at the call site: constructing LLM() twice in one process corrupts
    vLLM's global state). If the device cannot be queried we fall back to the
    historical constant rather than guessing.
    """
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
    except Exception as e:  # no CUDA, driver hiccup, etc.
        print(f"[vllm_worker] could not query GPU memory ({e}); "
              f"falling back to gpu_memory_utilization=0.5", flush=True)
        return 0.5
    frac = (free_b * headroom) / total_b
    frac = max(floor, min(ceiling, frac))
    print(f"[vllm_worker] free={free_b/2**30:.1f}GiB total={total_b/2**30:.1f}GiB "
          f"-> gpu_memory_utilization={frac:.3f}", flush=True)
    return frac


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter-path", default="")
    p.add_argument("--batch-file", required=True,
                    help="JSON: {set_name: [prompt_str, ...], ...}")
    p.add_argument("--output-file", required=True,
                    help="JSON: {set_name: [generated_text, ...], ...}")
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--max-lora-rank", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-flashinfer-autotune", action="store_true")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from vllm.lora.layers.vocal_parallel_embedding import VocabParallelEmbeddingWithLoRA
    if not hasattr(VocabParallelEmbeddingWithLoRA, "quant_method"):
        VocabParallelEmbeddingWithLoRA.quant_method = property(
            lambda self: self.base_layer.quant_method)

    with open(args.batch_file, encoding="utf-8") as f:
        batch: dict[str, list[str]] = json.load(f)

    adapter_path = args.adapter_path or None
    llm_kwargs = dict(model=args.model, trust_remote_code=args.trust_remote_code,
             dtype="bfloat16",
             gpu_memory_utilization=_gpu_mem_fraction(),
             max_model_len=args.max_model_len,
             enable_lora=bool(adapter_path),
             max_lora_rank=args.max_lora_rank,
             enforce_eager=args.enforce_eager,
             max_num_seqs=256)
    if args.disable_flashinfer_autotune:
        llm_kwargs["enable_flashinfer_autotune"] = False

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    lora_request = LoRARequest("adapter", 1, adapter_path) if adapter_path else None

    tokenizer = llm.get_tokenizer()
    prompt_budget = args.max_model_len - args.max_new_tokens - 8  # 8-token safety floor
    n_reclamped = 0

    def _clamp(prompt: str) -> str:
        nonlocal n_reclamped
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if len(ids) <= prompt_budget:
            return prompt
        n_reclamped += 1
        return tokenizer.decode(ids[-prompt_budget:], skip_special_tokens=True)

    results: dict[str, list[str]] = {}
    for set_name, prompts in batch.items():
        clamped = [_clamp(p) for p in prompts]
        outputs = llm.generate(clamped, sampling, lora_request=lora_request)
        results[set_name] = [o.outputs[0].text for o in outputs]
        print(f"[vllm_worker] {set_name}: generated {len(outputs)}/{len(prompts)}",
              flush=True)
    if n_reclamped:
        print(f"[vllm_worker] re-clamped {n_reclamped} prompt(s) that exceeded "
              f"the real vLLM tokenizer budget despite the caller's own truncation",
              flush=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
