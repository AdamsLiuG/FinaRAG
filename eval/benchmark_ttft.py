from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_PAYLOAD_FILE = REPO_ROOT / "fina_payload.json"


@dataclass(frozen=True)
class TargetConfig:
    label: str
    base_url: str
    model: str
    api_key: str | None = None

    @property
    def request_url(self) -> str:
        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _should_bypass_env_proxy(base_url: str) -> bool:
    hostname = urlparse(base_url).hostname
    if not hostname:
        return False

    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return hostname.endswith(".local")


def _load_payload(payload_file: Path) -> dict[str, Any]:
    with open(payload_file, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "messages" not in payload:
        raise ValueError(f"Payload file must contain a 'messages' field: {payload_file}")
    return payload


def _build_payload(
    payload_template: dict[str, Any],
    model: str,
    max_tokens: int | None,
    seed: int | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(payload_template)
    payload["model"] = model
    payload["stream"] = True

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed
    return payload


def _extract_stream_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return (
        delta.get("content")
        or delta.get("reasoning_content")
        or delta.get("reasoning")
        or ""
    )


def _measure_single_request(
    target: TargetConfig,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    request_kwargs = {
        "url": target.request_url,
        "headers": headers,
        "json": payload,
        "timeout": timeout,
        "stream": True,
    }

    start_time = time.perf_counter()

    def _consume_stream(response: requests.Response) -> dict[str, Any]:
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} Client Error for url: {response.url}\n{response.text[:1000]}",
                response=response,
            )

        first_token_latency = None
        chunk_count = 0
        output_chars = 0

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            text = _extract_stream_text(chunk)
            if not text:
                continue

            chunk_count += 1
            output_chars += len(text)
            if first_token_latency is None:
                first_token_latency = time.perf_counter() - start_time

        if first_token_latency is None:
            raise ValueError("Streaming response completed without any content-bearing chunk.")

        total_latency = time.perf_counter() - start_time
        return {
            "ttft_seconds": first_token_latency,
            "total_latency_seconds": total_latency,
            "chunk_count": chunk_count,
            "output_chars": output_chars,
        }

    if _should_bypass_env_proxy(target.base_url):
        with requests.Session() as session:
            session.trust_env = False
            with session.post(**request_kwargs) as response:
                return _consume_stream(response)

    with requests.post(**request_kwargs) as response:
        return _consume_stream(response)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentiles for an empty list.")
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = index - lower
    return lower_value + (upper_value - lower_value) * weight


def _summarize_metric(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _benchmark_target(
    target: TargetConfig,
    payload_template: dict[str, Any],
    runs: int,
    warmup: int,
    timeout: float,
    max_tokens: int | None,
    seed: int | None,
    continue_on_error: bool,
    verbose: bool,
) -> dict[str, Any]:
    total_runs = warmup + runs
    measured_runs: list[dict[str, Any]] = []
    warmup_runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run_index in range(total_runs):
        is_warmup = run_index < warmup
        payload = _build_payload(
            payload_template=payload_template,
            model=target.model,
            max_tokens=max_tokens,
            seed=seed,
        )

        try:
            result = _measure_single_request(target, payload=payload, timeout=timeout)
        except Exception as exc:
            failure = {
                "run_index": run_index + 1,
                "warmup": is_warmup,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            if verbose:
                phase = "warmup" if is_warmup else "measure"
                print(f"[{target.label}] {phase} run {run_index + 1}/{total_runs} failed: {failure['error']}")
            if not continue_on_error:
                raise
            continue

        run_record = {
            "run_index": run_index + 1,
            "warmup": is_warmup,
            **result,
        }
        if is_warmup:
            warmup_runs.append(run_record)
        else:
            measured_runs.append(run_record)

        if verbose:
            phase = "warmup" if is_warmup else "measure"
            print(
                f"[{target.label}] {phase} run {run_index + 1}/{total_runs}: "
                f"ttft={result['ttft_seconds'] * 1000:.1f} ms, "
                f"e2e={result['total_latency_seconds'] * 1000:.1f} ms"
            )

    if not measured_runs:
        raise RuntimeError(f"No successful measured runs were collected for target '{target.label}'.")

    ttft_values = [item["ttft_seconds"] for item in measured_runs]
    total_latency_values = [item["total_latency_seconds"] for item in measured_runs]

    return {
        "label": target.label,
        "base_url": target.base_url,
        "request_url": target.request_url,
        "model": target.model,
        "warmup": warmup,
        "requested_runs": runs,
        "successful_measured_runs": len(measured_runs),
        "failed_runs": len(failures),
        "ttft_seconds": _summarize_metric(ttft_values),
        "total_latency_seconds": _summarize_metric(total_latency_values),
        "measured_runs": measured_runs,
        "warmup_runs": warmup_runs,
        "failures": failures,
    }


def _reduction_ratio(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (baseline - candidate) / baseline


def _compare_targets(primary: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "baseline_label": primary["label"],
        "candidate_label": candidate["label"],
    }

    for metric_name in ("ttft_seconds", "total_latency_seconds"):
        baseline_metrics = primary[metric_name]
        candidate_metrics = candidate[metric_name]
        metric_comparison: dict[str, Any] = {}

        for stat_name in ("mean", "p50", "p95"):
            baseline_value = float(baseline_metrics[stat_name])
            candidate_value = float(candidate_metrics[stat_name])
            metric_comparison[stat_name] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "absolute_improvement": baseline_value - candidate_value,
                "reduction_ratio": _reduction_ratio(baseline_value, candidate_value),
            }

        comparison[metric_name] = metric_comparison

    return comparison


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.1f} ms"


def _format_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    return f"{ratio * 100:.1f}%"


def _print_summary(report: dict[str, Any]) -> None:
    print("TTFT benchmark summary")
    print(f"- payload: {report['payload_file']}")
    print(
        f"- runs: {report['settings']['runs']} measured + "
        f"{report['settings']['warmup']} warmup per target"
    )
    print(f"- forced stream: {report['settings']['force_stream']}")

    for target in report["targets"]:
        ttft = target["ttft_seconds"]
        total_latency = target["total_latency_seconds"]
        print(f"\n[{target['label']}] {target['model']}")
        print(f"- url: {target['request_url']}")
        print(
            f"- success: {target['successful_measured_runs']}/{target['requested_runs']} measured, "
            f"failures: {target['failed_runs']}"
        )
        print(
            f"- TTFT: p50={_format_ms(float(ttft['p50']))}, "
            f"p95={_format_ms(float(ttft['p95']))}, mean={_format_ms(float(ttft['mean']))}"
        )
        print(
            f"- E2E:  p50={_format_ms(float(total_latency['p50']))}, "
            f"p95={_format_ms(float(total_latency['p95']))}, mean={_format_ms(float(total_latency['mean']))}"
        )

    comparison = report.get("comparison")
    if comparison:
        print(f"\n[comparison] {comparison['candidate_label']} vs {comparison['baseline_label']}")
        for metric_name, title in (
            ("ttft_seconds", "TTFT"),
            ("total_latency_seconds", "E2E"),
        ):
            metric = comparison[metric_name]
            print(
                f"- {title}: "
                f"p50 improvement={_format_ms(metric['p50']['absolute_improvement'])} "
                f"({_format_ratio(metric['p50']['reduction_ratio'])}), "
                f"p95 improvement={_format_ms(metric['p95']['absolute_improvement'])} "
                f"({_format_ratio(metric['p95']['reduction_ratio'])}), "
                f"mean improvement={_format_ms(metric['mean']['absolute_improvement'])} "
                f"({_format_ratio(metric['mean']['reduction_ratio'])})"
            )


def _resolve_primary_target(args: argparse.Namespace) -> TargetConfig:
    base_url = args.base_url or _env_value("QWEN_BASE_URL", "LLM_BASE_URL")
    model = args.model or _env_value("QWEN_MODEL", "LLM_MODEL")
    api_key = args.api_key if args.api_key is not None else _env_value("QWEN_API_KEY", "LLM_API_KEY")

    if not base_url:
        raise ValueError("Missing primary base URL. Pass --base-url or configure QWEN_BASE_URL / LLM_BASE_URL.")
    if not model:
        raise ValueError("Missing primary model. Pass --model or configure QWEN_MODEL / LLM_MODEL.")

    return TargetConfig(
        label=args.label,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )


def _resolve_candidate_target(args: argparse.Namespace, primary_target: TargetConfig) -> TargetConfig | None:
    candidate_args = (
        args.candidate_base_url,
        args.candidate_model,
        args.candidate_api_key,
    )
    if not any(value is not None for value in candidate_args):
        return None

    candidate = TargetConfig(
        label=args.candidate_label,
        base_url=args.candidate_base_url or primary_target.base_url,
        model=args.candidate_model or primary_target.model,
        api_key=args.candidate_api_key if args.candidate_api_key is not None else primary_target.api_key,
    )

    if candidate.base_url == primary_target.base_url and candidate.model == primary_target.model:
        raise ValueError("Candidate target matches the primary target. Change --candidate-base-url and/or --candidate-model.")

    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT against an OpenAI-compatible / vLLM chat completions endpoint."
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--payload-file", type=Path, default=DEFAULT_PAYLOAD_FILE)
    parser.add_argument("--base-url", default=None, help="Primary endpoint base URL. Falls back to QWEN_BASE_URL / LLM_BASE_URL.")
    parser.add_argument("--model", default=None, help="Primary model name. Falls back to QWEN_MODEL / LLM_MODEL.")
    parser.add_argument("--api-key", default=None, help="Primary API key. Falls back to QWEN_API_KEY / LLM_API_KEY.")
    parser.add_argument("--label", default="primary", help="Primary target label.")
    parser.add_argument("--candidate-base-url", default=None, help="Optional comparison endpoint base URL.")
    parser.add_argument("--candidate-model", default=None, help="Optional comparison model name.")
    parser.add_argument("--candidate-api-key", default=None, help="Optional comparison API key.")
    parser.add_argument("--candidate-label", default="candidate", help="Optional comparison target label.")
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per target.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per target.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens in the payload.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override for deterministic comparisons.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep benchmarking after individual request failures.")
    parser.add_argument("--verbose", action="store_true", help="Print per-run TTFT and end-to-end latency.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report to stdout.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for saving the JSON report.")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")

    if args.env_file and args.env_file.exists():
        load_dotenv(args.env_file)
    elif args.env_file and args.env_file != DEFAULT_ENV_FILE:
        raise FileNotFoundError(f"Env file not found: {args.env_file}")

    if not args.payload_file.exists():
        raise FileNotFoundError(f"Payload file not found: {args.payload_file}")

    payload_template = _load_payload(args.payload_file)
    primary_target = _resolve_primary_target(args)
    candidate_target = _resolve_candidate_target(args, primary_target)

    targets = [primary_target]
    if candidate_target is not None:
        targets.append(candidate_target)

    target_reports = [
        _benchmark_target(
            target=target,
            payload_template=payload_template,
            runs=args.runs,
            warmup=args.warmup,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            seed=args.seed,
            continue_on_error=args.continue_on_error,
            verbose=args.verbose,
        )
        for target in targets
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "payload_file": str(args.payload_file),
        "settings": {
            "runs": args.runs,
            "warmup": args.warmup,
            "timeout_seconds": args.timeout,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "force_stream": True,
        },
        "targets": target_reports,
    }

    if len(target_reports) == 2:
        report["comparison"] = _compare_targets(target_reports[0], target_reports[1])
    else:
        report["comparison"] = None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)


if __name__ == "__main__":
    main()
