def main():
    import argparse
    import asyncio
    import json
    import subprocess
    import sys
    import time
    import unicodedata
    from datetime import datetime, timezone
    from pathlib import Path

    from copilot import CopilotClient

    PROMPT = (
        "Transcribe the historical French text in this image exactly as written. "
        "Preserve the original spelling, accents, punctuation, and word order. "
        "Do not modernize, translate, explain, or add a heading. "
        "Return only the transcription."
    )

    PROMPT_2 = (
        "Transcribe the historical French text in this image exactly as written. "
        "Preserve every character: original spelling, accents, punctuation, and word order. "
        "Do NOT modernize, 'fix,' translate, or improve the text. "
        "Do NOT add interpretive marks like [illegible], [unclear], or ellipsis (...). "
        "Do NOT reformat, reorder, or add spacing beyond what is visible. "
        "Preserve line breaks and layout exactly as shown. "
        "For unclear or barely legible characters, make your best guess based on context and handwriting patterns. "
        "Return only the transcription."
    )
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    def normalize(text):
        return " ".join(unicodedata.normalize("NFC", text).split())

    def edit_distance(reference, hypothesis):
        previous = list(range(len(hypothesis) + 1))
        for reference_index, reference_item in enumerate(reference, 1):
            current = [reference_index]
            for hypothesis_index, hypothesis_item in enumerate(hypothesis, 1):
                current.append(min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + (reference_item != hypothesis_item),
                ))
            previous = current
        return previous[-1]

    def score(reference, hypothesis):
        reference = normalize(reference)
        hypothesis = normalize(hypothesis)
        if not reference:
            raise ValueError("Reference transcript is empty")
        reference_words = reference.split()
        return {
            "character_errors": edit_distance(reference, hypothesis),
            "reference_characters": len(reference),
            "word_errors": edit_distance(reference_words, hypothesis.split()),
            "reference_words": len(reference_words),
        }

    def discover_samples(data_dir):
        samples = []
        for transcript in sorted(data_dir.rglob("transcript.txt")):
            images = sorted(path for path in transcript.parent.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if len(images) != 1:
                raise ValueError(f"{transcript.parent}: expected exactly one image, found {len(images)}")
            reference = transcript.read_text(encoding="utf-8-sig")
            if not normalize(reference):
                raise ValueError(f"{transcript}: reference transcript is empty")
            samples.append((transcript.parent.relative_to(data_dir).as_posix(), images[0], reference))
        if not samples:
            raise ValueError(f"No transcript.txt files found under {data_dir}")
        return samples

    async def run_benchmark(args):
        samples = [] if args.list_models else discover_samples(args.data_dir)
        async with CopilotClient() as client:
            all_models = [model for model in await client.list_models() if model.capabilities.supports.vision]
            
            # Select representative models of different sizes/architectures
            # Prioritize: small/efficient, medium, large, specialized/latest
            representative_ids = {
                "claude-sonnet-5.5",
                "gpt-6-luna"
            }
            
            models = [m for m in all_models if m.id in representative_ids]
            # Fallback: if specific models unavailable, use all available models
            if not models:
                models = all_models
            
            if args.list_models:
                for model in models:
                    print(f"{model.id}\t{model.name}")
                if not models:
                    print("No vision-capable models returned by Copilot.", file=sys.stderr)
                return 0 if models else 1

            available = {model.id: model for model in models}
            selected = args.model or list(available)
            unknown = set(selected) - available.keys()
            if unknown:
                raise ValueError(f"Models not available with vision support: {', '.join(sorted(unknown))}")
            if not selected:
                raise ValueError("No vision-capable models returned by Copilot")

            report = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_dir": str(args.data_dir.resolve()),
                "prompt": PROMPT_2,
                "normalization": "Unicode NFC and collapsed whitespace; case and punctuation preserved",
                "models": [],
            }
            for model_id in dict.fromkeys(selected):
                model_report = {"id": model_id, "name": available[model_id].name, "samples": []}
                for sample_id, image, reference in samples:
                    print(f"{model_id}: {sample_id}", flush=True)
                    started = time.monotonic()
                    result = {"sample": sample_id, "image": str(image.resolve())}
                    try:
                        async with await client.create_session(
                            model=model_id,
                            available_tools=[],
                            skip_custom_instructions=True,
                            enable_skills=False,
                            enable_session_store=False,
                            memory={"enabled": False},
                        ) as session:
                            response = await session.send_and_wait(
                                PROMPT_2,
                                attachments=[{"type": "file", "path": str(image.resolve())}],
                                timeout=args.timeout,
                            )
                        if response is None or not response.data.content or not response.data.content.strip():
                            raise RuntimeError("Model returned no transcription")
                        result["transcription"] = response.data.content
                        result.update(score(reference, response.data.content))
                    except Exception as error:
                        result["error"] = f"{type(error).__name__}: {error}"
                        print(f"  failed: {result['error']}", file=sys.stderr)
                    result["seconds"] = round(time.monotonic() - started, 3)
                    model_report["samples"].append(result)
                report["models"].append(model_report)

                successes = [result for result in model_report["samples"] if "error" not in result]
                model_report["successful_samples"] = len(successes)
                model_report["failed_samples"] = len(samples) - len(successes)
                if successes:
                    for unit in ("character", "word"):
                        errors = sum(result[f"{unit}_errors"] for result in successes)
                        total = sum(result[f"reference_{unit}s"] for result in successes)
                        model_report[f"{unit}_error_rate"] = errors / total
                    print(f"  CER: {model_report['character_error_rate']:.2%}; "
                          f"WER: {model_report['word_error_rate']:.2%}; "
                          f"completed: {len(successes)}/{len(samples)}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"Results written to {args.output}")
        return 1 if any(model["failed_samples"] for model in report["models"]) else 0

    parser = argparse.ArgumentParser(description="Benchmark Copilot vision models on historical French scans.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", action="append", help="Model ID to benchmark; repeat to select multiple models")
    parser.add_argument("--list-models", action="store_true", help="List available vision-capable models")
    parser.add_argument("--login", action="store_true", help="Authenticate with GitHub CLI for Copilot SDK access")
    parser.add_argument("--timeout", type=float, default=180, help="Seconds per model request (default: 180)")
    parser.add_argument("--output", type=Path, default=Path("results") / f"benchmark-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json")
    args = parser.parse_args()
    if args.login:
        try:
            subprocess.run(["gh", "auth", "login"], check=True)
        except FileNotFoundError:
            print("GitHub CLI (gh) is not installed or not on PATH.", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as error:
            print(f"GitHub authentication failed (exit code {error.returncode}).", file=sys.stderr)
            return error.returncode or 1
        print("GitHub authentication saved. The Copilot SDK will use these credentials.")
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.list_models and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        return asyncio.run(run_benchmark(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
