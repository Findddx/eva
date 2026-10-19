"""Streaming terminal client. Each interactive prompt is a separate single-user request."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def iter_sse(response, *, max_event_bytes=2 * 1024 * 1024):
    """Parse SSE data lines, including multi-line events and UTF-8 byte content."""
    data, size = [], 0
    for raw in response:
        size += len(raw)
        if size > max_event_bytes:
            raise ValueError("SSE event exceeds client limit")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
            data, size = [], 0
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
        # Ignore comments, event/id/retry fields; no evaluation of their contents.
    if data:
        raise ValueError("Connection ended with an incomplete SSE event")


def request_once(url, prompt, *, model="eva", maximum=256, mode="greedy", key="", timeout=900,
                 output=sys.stdout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maximum, "temperature": 0, "mode": mode, "stream": True},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
    started = time.perf_counter()
    first = None
    fragments, final, completed = [], None, False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("Server did not return an SSE stream")
            for payload in iter_sse(response):
                if payload == "[DONE]":
                    completed = True
                    break
                event = json.loads(payload)
                if "error" in event:
                    error = event["error"]
                    raise RuntimeError(str(error.get("code", "server_error")) + ": " + str(error.get("message", "")))
                choices = event.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                text = choice.get("delta", {}).get("content", choice.get("text", ""))
                if text:
                    if first is None:
                        first = time.perf_counter() - started
                    output.write(text)
                    output.flush()
                    fragments.append(text)
                if choice.get("finish_reason") is not None:
                    final = event
    except urllib.error.HTTPError as exc:
        # Bound diagnostic size. The server does not send prompt bodies in errors.
        message = exc.read(16384).decode("utf-8", errors="replace")
        raise RuntimeError("HTTP " + str(exc.code) + ": " + message) from None
    if not completed or final is None:
        raise RuntimeError("Stream ended without a final metrics event and [DONE]")
    observed = {"first_text_s": first, "e2e_s": time.perf_counter() - started,
                "scope": "Client request through received DONE; first_text excludes empty role chunks"}
    return {"text": "".join(fragments), "finish_reason": final["choices"][0]["finish_reason"],
            "usage": final.get("usage", {}), "metrics": final.get("metrics", {}), "client": observed}


def print_metrics(result):
    engine = result.get('metrics', {}).get('engine', {})
    observed = result.get('client', {})
    def number(value, suffix):
        return 'n/a' if value is None else f'{value:.3f}{suffix}'
    print('\n\n' + ' | '.join((
        'First text: '+number(observed.get('first_text_s'), 's'),
        'Decode: '+number(engine.get('decode_tok_s'), ' tok/s'),
        'Total: '+number(observed.get('e2e_s'), 's'),
        'Output: '+str(result.get('usage', {}).get('completion_tokens', 0))+' tokens',
        'Finish: '+result.get('finish_reason', 'unknown'))), file=sys.stderr, flush=True)


def record(path, name, prompt, result=None, error=None):
    if path is None:
        return
    item = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "case": name,
            "prompt": prompt,
            "result": result, "error": error, "scope": "Actual response and timings from this request"}
    # Caller selects the client-local file. No server asset or source paths are accepted.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="eva")
    p.add_argument("--key-env", default="EVA_API_KEY")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--mode", choices=("greedy", "lookup"), default="greedy")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--prompt")
    group.add_argument("--prompt-file", type=Path)
    group.add_argument("--suite", type=Path)
    p.add_argument("--record", type=Path, help="Append client-local JSONL with prompts, outputs and timings")
    args = p.parse_args(argv)
    if args.max_tokens <= 0:
        p.error("max-tokens must be positive")
    key = os.environ.get(args.key_env, "")

    def run(name, prompt, maximum, mode):
        try:
            result = request_once(args.url, prompt, model=args.model, maximum=maximum, mode=mode, key=key)
            print_metrics(result)
            record(args.record, name, prompt, result=result)
            return True
        except KeyboardInterrupt:
            # urlopen context closes the connection; gateway waits for worker acknowledgement.
            print("\nClient disconnected; cancellation waits for the worker's safe boundary.", file=sys.stderr)
            record(args.record, name, prompt, error="client_cancelled")
            return False
        except Exception as exc:
            print("\nRequest failed: " + str(exc), file=sys.stderr)
            record(args.record, name, prompt, error=str(exc))
            return False

    if args.suite:
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        cases = suite.get("cases") if isinstance(suite, dict) else suite
        if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
            p.error("suite must contain1..64 cases")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("prompt"), str):
                p.error("Each suite case needs a literal prompt string")
            print("\n[" + str(case.get("name", "case")) + "]", flush=True)
            if not run(case.get("name", "case"), case["prompt"], case.get("max_tokens", args.max_tokens), case.get("mode", args.mode)):
                return 1
        return 0
    if args.prompt is not None or args.prompt_file:
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
        return 0 if run("prompt", prompt, args.max_tokens, args.mode) else 1
    print("Single-user, stateless requests. /quit exits; /paste ends on a line containing only '.'", file=sys.stderr)
    while True:
        try:
            prompt = input("\neva> ")
            if prompt.strip() in ("/quit", "/exit"):
                return 0
            if prompt.strip() == "/paste":
                lines = []
                while True:
                    line = input()
                    if line == ".":
                        break
                    lines.append(line)
                prompt = "\n".join(lines)
            if prompt.strip():
                run("interactive", prompt, args.max_tokens, args.mode)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
