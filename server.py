"""Single-worker streaming HTTP gateway. Standard library only; no model imports.

JSONL worker: ready/status, generate -> delta* -> done|error, cancel, shutdown.
Disconnect requests cancellation; only a worker terminal event releases residency.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_WORKER_LINE = 2 * 1024 * 1024


class Rejected(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class Job:
    def __init__(self, payload):
        self.id = "cmpl-" + uuid.uuid4().hex
        self.payload = payload
        self.events = queue.Queue(maxsize=256)
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.created = time.perf_counter()
        self.started = self.first_delta = self.finished = None
        self.result = None
        self.cancel_sent = False
        self.lock = threading.Lock()

    def finish(self, result):
        with self.lock:
            if self.done.is_set():
                return
            self.result, self.finished = result, time.perf_counter()
            self.done.set()

    def server_metrics(self):
        return {
            "queue_s": None if self.started is None else self.started - self.created,
            "first_text_delta_s": None if self.first_delta is None else self.first_delta - self.created,
            "e2e_s": None if self.finished is None else self.finished - self.created,
            "scope": "HTTP acceptance through worker terminal; includes queue, not network receipt acknowledgement",
        }


class WorkerBridge:
    def __init__(self, command, *, max_queue=2):
        self.pending = queue.Queue(maxsize=max_queue)
        self.lock, self.write_lock = threading.Lock(), threading.RLock()
        self.active = None
        self.ready = None
        self.stage = "starting"
        self.failure = None
        self.stopping = False
        self.closed = threading.Event()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, text=True, encoding="utf-8", errors="strict",
                                        bufsize=1, env=env)
        self.reader = threading.Thread(target=self._read, daemon=True, name="worker-jsonl")
        self.dispatcher = threading.Thread(target=self._dispatch, daemon=True, name="worker-dispatch")
        self.reader.start()
        self.dispatcher.start()

    def health(self):
        with self.lock:
            active = self.active
            return {
                "status": "failed" if self.failure else ("ready" if self.ready else "loading"),
                "ready": bool(self.ready) and not self.failure and not self.stopping and not self.closed.is_set(),
                "stage": self.stage, "queued_requests": self.pending.qsize(),
                "inflight": active is not None,
                "cancellation_pending": bool(active and active.cancelled.is_set()),
                "worker": self.ready,
                "error": self.failure,
            }

    def _send(self, item):
        encoded = json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self.write_lock:
            self.process.stdin.write(encoded + "\n")
            self.process.stdin.flush()

    def _fail(self, message):
        with self.lock:
            if self.failure is None:
                self.failure = message
            active = self.active
        if active:
            active.finish({"type": "error", "code": "worker_unavailable", "message": message, "fatal": True})

    def _read(self):
        try:
            while not self.closed.is_set():
                line = self.process.stdout.readline(MAX_WORKER_LINE + 1)
                if not line:
                    if self.stopping:
                        return
                    raise RuntimeError("Worker exited before gateway shutdown")
                if len(line) > MAX_WORKER_LINE or not line.endswith("\n"):
                    raise RuntimeError("Worker JSONL line exceeds protocol limit or is incomplete")
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise RuntimeError("Worker event must be an object")
                kind = event.get("type")
                if kind == "status":
                    self.stage = str(event.get("stage", "loading"))[:128]
                    continue
                if kind == "ready":
                    if self.ready is not None:
                        raise RuntimeError("Duplicate worker ready event")
                    for key in ("context_limit", "max_output_tokens"):
                        if type(event.get(key)) is not int or event[key] <= 0:
                            raise RuntimeError("Worker ready limits are missing")
                    if not isinstance(event.get("model"), str) or not event["model"]:
                        raise RuntimeError("Worker model alias is missing")
                    self.ready, self.stage = event, "ready"
                    continue
                with self.lock:
                    job = self.active
                if not job or event.get("id") != job.id:
                    if kind == "error" and event.get("fatal") is True:
                        raise RuntimeError(str(event.get("message", "Fatal worker failure")))
                    raise RuntimeError("Worker event does not belong to the inflight request")
                if kind == "delta":
                    text = event.get("text")
                    if not isinstance(text, str):
                        raise RuntimeError("Worker delta text is not a string")
                    if text and job.first_delta is None:
                        job.first_delta = time.perf_counter()
                    if not job.cancelled.is_set():
                        try:
                            job.events.put_nowait(text)
                        except queue.Full:
                            self.cancel(job, "slow_consumer")
                elif kind in ("done", "error"):
                    if kind == "done":
                        usage, metrics = event.get("usage"), event.get("metrics")
                        if event.get("finish_reason") not in ("stop", "length", "cancelled"):
                            raise RuntimeError("Worker terminal finish reason is invalid")
                        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                            raise RuntimeError("Worker usage counters are missing or invalid")
                        if not isinstance(metrics, dict) or not {"first_token_s", "decode_s", "decode_tokens", "decode_tok_s", "e2e_s"} <= set(metrics):
                            raise RuntimeError("Worker terminal timing fields are missing")
                        if type(metrics["decode_tokens"]) is not int or metrics["decode_tokens"] < 0:
                            raise RuntimeError("Worker decode token count is invalid")
                        for key in ("first_token_s", "decode_s", "decode_tok_s", "e2e_s"):
                            value = metrics[key]
                            if value is None and key in ("first_token_s", "decode_tok_s"):
                                continue
                            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                                raise RuntimeError("Worker timing is invalid")
                    job.finish(event)
                    if kind == "done" and not event.get("fatal") and not self.stopping and not self.failure:
                        self.stage = "ready"
                    if event.get("fatal") is True:
                        self._fail(str(event.get("message", "Fatal worker failure")))
                else:
                    raise RuntimeError("Unknown worker event type")
        except Exception as exc:
            if not self.closed.is_set():
                self._fail(str(exc))

    def _dispatch(self):
        while not self.closed.is_set():
            try:
                job = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if job.cancelled.is_set():
                    job.finish({"type": "done", "finish_reason": "cancelled", "usage": {}, "metrics": {}})
                    continue
                with self.write_lock:
                    with self.lock:
                        failed = self.failure or ("Server is shutting down" if self.stopping else None)
                        if not failed:
                            self.active = job
                    if failed:
                        job.finish({"type": "error", "code": "worker_unavailable", "message": failed})
                        continue
                    job.started = time.perf_counter()
                    try:
                        self._send(dict(job.payload, type="generate", id=job.id))
                    except Exception as exc:
                        self._fail(str(exc))
                if job.cancelled.is_set():
                    self.cancel(job)
                # Cancellation/disconnection never short-circuits this wait.
                while not job.done.wait(0.2) and not self.closed.is_set():
                    pass
                with self.lock:
                    if self.active is job:
                        self.active = None
            finally:
                self.pending.task_done()

    def submit(self, payload):
        with self.lock:
            if self.closed.is_set() or self.failure or self.stopping:
                raise Rejected(503, "worker_unavailable", "Worker is unavailable")
            if self.ready is None:
                raise Rejected(503, "model_loading", "Model is still loading; inspect /health")
        job = Job(payload)
        try:
            self.pending.put_nowait(job)
        except queue.Full:
            raise Rejected(429, "queue_full", "Request queue is full") from None
        return job

    def cancel(self, job, reason="client_disconnected"):
        job.cancelled.set()
        with self.lock:
            active = self.active is job
        with job.lock:
            if not active or job.done.is_set() or job.cancel_sent:
                return
            job.cancel_sent = True
        try:
            self._send({"type": "cancel", "id": job.id, "reason": reason})
        except Exception as exc:
            self._fail(str(exc))

    def close(self, timeout=30):
        if self.closed.is_set():
            return self.process.returncode == 0 and self.failure is None
        with self.lock:
            self.stopping = True
            active = self.active
        if active:
            self.cancel(active, "server_shutdown")
            if not active.done.wait(timeout):
                # Do not kill CUDA owner or claim that resources were retired.
                return False
        try:
            self._send({"type": "shutdown"})
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        # The child may emit close_failed immediately before exiting. Drain its
        # protocol before marking closed, so that event is never discarded.
        self.reader.join(timeout=1)
        if self.reader.is_alive():
            return False
        if self.process.returncode != 0:
            self._fail("Worker exited with code " + str(self.process.returncode))
        self.closed.set()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass
        return self.failure is None


def validate_request(data, chat, ready, max_prompt_chars):
    if not isinstance(data, dict):
        raise Rejected(400, "invalid_request", "JSON body must be an object")
    allowed = {"model", "prompt", "messages", "max_tokens", "max_completion_tokens", "temperature",
               "stream", "stop", "mode", "n", "stream_options"}
    if set(data) - allowed:
        raise Rejected(400, "unsupported_parameter", "Unsupported parameters: " + ", ".join(sorted(set(data) - allowed)))
    model = data.get("model", ready["model"])
    if model != ready["model"]:
        raise Rejected(404, "model_not_found", "Requested model is not served")
    if data.get("temperature", 0) != 0 or isinstance(data.get("temperature"), bool):
        raise Rejected(400, "unsupported_sampling", "Only deterministic temperature=0 is supported")
    if type(data.get("n", 1)) is not int or data.get("n", 1) != 1:
        raise Rejected(400, "unsupported_n", "Only n=1 is supported")
    if "max_tokens" in data and "max_completion_tokens" in data:
        raise Rejected(400, "invalid_request", "Specify one output-token limit")
    maximum = data.get("max_tokens", data.get("max_completion_tokens", min(256, ready["max_output_tokens"])))
    if type(maximum) is not int or not 1 <= maximum <= ready["max_output_tokens"]:
        raise Rejected(400, "invalid_max_tokens", "Output token limit is outside worker bounds")
    mode = data.get("mode", "greedy")
    if mode not in ready.get("modes", ["greedy"]):
        raise Rejected(400, "unsupported_mode", "Unsupported decoding mode")
    stream = data.get("stream", False)
    if type(stream) is not bool:
        raise Rejected(400, "invalid_stream", "stream must be a boolean")
    if "stream_options" in data and data["stream_options"] != {"include_usage": True}:
        raise Rejected(400, "unsupported_stream_options", "Only include_usage=true is supported")
    if chat:
        messages = data.get("messages")
        if ("prompt" in data or not isinstance(messages, list) or len(messages) != 1 or
            not isinstance(messages[0], dict) or set(messages[0]) != {"role", "content"} or
            messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str)):
            raise Rejected(400, "unsupported_messages", "This worker supports exactly one text user message")
        prompt = messages[0]["content"]
        payload = {"messages": messages}
    else:
        prompt = data.get("prompt")
        if "messages" in data or not isinstance(prompt, str):
            raise Rejected(400, "unsupported_prompt", "prompt must be one string")
        payload = {"prompt": prompt}
    if not prompt.strip() or len(prompt) > max_prompt_chars:
        raise Rejected(400, "invalid_prompt", "Prompt is empty or exceeds the character limit")
    stop = data.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 4 or any(not isinstance(x, str) or not 1 <= len(x) <= 128 for x in stop):
        raise Rejected(400, "invalid_stop", "stop supports up to four nonempty literal strings, each <=128 characters")
    payload.update(model=model, max_tokens=maximum, temperature=0, mode=mode, stop=stop)
    # Exact token/context accounting belongs to the worker's actual tokenizer.
    return payload, stream


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Do not log prompt bodies, authorization, or query strings.
        sys.stderr.write("http " + getattr(self, "command", "?") + " " + self.path.split("?", 1)[0][:128] + "\n")

    def _authorized(self):
        key = self.server.api_key
        return not key or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key)

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authorized():
            return self._json(401, {"error": {"code": "unauthorized", "message": "Bearer token required"}})
        if self.path == "/health":
            return self._json(200, self.server.bridge.health())
        if self.path == "/v1/models":
            health = self.server.bridge.health()
            model = (health["worker"] or {}).get("model", "eva")
            return self._json(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "local"}]})
        self._json(404, {"error": {"code": "not_found", "message": "Unknown endpoint"}})

    def _sse(self, value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.wfile.write(("data: " + text + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        job = None
        streaming = False
        headers_sent = False
        try:
            if not self._authorized():
                raise Rejected(401, "unauthorized", "Bearer token required")
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                raise Rejected(404, "not_found", "Unknown endpoint")
            if self.headers.get("Transfer-Encoding"):
                raise Rejected(400, "unsupported_encoding", "Use a Content-Length JSON body")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise Rejected(400, "invalid_length", "Invalid Content-Length") from None
            if not 1 <= length <= self.server.max_body_bytes:
                raise Rejected(413, "request_too_large", "JSON body exceeds configured limit")
            self.connection.settimeout(15)
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                raise Rejected(400, "invalid_json", "Malformed JSON body") from None
            health = self.server.bridge.health()
            if not health["ready"]:
                raise Rejected(503, "worker_not_ready", "Worker is loading or unavailable; inspect /health")
            chat = self.path == "/v1/chat/completions"
            payload, streaming = validate_request(data, chat, health["worker"], self.server.max_prompt_chars)
            job = self.server.bridge.submit(payload)
            created = int(time.time())
            def frame(text=None, final=None):
                choice = {"index": 0, "finish_reason": final}
                choice["delta" if chat else "text"] = ({"content": text} if text is not None else {}) if chat else (text or "")
                return {"id": job.id, "object": "chat.completion.chunk" if chat else "text_completion",
                        "created": created, "model": payload["model"], "choices": [choice]}
            if streaming:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                headers_sent = True
                if chat:
                    initial = frame()
                    initial["choices"][0]["delta"] = {"role": "assistant"}
                    self._sse(initial)
            texts = []
            last_ping = time.perf_counter()
            while True:
                if time.perf_counter() - job.created > self.server.request_timeout:
                    self.server.bridge.cancel(job, "request_timeout")
                    raise Rejected(504, "request_timeout", "Request timed out; worker cancellation is pending")
                try:
                    text = job.events.get(timeout=0.2)
                    if streaming:
                        self._sse(frame(text))
                    else:
                        texts.append(text)
                except queue.Empty:
                    if job.done.is_set():
                        break
                    if streaming and time.perf_counter() - last_ping > 2:
                        self.wfile.write(b": waiting\n\n")
                        self.wfile.flush()
                        last_ping = time.perf_counter()
            result = job.result
            if result["type"] == "error":
                raise Rejected(503 if result.get("fatal") else 400, result.get("code", "worker_error"), result.get("message", "Worker error"))
            final = frame(final=result.get("finish_reason", "stop"))
            final["usage"] = result.get("usage", {})
            if "prompt_tokens" in final["usage"] and "completion_tokens" in final["usage"]:
                final["usage"] = dict(final["usage"], total_tokens=final["usage"]["prompt_tokens"] + final["usage"]["completion_tokens"])
            final["metrics"] = {"engine": result.get("metrics", {}), "server": job.server_metrics()}
            if streaming:
                self._sse(final)
                self._sse("[DONE]")
            else:
                final["object"] = "chat.completion" if chat else "text_completion"
                content = "".join(texts)
                if chat:
                    final["choices"][0].pop("delta")
                    final["choices"][0]["message"] = {"role": "assistant", "content": content}
                else:
                    final["choices"][0]["text"] = content
                self._json(200, final)
        except Rejected as exc:
            error = {"error": {"message": exc.message, "type": "invalid_request_error", "code": exc.code}}
            try:
                if headers_sent:
                    self._sse(error)
                    self._sse("[DONE]")
                else:
                    self.close_connection = True
                    self._json(exc.status, error)
            except (OSError, ValueError):
                pass
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            if job:
                self.server.bridge.cancel(job)
        finally:
            if job and not job.done.is_set():
                self.server.bridge.cancel(job)


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, bridge, *, api_key="", max_body_bytes=1024 * 1024,
                 max_prompt_chars=131072, request_timeout=600):
        self.bridge, self.api_key = bridge, api_key
        self.max_body_bytes, self.max_prompt_chars = max_body_bytes, max_prompt_chars
        self.request_timeout = request_timeout
        super().__init__(address, GatewayHandler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-queue", type=int, default=2)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--api-key-env", default="EVA_API_KEY")
    p.add_argument("--worker", nargs=argparse.REMAINDER, required=True, help="Worker command, placed last")
    args = p.parse_args(argv)
    if not args.worker or not 1 <= args.max_queue <= 32 or not 1 <= args.request_timeout <= 86400:
        p.error("Need worker command, queue1..32, timeout1..86400")
    key = os.environ.get(args.api_key_env, "")
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and len(key) < 16:
        p.error("Non-loopback binding requires an API key of at least16 characters via the configured environment variable")
    # Reserve the listener before spawning a potentially expensive GPU owner.
    # No handler is served until the bridge has been attached below.
    httpd = Gateway((args.host, args.port), None, api_key=key, request_timeout=args.request_timeout)
    bridge = None
    try:
        bridge = WorkerBridge(args.worker, max_queue=args.max_queue)
        httpd.bridge = bridge
        sys.stderr.write("HTTP listening; /health reports loading until worker ready. Single inference worker.\n")
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    finally:
        httpd.server_close()
        if bridge is not None and not bridge.close():
            sys.stderr.write("Worker has not acknowledged safe retirement; it was not forcibly killed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
