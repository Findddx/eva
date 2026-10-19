"""CPU tests for streaming HTTP, cancellation, queuing and metrics."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location("eva_test_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S, C = load("server"), load("client")
FAKE = r'''
import json,queue,sys,threading,time
q=queue.Queue();cancelled=set();closing=threading.Event();lock=threading.Lock()
def emit(x):
 with lock: print(json.dumps(x,ensure_ascii=False),flush=True)
def reader():
 for line in sys.stdin:
  x=json.loads(line)
  if x['type']=='cancel':cancelled.add(x['id'])
  elif x['type']=='shutdown':closing.set();q.put(None);return
  else:q.put(x)
threading.Thread(target=reader,daemon=True).start()
emit({'type':'status','stage':'CPU_fixture_loading'})
time.sleep(float(sys.argv[1]))
emit({'type':'ready','model':'eva','context_limit':1024,'max_output_tokens':1024,'modes':['greedy','lookup'],'scope':'CPU protocol fixture only'})
while not closing.is_set():
 x=q.get()
 if x is None:break
 ident=x['id'];prompt=x.get('prompt',x.get('messages',[{'content':''}])[0]['content'])
 if prompt in ('error','fatal'):
  emit({'type':'error','id':ident,'code':'fixture_error','message':'CPU fixture','fatal':prompt=='fatal'})
  if prompt=='fatal':break
  continue
 if prompt!='zero':emit({'type':'delta','id':ident,'text':'你'})
 if prompt=='hold':
  while ident not in cancelled and not closing.is_set():time.sleep(.005)
  time.sleep(.2)
 if ident not in cancelled and prompt!='zero':emit({'type':'delta','id':ident,'text':'好'+chr(10)})
 emit({'type':'done','id':ident,'finish_reason':'cancelled' if ident in cancelled else 'stop','usage':{'prompt_tokens':3,'completion_tokens':1 if prompt=='zero' else 2},'metrics':{'first_token_s':.01,'decode_tokens':0 if prompt=='zero' else 1,'decode_s':0 if prompt=='zero' else .02,'decode_tok_s':None if prompt=='zero' else 50,'e2e_s':.03,'scope':'CPU fixture numbers, not measured GPU performance'}})
if len(sys.argv)>2 and sys.argv[2]=='close_failed':
 emit({'type':'error','id':None,'code':'close_failed','message':'CPU fixture cleanup failed','fatal':True})
 sys.exit(1)
'''


def eventually(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for CPU protocol fixture")


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="protocol-cpu-", dir=HERE)
        self.worker = Path(self.tmp.name) / "worker.py"
        self.worker.write_text(FAKE, encoding="utf-8")
        self.bridge = None
        self.httpd = None
        self.expected_close = True

    def start(self, delay="0", max_queue=2, api_key="", timeout=5, shutdown_case="normal"):
        self.bridge = S.WorkerBridge([sys.executable, "-u", "-B", str(self.worker), delay, shutdown_case], max_queue=max_queue)
        self.httpd = S.Gateway(("127.0.0.1", 0), self.bridge, api_key=api_key, request_timeout=timeout)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.httpd.server_port)

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.bridge:
            self.assertEqual(self.bridge.close(timeout=3), self.expected_close,
                             "Successful retirement must not hide worker failure")
        self.tmp.cleanup()

    def ready(self):
        eventually(lambda: self.bridge.health()["ready"])

    def test_health_during_load_and_early_request_rejection(self):
        self.start(delay=".4")
        with urllib.request.urlopen(self.url + "/health") as r:
            self.assertFalse(json.load(r)["ready"])
        with self.assertRaises(S.Rejected) as cm:
            self.bridge.submit({"prompt": "x"})
        self.assertEqual(cm.exception.status, 503)
        self.ready()

    def test_normal_shutdown_eof_is_not_worker_failure(self):
        self.start()
        self.ready()
        self.assertTrue(self.bridge.close(timeout=3))
        self.assertEqual(self.bridge.process.returncode, 0)
        self.assertIsNone(self.bridge.failure)
        self.assertFalse(self.bridge.reader.is_alive())
        self.assertTrue(self.bridge.close(timeout=3))

    def test_close_failed_final_event_and_exit_are_not_success(self):
        self.start(shutdown_case="close_failed")
        self.expected_close = False
        self.ready()
        self.assertFalse(self.bridge.close(timeout=3))
        self.assertEqual(self.bridge.process.returncode, 1)
        self.assertIn("CPU fixture cleanup failed", self.bridge.failure)
        self.assertFalse(self.bridge.close(timeout=3))

    def test_cancel_does_not_release_until_worker_terminal(self):
        self.start(max_queue=1)
        self.ready()
        first = self.bridge.submit({"prompt": "hold"})
        eventually(lambda: first.first_delta is not None)
        second = self.bridge.submit({"prompt": "next"})
        with self.assertRaises(S.Rejected) as cm:
            self.bridge.submit({"prompt": "overflow"})
        self.assertEqual(cm.exception.status, 429)
        self.bridge.cancel(first)
        time.sleep(.04)
        self.assertFalse(first.done.is_set())
        self.assertIsNone(second.started)
        self.assertTrue(first.done.wait(2))
        self.assertTrue(second.done.wait(2))
        self.assertGreaterEqual(second.started, first.finished)

    def test_queued_cancel_never_becomes_generate(self):
        self.start(max_queue=1)
        self.ready()
        first = self.bridge.submit({"prompt": "hold"})
        eventually(lambda: first.first_delta is not None)
        second = self.bridge.submit({"prompt": "next"})
        self.bridge.cancel(second)
        self.bridge.cancel(first)
        self.assertTrue(second.done.wait(2))
        self.assertIsNone(second.started)
        self.assertEqual(second.result["finish_reason"], "cancelled")

    def test_http_sse_utf8_and_engine_client_scopes(self):
        self.start()
        self.ready()
        out = io.StringIO()
        result = C.request_once(self.url, "hi", output=out)
        self.assertEqual(out.getvalue(), "你好\n")
        self.assertEqual(result["text"], out.getvalue())
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertIn("not measured GPU", result["metrics"]["engine"]["scope"])
        self.assertGreater(result["client"]["e2e_s"], 0)
        self.assertIsNotNone(result["client"]["first_text_s"])
        self.assertIn("queue_s", result["metrics"]["server"])

    def test_zero_decode_does_not_invent_rate(self):
        self.start()
        self.ready()
        result = C.request_once(self.url, "zero", output=io.StringIO())
        self.assertIsNone(result["client"]["first_text_s"])
        self.assertIsNone(result["metrics"]["engine"]["decode_tok_s"])

    def test_completions_nonstream_contract(self):
        self.start()
        self.ready()
        request = urllib.request.Request(self.url + "/v1/completions",
                                        data=json.dumps({"prompt": "hello", "stream": False}).encode(),
                                        headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            value = json.load(response)
        self.assertEqual(value["choices"][0]["text"], "你好\n")
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        self.assertEqual(value["usage"]["completion_tokens"], 2)
        self.assertIn("engine", value["metrics"])

    def test_recoverable_error_and_fatal_worker_failure(self):
        self.start()
        self.ready()
        with self.assertRaisesRegex(RuntimeError, "fixture_error"):
            C.request_once(self.url, "error", output=io.StringIO())
        self.assertEqual(C.request_once(self.url, "good", output=io.StringIO())["text"], "你好\n")
        with self.assertRaises(RuntimeError):
            C.request_once(self.url, "fatal", output=io.StringIO())
        self.expected_close = False
        eventually(lambda: self.bridge.failure is not None)
        with self.assertRaises(S.Rejected):
            self.bridge.submit({"prompt": "never_run"})

    def test_real_http_disconnect_requests_safe_cancel(self):
        self.start()
        self.ready()
        body = json.dumps({"messages": [{"role": "user", "content": "hold"}], "stream": True}).encode()
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        response = urllib.request.urlopen(request)
        iterator = C.iter_sse(response)
        next(iterator)  # role frame
        next(iterator)  # actual first text
        response.close()
        eventually(lambda: self.bridge.active is None, timeout=5)
        self.assertTrue(self.bridge.health()["ready"])

    def test_auth_and_unsupported_messages_before_work(self):
        self.start(api_key="fixture-key")
        self.ready()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.url + "/health")
        self.assertEqual(cm.exception.code, 401)
        ready = self.bridge.ready
        bad = [
            {"messages": [{"role": "system", "content": "x"}]},
            {"messages": [{"role": "user", "content": "x"}] * 2},
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            {"messages": [{"role": "user", "content": "x"}], "prompt_file": "/private"},
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": True},
            {"messages": [{"role": "user", "content": "x"}], "temperature": 1},
        ]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(S.Rejected):
                S.validate_request(payload, True, ready, 100)
        self.assertIsNone(self.bridge.active)


class ParserTests(unittest.TestCase):
    def test_multiline_crlf_comments_and_partial_event(self):
        self.assertEqual(list(C.iter_sse(io.BytesIO(b": ping\r\ndata: a\r\ndata: b\r\n\r\n"))), ["a\nb"])
        with self.assertRaises(ValueError):
            list(C.iter_sse(io.BytesIO(b"data: unfinished\n")))
        with self.assertRaises(ValueError):
            list(C.iter_sse(io.BytesIO(b"data: abcdef\n\n"), max_event_bytes=4))

    def test_parser_does_not_treat_text_as_done(self):
        events = list(C.iter_sse(io.BytesIO(b'data: {"text":"[DONE]"}\n\ndata: [DONE]\n\n')))
        self.assertEqual(len(events), 2)
        self.assertEqual(json.loads(events[0])["text"], "[DONE]")


if __name__ == "__main__":
    unittest.main()
