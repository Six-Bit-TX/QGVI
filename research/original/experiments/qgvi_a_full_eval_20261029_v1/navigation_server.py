"""Loopback gateway to one resident, continuously movable Gaussian camera."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import signal
import threading

from navigation_backend import NavigationSession, NavigationError, write, STEPS, validate_cache_namespace

HERE = Path(__file__).resolve().parent


def identity():
    import os
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    return {'pid': os.getpid(), 'start_ticks': int(fields[19])}


def make_handler(scene, session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send(self, value, code=200):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers()
            try: self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            if self.path != '/health': self.send({'ok': False, 'kind': 'not_found'}, 404); return
            alive = session.process is not None and session.process.poll() is None
            self.send({'ok': alive, 'scene': scene, 'status': 'ready' if alive else 'worker_exited',
                       'startup': session.readiness}, 200 if alive else 503)

        def do_POST(self):
            if self.path not in {'/navigate', '/render'}:
                self.send({'ok': False, 'kind': 'not_found'}, 404); return
            try:
                self.connection.settimeout(10)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16000: raise NavigationError('invalid_request_size')
                data = json.loads(self.rfile.read(length))
                required = {'scene', 'frame', 'movement'} if self.path == '/navigate' else {'scene', 'frame'}
                if not isinstance(data, dict) or set(data) not in (required, required | {'cache_namespace'}): raise NavigationError('invalid_request_fields')
                if data['scene'] != scene: raise NavigationError('wrong_scene')
                namespace = validate_cache_namespace(data.get('cache_namespace', 'default'))
                scoped = session.for_namespace(namespace)
                result = scoped.navigate(data['frame'], data['movement']) if self.path == '/navigate' else scoped.propose_view(data['frame'])
                self.send({'ok': True, **result})
            except NavigationError as exc:
                self.send({'ok': False, 'kind': exc.kind, 'message': str(exc), 'details': exc.details}, 422)
            except (ValueError, TypeError, KeyError):
                self.send({'ok': False, 'kind': 'invalid_request'}, 400)
            except Exception as exc:
                self.send({'ok': False, 'kind': 'worker_failure', 'message': type(exc).__name__}, 503)
    return Handler


def serve(scene, port=8782, cache_dir=HERE / 'navigation'):
    status_path = Path(cache_dir) / 'server_status.json'
    state = {'status': 'loading', 'scene': scene, 'port': port, 'worker': identity()}
    stopped = threading.Event(); session = None; server = None
    def publish():
        write(status_path, {**state, 'updated_at': datetime.now(timezone.utc).isoformat()})
    def heartbeat():
        while not stopped.wait(5): publish()
    def halt(*_):
        if server is None: raise KeyboardInterrupt('Stop while loading renderer')
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)
    publish(); thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        session = NavigationSession(scene, cache_dir)
        server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(scene, session))
        # Wait for active render handlers before releasing the resident worker/locks.
        server.daemon_threads = False
        state.update(status='ready', startup=session.readiness); publish()
        server.serve_forever(poll_interval=.2)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error_type=type(exc).__name__)
        if not isinstance(exc, KeyboardInterrupt): raise
    finally:
        stopped.set(); thread.join()
        if server is not None: server.server_close()
        if session is not None: session.close()
        if state['status'] == 'ready': state['status'] = 'closed'
        publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=tuple(STEPS))
    parser.add_argument('--port', type=int, default=8782)
    parser.add_argument('--cache-dir', type=Path, default=HERE / 'navigation')
    args = parser.parse_args(); serve(args.scene, args.port, args.cache_dir)
