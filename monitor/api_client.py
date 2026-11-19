"""Small HTTP client for the FastAPI detection backend (app/main.py).

Shared by monitor/windows_monitor.py and simulation/.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://localhost:8000"


class DetectionClient:
    """Thin wrapper around the /api endpoints."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout: float = 5.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> Optional[Dict]:
        try:
            r = requests.get(f"{self.server_url}/api/status", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except requests.RequestException:
            return None

    def is_ready(self) -> bool:
        """Server is reachable and the models are loaded."""
        status = self.status()
        return bool(status and status.get("models"))

    def detections(self, limit: int = 50) -> List[Dict]:
        try:
            r = requests.get(f"{self.server_url}/api/detections", params={"limit": limit}, timeout=self.timeout)
            return r.json().get("detections", []) if r.status_code == 200 else []
        except requests.RequestException:
            return []

    def predict_features(self, features: Dict[str, float]) -> Optional[Dict]:
        try:
            r = requests.post(f"{self.server_url}/api/predict", json=features, timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except requests.RequestException:
            return None

    def predict_trace(self, process_id: int, process_name: str, events: List[Dict]) -> Optional[Dict]:
        """POST a batch of file operations for one process to /api/predict/trace."""
        payload = {"process_id": process_id, "process_name": process_name, "events": events}
        try:
            r = requests.post(f"{self.server_url}/api/predict/trace", json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning(f"/api/predict/trace returned {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            logger.debug(f"Failed to send traces: {e}")
        return None


class TraceBuffer:
    """Collects trace events per process and flushes them as batches.

    A batch is sent when ``flush_interval`` seconds have passed or
    ``max_batch`` events are queued for a process. ``on_result`` receives each
    detection result returned by the server.
    """

    def __init__(self, client: DetectionClient, flush_interval: float = 1.0, max_batch: int = 200,
                 on_result: Optional[Callable[[Dict], None]] = None):
        self.client = client
        self.flush_interval = flush_interval
        self.max_batch = max_batch
        self.on_result = on_result
        self._events: Dict[tuple, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def add(self, trace: Dict):
        """Add one trace: keys timestamp, operation_type, file_path, offset, size, process_id, process_name."""
        key = (int(trace.get("process_id", 0)), str(trace.get("process_name", "unknown")))
        event = {
            "timestamp": float(trace.get("timestamp", time.time())),
            "operation_type": str(trace.get("operation_type", "read")),
            "file_path": str(trace.get("file_path", "")),
            "offset": int(trace.get("offset", 0) or 0),
            "size": int(trace.get("size", 0) or 0),
        }
        flush_now = False
        with self._lock:
            self._events[key].append(event)
            flush_now = len(self._events[key]) >= self.max_batch
        if flush_now:
            self._flush_key(key)

    def _flush_key(self, key):
        with self._lock:
            events = self._events.pop(key, [])
        if events:
            result = self.client.predict_trace(key[0], key[1], events)
            if result and self.on_result:
                self.on_result(result)

    def flush(self):
        with self._lock:
            keys = list(self._events.keys())
        for key in keys:
            self._flush_key(key)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.flush()

    def _loop(self):
        while self._running:
            time.sleep(self.flush_interval)
            try:
                self.flush()
            except Exception as e:
                logger.error(f"Trace flush error: {e}")
