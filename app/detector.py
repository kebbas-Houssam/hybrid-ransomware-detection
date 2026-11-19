"""I/O monitoring, feature computation and the detection engine."""

import logging
import random
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from training.data import FEATURE_COLUMNS, pad_sequence

from app.config import config
from app.model_manager import ModelManager
from app.schemas import DetectionResult, IOEvent, ProcessStats, SystemStatus, TraceBatch

logger = logging.getLogger(__name__)


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """13 features in the model's input order (training/data.py::FEATURE_COLUMNS)."""
    vector = np.array([features.get(name, 0.0) for name in FEATURE_COLUMNS], dtype=np.float64)
    return np.nan_to_num(vector, nan=0.0)


def assess_risk(prediction: float) -> Tuple[str, bool]:
    """Assess risk level based on prediction"""
    if prediction >= 0.8:
        return "CRITICAL", True
    elif prediction >= 0.6:
        return "HIGH", True
    elif prediction >= 0.4:
        return "MEDIUM", False
    elif prediction >= 0.2:
        return "LOW", False
    else:
        return "MINIMAL", False


class IOMonitor:
    """Process I/O monitoring based on psutil counters.

    psutil exposes per-process read/write counters but not individual file
    operations, so this built-in loop synthesizes file events from those
    counters (``_simulate_file_operations``). Real file events come from
    ``monitor/windows_monitor.py`` through ``POST /api/predict/trace``.
    """

    def __init__(self):
        self.process_stats = {}
        self.event_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.monitoring = False
        self.monitor_thread = None
        self.feature_calculator = FeatureCalculator()

    def start_monitoring(self):
        """Start I/O monitoring"""
        if self.monitoring:
            return False

        logger.info("Starting I/O monitoring...")
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        return True

    def stop_monitoring(self):
        """Stop I/O monitoring"""
        logger.info("Stopping I/O monitoring...")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                current_time = time.time()

                # Get current processes
                for proc in psutil.process_iter(['pid', 'name', 'create_time']):
                    try:
                        proc_info = proc.info
                        pid = proc_info['pid']
                        name = proc_info['name']

                        # Skip system processes
                        if pid < 10 or name in ['System', 'smss.exe', 'csrss.exe']:
                            continue

                        # Initialize process stats if new
                        if pid not in self.process_stats:
                            self.process_stats[pid] = ProcessStats(
                                process_id=pid,
                                process_name=name,
                                start_time=current_time
                            )

                        # Get I/O statistics
                        try:
                            io_counters = proc.io_counters()
                            stats = self.process_stats[pid]

                            # Update stats
                            stats.read_ops = io_counters.read_count
                            stats.write_ops = io_counters.write_count
                            stats.total_bytes_read = io_counters.read_bytes
                            stats.total_bytes_written = io_counters.write_bytes

                            # Simulate file operations for testing
                            if io_counters.write_count > 0:
                                self._simulate_file_operations(pid, name, current_time)

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Clean old processes
                self._cleanup_old_processes(current_time)

                time.sleep(config.UPDATE_INTERVAL)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)

    def _simulate_file_operations(self, pid: int, name: str, timestamp: float):
        """Simulate file operations for testing"""
        try:
            stats = self.process_stats[pid]

            operations = ['read', 'write', 'create']
            file_extensions = ['.txt', '.doc', '.pdf', '.jpg', '.exe', '.dll']

            for _ in range(random.randint(1, 3)):
                operation = random.choice(operations)
                extension = random.choice(file_extensions)
                fake_path = f"C:\\Users\\Document\\file_{random.randint(1000,9999)}{extension}"

                event = IOEvent(
                    timestamp=timestamp,
                    process_id=pid,
                    process_name=name,
                    file_path=fake_path,
                    operation=operation,
                    bytes_transferred=random.randint(512, 65536),
                    offset=random.randint(0, 1000000)
                )

                stats.file_operations.append(event)
                stats.unique_files.add(fake_path)
                self.event_buffer.append(event)

                # Keep only recent events
                cutoff_time = timestamp - config.MONITORING_WINDOW
                stats.file_operations = [
                    op for op in stats.file_operations
                    if op.timestamp > cutoff_time
                ]

        except Exception as e:
            logger.error(f"Error simulating file operations: {e}")

    def _cleanup_old_processes(self, current_time: float):
        """Remove old process statistics"""
        cutoff_time = current_time - 300  # 5 minutes
        old_pids = [
            pid for pid, stats in self.process_stats.items()
            if stats.start_time < cutoff_time
        ]

        for pid in old_pids:
            del self.process_stats[pid]

    def get_process_features(self, pid: int) -> Optional[Dict[str, float]]:
        """Calculate features for a specific process"""
        if pid not in self.process_stats:
            return None

        stats = self.process_stats[pid]
        return self.feature_calculator.calculate_features(stats)

class FeatureCalculator:
    """Calculate features from process I/O statistics"""

    def calculate_features(self, stats: ProcessStats) -> Dict[str, float]:
        """Calculate the 13 features used by the model"""
        try:
            current_time = time.time()
            recent_events = [
                event for event in stats.file_operations
                if current_time - event.timestamp <= config.MONITORING_WINDOW
            ]

            if not recent_events:
                return self._default_features()

            # Separate read and write operations
            read_events = [e for e in recent_events if e.operation == 'read']
            write_events = [e for e in recent_events if e.operation == 'write']

            read_count = len(read_events)
            write_count = len(write_events)
            total_ops = read_count + write_count

            if total_ops == 0:
                return self._default_features()

            # Calculate features
            features = {}

            features['read_write_ratio'] = read_count / write_count if write_count > 0 else read_count
            features['war_ratio'] = self._calculate_war_ratio(recent_events) / total_ops if total_ops > 0 else 0.0
            features['wss'] = len(stats.unique_files)
            features['entropy'] = self._calculate_entropy(recent_events)
            features['read_pct'] = read_count / total_ops if total_ops > 0 else 0.0
            features['write_pct'] = write_count / total_ops if total_ops > 0 else 0.0
            features['repeat_ratio'] = self._calculate_repeat_ratio(recent_events)
            features['read_entropy'] = self._calculate_entropy(read_events) if read_events else 0.0
            features['write_entropy'] = self._calculate_entropy(write_events) if write_events else 0.0
            features['total_ops'] = float(total_ops)

            unique_write_files = len(set(e.file_path for e in write_events))
            features['write_to_unique_ratio'] = write_count / unique_write_files if unique_write_files > 0 else 0.0
            features['avg_offset_gap'] = self._calculate_avg_offset_gap(recent_events)
            features['burstiness'] = self._calculate_burstiness(recent_events)

            return features

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return self._default_features()

    def _default_features(self) -> Dict[str, float]:
        """Return default feature values"""
        return {
            'read_write_ratio': 0.0, 'war_ratio': 0.0, 'wss': 0.0, 'entropy': 0.0,
            'read_pct': 0.0, 'write_pct': 0.0, 'repeat_ratio': 0.0,
            'read_entropy': 0.0, 'write_entropy': 0.0, 'total_ops': 0.0,
            'write_to_unique_ratio': 0.0, 'avg_offset_gap': 0.0, 'burstiness': 0.0
        }

    def _calculate_war_ratio(self, events: List[IOEvent]) -> int:
        war_count = 0
        file_reads = set()
        for event in sorted(events, key=lambda x: x.timestamp):
            if event.operation == 'read':
                file_reads.add(event.file_path)
            elif event.operation == 'write' and event.file_path in file_reads:
                war_count += 1
        return war_count

    def _calculate_entropy(self, events: List[IOEvent]) -> float:
        if not events:
            return 0.0
        file_counts = defaultdict(int)
        for event in events:
            file_counts[event.file_path] += 1
        total = len(events)
        entropy = 0.0
        for count in file_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _calculate_repeat_ratio(self, events: List[IOEvent]) -> float:
        if not events:
            return 0.0
        file_counts = defaultdict(int)
        for event in events:
            file_counts[event.file_path] += 1
        repeated = sum(1 for count in file_counts.values() if count > 1)
        return repeated / len(file_counts) if file_counts else 0.0

    def _calculate_avg_offset_gap(self, events: List[IOEvent]) -> float:
        if len(events) < 2:
            return 0.0
        offsets = [e.offset for e in events if e.offset is not None]
        if len(offsets) < 2:
            return 0.0
        offsets.sort()
        gaps = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]
        return np.mean(gaps) if gaps else 0.0

    def _calculate_burstiness(self, events: List[IOEvent]) -> float:
        if len(events) < 2:
            return 0.0
        timestamps = [e.timestamp for e in events]
        timestamps.sort()
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if not intervals:
            return 0.0
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        if mean_interval == 0:
            return 0.0
        return std_interval / mean_interval

class RansomwareDetector:
    """Main detection engine combining I/O monitoring and ML inference"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.io_monitor = IOMonitor()
        self.detection_results = deque(maxlen=100)
        self.alert_count = 0
        self.start_time = time.time()
        self.detection_thread = None
        self.detecting = False
        # State for traces pushed by monitor/ and simulation/ (POST /api/predict/trace)
        self.trace_stats: Dict[int, ProcessStats] = {}
        self.trace_sequences: Dict[int, deque] = {}
        self._trace_lock = threading.Lock()

    def load_models(self, h5_path: Optional[str] = None, xgb_path: Optional[str] = None,
                    scaler_path: Optional[str] = None, xgb_scaler_path: Optional[str] = None) -> bool:
        """Load the detection models"""
        return self.model_manager.load_models(h5_path, xgb_path, scaler_path, xgb_scaler_path)

    def start_detection(self) -> bool:
        """Start the detection system"""
        if not self.model_manager.model_loaded:
            logger.error("Models not loaded")
            return False

        logger.info("Starting ransomware detection system...")

        if not self.io_monitor.start_monitoring():
            logger.error("Failed to start I/O monitoring")
            return False

        self.detecting = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

        logger.info("Ransomware detection system started!")
        return True

    def stop_detection(self):
        """Stop the detection system"""
        logger.info("Stopping ransomware detection system...")
        self.detecting = False
        self.io_monitor.stop_monitoring()

        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        logger.info("Detection system stopped")

    def _detection_loop(self):
        """Main detection loop"""
        sequence_buffer = {}

        while self.detecting:
            try:
                current_time = time.time()

                for pid, stats in self.io_monitor.process_stats.items():
                    try:
                        features = self.io_monitor.get_process_features(pid)
                        if not features:
                            continue

                        feature_vector = features_to_vector(features)

                        if pid not in sequence_buffer:
                            sequence_buffer[pid] = deque(maxlen=config.SEQUENCE_LENGTH)

                        sequence_buffer[pid].append(feature_vector)

                        if len(sequence_buffer[pid]) == config.SEQUENCE_LENGTH:
                            sequence = np.array(list(sequence_buffer[pid]))

                            # ModelManager.predict applies the feature scaler
                            pred_primary, pred_hybrid = self.model_manager.predict(sequence)
                            risk_level, alert = assess_risk(pred_hybrid)

                            if alert:
                                self.alert_count += 1

                            result = DetectionResult(
                                timestamp=current_time,
                                process_id=pid,
                                process_name=stats.process_name,
                                prediction_primary=pred_primary,
                                prediction_hybrid=pred_hybrid,
                                risk_level=risk_level,
                                confidence=max(pred_hybrid, 1 - pred_hybrid),
                                features=features,
                                alert=alert
                            )

                            self.detection_results.append(result)

                            if alert:
                                logger.warning(f"RANSOMWARE ALERT: Process {stats.process_name} (PID: {pid}) - Risk: {pred_hybrid:.3f}")

                    except Exception as e:
                        logger.error(f"Error processing PID {pid}: {e}")
                        continue

                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)

    def _assess_risk(self, prediction: float) -> Tuple[str, bool]:
        return assess_risk(prediction)

    def process_trace_batch(self, batch: TraceBatch) -> DetectionResult:
        """Score a batch of file operations reported for one process.

        The batch is added to the process' event history, the 13 features are
        computed over the last 5 s window, and the window is appended to the
        process' sequence (front-padded until 50 windows are available).
        """
        pid = batch.process_id
        now = time.time()
        with self._trace_lock:
            stats = self.trace_stats.get(pid)
            if stats is None:
                stats = ProcessStats(process_id=pid, process_name=batch.process_name, start_time=now)
                self.trace_stats[pid] = stats
            for e in batch.events:
                op = e.operation_type.lower()
                event = IOEvent(timestamp=e.timestamp, process_id=pid, process_name=batch.process_name,
                                file_path=e.file_path, operation=op,
                                bytes_transferred=e.size, offset=e.offset)
                stats.file_operations.append(event)
                stats.unique_files.add(e.file_path)
                if op == 'read':
                    stats.read_ops += 1
                    stats.total_bytes_read += e.size
                elif op == 'write':
                    stats.write_ops += 1
                    stats.total_bytes_written += e.size
            cutoff = now - config.MONITORING_WINDOW
            stats.file_operations = [op for op in stats.file_operations if op.timestamp > cutoff]

            features = self.io_monitor.feature_calculator.calculate_features(stats)
            history = self.trace_sequences.setdefault(pid, deque(maxlen=config.SEQUENCE_LENGTH))
            history.append(features_to_vector(features))
            sequence = pad_sequence(np.array(history), config.SEQUENCE_LENGTH)

        pred_primary, pred_hybrid = self.model_manager.predict(sequence)
        risk_level, alert = assess_risk(pred_hybrid)
        if alert:
            self.alert_count += 1
            logger.warning(f"RANSOMWARE ALERT: Process {batch.process_name} (PID: {pid}) - Risk: {pred_hybrid:.3f}")

        result = DetectionResult(
            timestamp=now,
            process_id=pid,
            process_name=batch.process_name,
            prediction_primary=pred_primary,
            prediction_hybrid=pred_hybrid,
            risk_level=risk_level,
            confidence=max(pred_hybrid, 1 - pred_hybrid),
            features=features,
            alert=alert
        )
        self.detection_results.append(result)
        return result

    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        last_detection = None
        if self.detection_results:
            last_detection = datetime.fromtimestamp(self.detection_results[-1].timestamp).isoformat()

        return SystemStatus(
            status="RUNNING" if self.detecting else "STOPPED",
            monitored_processes=len(self.io_monitor.process_stats),
            alerts_count=self.alert_count,
            uptime=time.time() - self.start_time,
            last_detection=last_detection
        )

    def get_recent_detections(self, limit: int = 50) -> List[DetectionResult]:
        """Get recent detection results"""
        return list(self.detection_results)[-limit:]
