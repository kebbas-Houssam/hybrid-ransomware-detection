"""Pydantic models used by the API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class IOEvent(BaseModel):
    timestamp: float
    process_id: int
    process_name: str
    file_path: str
    operation: str  # 'read', 'write', 'create', 'delete'
    bytes_transferred: int
    offset: Optional[int] = None


class ProcessStats(BaseModel):
    process_id: int
    process_name: str
    start_time: float
    read_ops: int = 0
    write_ops: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    unique_files: set = set()
    file_operations: List[IOEvent] = []


class DetectionResult(BaseModel):
    timestamp: float
    process_id: int
    process_name: str
    prediction_primary: float
    prediction_hybrid: float
    risk_level: str
    confidence: float
    features: Dict[str, float]
    alert: bool


class SystemStatus(BaseModel):
    status: str
    monitored_processes: int
    alerts_count: int
    uptime: float
    last_detection: Optional[str]


class TraceEvent(BaseModel):
    """One file I/O operation reported by monitor/ or simulation/."""
    timestamp: float
    operation_type: str  # 'read', 'write', 'create', 'delete'
    file_path: str
    offset: int = 0
    size: int = 0


class TraceBatch(BaseModel):
    """I/O operations of one process, sent to POST /api/predict/trace."""
    process_id: int
    process_name: str
    events: List[TraceEvent]
