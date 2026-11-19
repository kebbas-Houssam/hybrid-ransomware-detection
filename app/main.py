"""FastAPI backend: real-time ransomware detection with the PSO-optimized hybrid model.

Run from the repository root:

    uvicorn app.main:app --host 127.0.0.1 --port 8000
    # or: python -m app.main --auto-start

Dashboard: /dashboard · API docs: /docs · WebSocket: /ws
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Optional

if __package__ in (None, ''):  # allow `python app/main.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import joblib
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR, config
from app.detector import RansomwareDetector, assess_risk, features_to_vector
from app.model_manager import ModelManager, default_model_files
from app.schemas import TraceBatch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(title="Ransomware Detection System", version="1.0.0",
              description="PSO-optimized CNN-LSTM-Attention + XGBoost ransomware detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = RansomwareDetector()
connected_websockets = set()
AUTO_START = os.environ.get("RSW_AUTO_START") == "1"


# API Routes
@app.on_event("startup")
async def startup_event():
    """Load the models from models/ and optionally start detection."""
    logger.info("Starting Ransomware Detection API...")
    try:
        default_model_files()
    except FileNotFoundError as e:
        logger.warning(f"Model files not found: {e}")
        return

    if detector.load_models():
        logger.info("Models loaded successfully on startup")
        if AUTO_START:
            detector.start_detection()
    else:
        logger.error("Failed to load models on startup")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    detector.stop_detection()
    logger.info("Detection system stopped")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Ransomware Detection System API", "version": "1.0.0",
            "dashboard": "/dashboard", "docs": "/docs"}


@app.get("/api/status")
async def get_status():
    """Get system status"""
    status = detector.get_system_status()
    model_info = detector.model_manager.model_info if detector.model_manager.model_loaded else {}

    return {
        "system": status.model_dump(),
        "models": model_info,
        "websockets": len(connected_websockets)
    }


@app.post("/api/load-models")
async def load_models(
    h5_path: Optional[str] = None,
    xgb_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    xgb_scaler_path: Optional[str] = None
):
    """Load detection models. Paths default to the artifacts in models/."""
    try:
        logger.info(f"API: Load models request - H5: {h5_path}, XGB: {xgb_path}")

        for label, path in [("H5 model", h5_path), ("XGBoost model", xgb_path),
                            ("Scaler", scaler_path), ("XGBoost scaler", xgb_scaler_path)]:
            if path and not os.path.exists(path):
                raise HTTPException(status_code=404, detail=f"{label} file not found: {path}")

        success = detector.load_models(h5_path, xgb_path, scaler_path, xgb_scaler_path)

        if success:
            await broadcast_to_websockets({
                "type": "model_status",
                "data": {"status": "loaded", "timestamp": time.time()}
            })
            return {
                "status": "success",
                "message": "Models loaded successfully",
                "model_info": detector.model_manager.model_info
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load models")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-detection")
async def start_detection():
    """Start the detection system"""
    try:
        if detector.detecting:
            return {"status": "already_running", "message": "Detection already running"}

        success = detector.start_detection()

        if success:
            await broadcast_to_websockets({
                "type": "system_status",
                "data": {"status": "started", "timestamp": time.time()}
            })
            return {"status": "success", "message": "Detection started"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start detection")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop-detection")
async def stop_detection():
    """Stop the detection system"""
    try:
        detector.stop_detection()

        await broadcast_to_websockets({
            "type": "system_status",
            "data": {"status": "stopped", "timestamp": time.time()}
        })

        return {"status": "success", "message": "Detection stopped"}

    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detections")
async def get_detections(limit: int = 50):
    """Get recent detection results"""
    try:
        detections = detector.get_recent_detections(limit)
        return {"detections": [d.model_dump() for d in detections]}

    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/processes")
async def get_processes():
    """Get monitored processes"""
    try:
        processes = []
        for pid, stats in list(detector.io_monitor.process_stats.items()):
            features = detector.io_monitor.get_process_features(pid)
            processes.append({
                "pid": pid,
                "name": stats.process_name,
                "start_time": stats.start_time,
                "read_ops": stats.read_ops,
                "write_ops": stats.write_ops,
                "total_bytes_read": stats.total_bytes_read,
                "total_bytes_written": stats.total_bytes_written,
                "unique_files": len(stats.unique_files),
                "features": features or {}
            })

        return {"processes": processes}

    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def manual_prediction(features: Dict[str, float]):
    """Predict from one set of the 13 window features (repeated over 50 time steps)."""
    try:
        if not detector.model_manager.model_loaded:
            raise HTTPException(status_code=400, detail="Models not loaded")

        feature_vector = features_to_vector(features)
        sequence = np.tile(feature_vector, (config.SEQUENCE_LENGTH, 1))
        pred_primary, pred_hybrid = detector.model_manager.predict(sequence)
        risk_level, alert = assess_risk(pred_hybrid)

        return {
            "prediction_primary": pred_primary,
            "prediction_hybrid": pred_hybrid,
            "risk_level": risk_level,
            "alert": alert,
            "confidence": max(pred_hybrid, 1 - pred_hybrid)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/trace")
async def trace_prediction(batch: TraceBatch):
    """Predict from raw file I/O operations of one process (used by monitor/ and simulation/).

    Features are computed over the last 5 s of the process' reported events and
    appended to its 50-window sequence; the result is recorded in /api/detections.
    """
    try:
        if not detector.model_manager.model_loaded:
            raise HTTPException(status_code=400, detail="Models not loaded")
        if not batch.events:
            raise HTTPException(status_code=422, detail="Batch contains no events")

        result = await asyncio.to_thread(detector.process_trace_batch, batch)
        if result.alert:
            await broadcast_to_websockets({"type": "alert", "data": result.model_dump()})

        payload = result.model_dump()
        payload["windows_observed"] = len(detector.trace_sequences.get(batch.process_id, []))
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trace prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _model_files_or_empty():
    try:
        return default_model_files()
    except FileNotFoundError:
        return {}


@app.get("/api/model-status")
async def get_model_status():
    """Get current model loading status"""
    try:
        files = _model_files_or_empty()
        model_files = {role: os.path.exists(path) for role, path in files.items()}
        return {
            "models_loaded": detector.model_manager.model_loaded,
            "model_info": detector.model_manager.model_info,
            "tensorflow_available": True,
            "xgboost_available": True,
            "model_directory": config.MODEL_DIR,
            "timestamp": time.time(),
            "model_files_present": model_files,
            "all_files_present": len(model_files) == 4 and all(model_files.values()),
        }

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/list-model-files")
async def list_model_files():
    """List available model files in models/"""
    try:
        files = {
            "h5_files": [],
            "pkl_files": [],
            "model_directory": config.MODEL_DIR
        }

        for item in sorted(os.listdir(config.MODEL_DIR)):
            path = os.path.join(config.MODEL_DIR, item)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                file_info = {
                    "name": item,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "path": path
                }

                if item.endswith('.h5'):
                    files["h5_files"].append(file_info)
                elif item.endswith('.pkl'):
                    files["pkl_files"].append(file_info)

        return files

    except Exception as e:
        logger.error(f"Error listing model files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug-load-test")
async def debug_load_test():
    """Debug endpoint to test model loading directly"""
    try:
        result = {
            "step1_files_exist": {},
            "step2_tensorflow_test": {},
            "step3_h5_load_test": {},
            "step4_xgb_load_test": {},
            "errors": []
        }

        files = _model_files_or_empty()
        h5_path = files.get('primary', '')
        xgb_path = files.get('xgboost', '')

        result["step1_files_exist"] = {
            "h5_exists": os.path.exists(h5_path),
            "h5_size": os.path.getsize(h5_path) if os.path.exists(h5_path) else 0,
            "xgb_exists": os.path.exists(xgb_path),
            "xgb_size": os.path.getsize(xgb_path) if os.path.exists(xgb_path) else 0
        }

        try:
            result["step2_tensorflow_test"] = {
                "version": tf.__version__,
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "can_create_model": True
            }
            tf.keras.Sequential([tf.keras.layers.Dense(1)])
        except Exception as e:
            result["step2_tensorflow_test"]["error"] = str(e)
            result["errors"].append(f"TensorFlow test failed: {e}")

        try:
            test_model = ModelManager().load_model_compatible(h5_path)
            result["step3_h5_load_test"] = {
                "loaded_with_compatibility": True,
                "input_shape": str(test_model.input_shape),
                "output_shape": str(test_model.output_shape)
            }
        except Exception as e:
            result["step3_h5_load_test"] = {"loaded_with_compatibility": False, "error": str(e)}
            result["errors"].append(f"H5 loading failed: {e}")

        try:
            xgb_model = joblib.load(xgb_path)
            result["step4_xgb_load_test"] = {"loaded": True, "type": str(type(xgb_model))}
        except Exception as e:
            result["step4_xgb_load_test"] = {"loaded": False, "error": str(e)}
            result["errors"].append(f"XGBoost loading failed: {e}")

        return result

    except Exception as e:
        return {"critical_error": str(e)}


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time detection updates"""
    await websocket.accept()
    connected_websockets.add(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(connected_websockets)}")

    try:
        status = detector.get_system_status()
        await websocket.send_json({
            "type": "system_status",
            "data": status.model_dump()
        })

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                if detector.detecting:
                    recent_detections = detector.get_recent_detections(5)
                    if recent_detections:
                        await websocket.send_json({
                            "type": "detections",
                            "data": [d.model_dump() for d in recent_detections[-5:]]
                        })

                    status = detector.get_system_status()
                    await websocket.send_json({
                        "type": "system_status",
                        "data": status.model_dump()
                    })

                await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_websockets.discard(websocket)
        logger.info(f"WebSocket removed. Total connections: {len(connected_websockets)}")


async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected websockets"""
    if not connected_websockets:
        return

    disconnected = set()
    for websocket in list(connected_websockets):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting to websocket: {e}")
            disconnected.add(websocket)

    connected_websockets.difference_update(disconnected)


# Static files (for frontend)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard"""
    html_path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Ransomware Detection Dashboard</title></head>
        <body>
            <h1>Dashboard not found</h1>
            <p>Frontend files not available. API is running at <a href="/docs">/docs</a></p>
            <p>Model status: <a href="/api/model-status">/api/model-status</a></p>
        </body>
        </html>
        """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ransomware Detection Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--model-dir", default=config.MODEL_DIR, help="Folder with the model artifacts")
    parser.add_argument("--auto-start", action="store_true", help="Auto start detection")
    args = parser.parse_args()

    config.MODEL_DIR = os.path.abspath(args.model_dir)
    AUTO_START = AUTO_START or args.auto_start

    print("=" * 80)
    print("RANSOMWARE DETECTION SYSTEM - BACKEND SERVER")
    print("=" * 80)
    print(f"Host: {args.host}:{args.port}")
    print(f"Models: {config.MODEL_DIR}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Auto-start: {AUTO_START}")
    print("=" * 80)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
