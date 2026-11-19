#!/usr/bin/env python3
"""
Ransomware Detection Simulation System (safe)

Generates ransomware-like file I/O traces and sends them to the FastAPI
backend (POST /api/predict/trace), which scores them with the hybrid model.

SAFETY: the only files this script creates, renames or deletes live in
``simulation/simulation_files/``. It never encrypts, overwrites or deletes
user files. "Locker" and "wiper" behaviours only *report* virtual trace
events; no other file is opened.

Run from the repository root:  python -m simulation.simulation_system [server_url]
"""

import logging
import os
import random
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ''):  # allow `python simulation/simulation_system.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.api_client import DEFAULT_SERVER_URL, DetectionClient, TraceBuffer

SIMULATION_DIR = (Path(__file__).resolve().parent / "simulation_files")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ransomware_simulation")


def _inside_sandbox(path: Path) -> Path:
    """Refuse any path outside SIMULATION_DIR."""
    resolved = Path(path).resolve()
    if resolved != SIMULATION_DIR and SIMULATION_DIR not in resolved.parents:
        raise PermissionError(f"Refusing to touch {resolved}: outside {SIMULATION_DIR}")
    return resolved


def print_alert(result: Dict):
    """Pretty-print a detection result returned by the server when it is an alert."""
    if not result.get('alert', False):
        return
    logger.warning(f"🚨 RANSOMWARE DETECTED: {result.get('process_name')} - Risk: {result.get('risk_level')}")
    print(f"\n{'='*60}")
    print(f"🚨 RANSOMWARE ALERT TRIGGERED!")
    print(f"Process: {result.get('process_name')} (PID: {result.get('process_id')})")
    print(f"Risk Level: {result.get('risk_level', 'UNKNOWN')}")
    print(f"Hybrid score: {result.get('prediction_hybrid', 0):.4f}  "
          f"(CNN-LSTM: {result.get('prediction_primary', 0):.4f})")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


class RansomwareSimulator:
    """Simulate various ransomware behaviors for testing"""

    def __init__(self, detection_server_url=DEFAULT_SERVER_URL, on_result=print_alert):
        self.server_url = detection_server_url
        self.client = DetectionClient(detection_server_url)
        self.trace_buffer = TraceBuffer(self.client, flush_interval=1.0, on_result=on_result)
        self.trace_buffer.start()
        self.simulation_dir = SIMULATION_DIR
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        self.running_simulations = []

    def create_test_files(self, count=100):
        """Create test files for simulation (inside simulation_files/ only)"""
        logger.info(f"Creating {count} test files...")
        for old in self.simulation_dir.glob("test_file_*"):
            _inside_sandbox(old).unlink()
        for i in range(count):
            file_path = _inside_sandbox(self.simulation_dir / f"test_file_{i:03d}.txt")
            with open(file_path, 'w') as f:
                # Create files with varying content sizes
                content_size = random.randint(100, 5000)
                content = "A" * content_size
                f.write(content)
        logger.info(f"Created {count} test files in {self.simulation_dir}")

    def simulate_crypto_ransomware(self, duration=60):
        """Simulate CryptoLocker-style behaviour: read + write every block, then rename.

        The rename (``.txt`` -> ``.encrypted``) happens inside simulation_files/;
        file contents are never modified.
        """
        logger.info("Starting CryptoLocker simulation...")

        def crypto_behavior():
            process_name = "crypto_sim.exe"
            pid = random.randint(2000, 9999)

            start_time = time.time()
            file_count = 0

            while time.time() - start_time < duration:
                try:
                    files = list(self.simulation_dir.glob("*.txt"))
                    if not files:
                        break

                    target_file = _inside_sandbox(random.choice(files))

                    traces = self._generate_crypto_traces(target_file, process_name, pid)

                    for trace in traces:
                        self._send_trace(trace)
                        time.sleep(random.uniform(0.01, 0.05))  # Rapid I/O

                    file_count += 1

                    encrypted_path = _inside_sandbox(target_file.with_suffix('.encrypted'))
                    if target_file.exists():
                        target_file.rename(encrypted_path)

                    if file_count % 10 == 0:
                        logger.info(f"Crypto simulation: {file_count} files processed")

                except Exception as e:
                    logger.error(f"Crypto simulation error: {e}")

            logger.info(f"CryptoLocker simulation completed: {file_count} files processed")

        thread = threading.Thread(target=crypto_behavior, daemon=True)
        thread.start()
        self.running_simulations.append(thread)
        return thread

    def simulate_locker_ransomware(self, duration=30):
        """Simulate screen-locker behaviour: repeated reads of system files (trace events only)"""
        logger.info("Starting Locker simulation...")

        def locker_behavior():
            process_name = "locker_sim.exe"
            pid = random.randint(3000, 9999)

            start_time = time.time()

            # Virtual paths: reported to the server, never opened
            system_files = [
                "C:\\Windows\\System32\\kernel32.dll",
                "C:\\Windows\\System32\\user32.dll",
                "C:\\Windows\\System32\\ntdll.dll",
                "C:\\Windows\\explorer.exe"
            ]

            while time.time() - start_time < duration:
                try:
                    for sys_file in system_files:
                        self._send_trace({
                            "timestamp": time.time(),
                            "operation_type": "read",
                            "file_path": sys_file,
                            "offset": random.randint(0, 100000),
                            "size": random.randint(1000, 10000),
                            "process_id": pid,
                            "process_name": process_name
                        })

                    time.sleep(random.uniform(0.5, 2.0))

                except Exception as e:
                    logger.error(f"Locker simulation error: {e}")

            logger.info("Locker simulation completed")

        thread = threading.Thread(target=locker_behavior, daemon=True)
        thread.start()
        self.running_simulations.append(thread)
        return thread

    def simulate_wiper_ransomware(self, duration=45):
        """Simulate data-wiper behaviour: bursts of large overwrites (trace events only)"""
        logger.info("Starting Wiper simulation...")

        def wiper_behavior():
            process_name = "wiper_sim.exe"
            pid = random.randint(4000, 9999)

            start_time = time.time()
            files_wiped = 0

            while time.time() - start_time < duration:
                try:
                    # Simulate aggressive file overwriting (virtual paths, nothing is written)
                    for _ in range(random.randint(3, 8)):
                        self._send_trace({
                            "timestamp": time.time(),
                            "operation_type": "write",
                            "file_path": str(self.simulation_dir / f"virtual_important_{random.randint(1, 100)}.doc"),
                            "offset": 0,
                            "size": random.randint(50000, 500000),  # Large writes
                            "process_id": pid,
                            "process_name": process_name
                        })
                        files_wiped += 1

                    time.sleep(random.uniform(0.1, 0.3))  # Aggressive timing

                except Exception as e:
                    logger.error(f"Wiper simulation error: {e}")

            logger.info(f"Wiper simulation completed: {files_wiped} operations")

        thread = threading.Thread(target=wiper_behavior, daemon=True)
        thread.start()
        self.running_simulations.append(thread)
        return thread

    def _generate_crypto_traces(self, file_path: Path, process_name: str, pid: int) -> List[Dict]:
        """Generate realistic crypto-ransomware I/O traces"""
        traces = []
        file_size = file_path.stat().st_size if file_path.exists() else 1000

        # Read original file
        for offset in range(0, file_size, 4096):
            traces.append({
                "timestamp": time.time(),
                "operation_type": "read",
                "file_path": str(file_path),
                "offset": offset,
                "size": min(4096, file_size - offset),
                "process_id": pid,
                "process_name": process_name
            })

        # Write encrypted data back (reported only)
        for offset in range(0, file_size, 4096):
            traces.append({
                "timestamp": time.time(),
                "operation_type": "write",
                "file_path": str(file_path),
                "offset": offset,
                "size": min(4096, file_size - offset),
                "process_id": pid,
                "process_name": process_name
            })

        return traces

    def _send_trace(self, trace: Dict):
        """Queue trace for the detection platform (sent in batches)"""
        self.trace_buffer.add(trace)

    def wait(self):
        for t in self.running_simulations:
            t.join()
        self.trace_buffer.flush()

    def cleanup_files(self):
        """Delete simulation_files/ (the only folder this script writes to)"""
        if self.simulation_dir.exists():
            shutil.rmtree(_inside_sandbox(self.simulation_dir))
            logger.info(f"Removed {self.simulation_dir}")


class SimulationController:
    """Main controller for the simulation system"""

    def __init__(self, detection_server_url=DEFAULT_SERVER_URL):
        self.server_url = detection_server_url
        self.client = DetectionClient(detection_server_url)
        self.simulator = RansomwareSimulator(detection_server_url)
        self.running = False

    def start_simulation(self):
        """Start the complete simulation system"""
        print("\n" + "="*80)
        print("🛡️  RANSOMWARE DETECTION SIMULATION SYSTEM")
        print("    Master's Degree Defense - Live Demonstration")
        print("="*80)

        if not self.client.is_ready():
            print("❌ Detection server is not running or models are not loaded!")
            print("   Start it first: uvicorn app.main:app --host 127.0.0.1 --port 8000")
            return False

        print("✅ Detection server is running")

        self.simulator.create_test_files(50)
        self.running = True
        self._show_menu()
        return True

    def _show_menu(self):
        """Show interactive menu"""
        while self.running:
            print("\n" + "-"*50)
            print("SIMULATION CONTROL MENU:")
            print("1. Start CryptoLocker Simulation (60s)")
            print("2. Start Screen Locker Simulation (30s)")
            print("3. Start Data Wiper Simulation (45s)")
            print("4. Start All Simulations")
            print("5. Run a single prediction (/api/predict)")
            print("6. Check System Status")
            print("7. View Recent Alerts")
            print("8. Recreate test files")
            print("0. Exit")
            print("-"*50)

            try:
                choice = input("Select option (0-8): ").strip()

                if choice == "1":
                    self.simulator.simulate_crypto_ransomware(60)
                    print("🔄 CryptoLocker simulation started (60 seconds)")

                elif choice == "2":
                    self.simulator.simulate_locker_ransomware(30)
                    print("🔄 Screen Locker simulation started (30 seconds)")

                elif choice == "3":
                    self.simulator.simulate_wiper_ransomware(45)
                    print("🔄 Data Wiper simulation started (45 seconds)")

                elif choice == "4":
                    self.simulator.simulate_crypto_ransomware(60)
                    time.sleep(2)
                    self.simulator.simulate_locker_ransomware(30)
                    time.sleep(2)
                    self.simulator.simulate_wiper_ransomware(45)
                    print("🔄 All ransomware simulations started!")

                elif choice == "5":
                    self._single_prediction()

                elif choice == "6":
                    self._show_system_status()

                elif choice == "7":
                    self._show_recent_alerts()

                elif choice == "8":
                    self.simulator.create_test_files(50)

                elif choice == "0":
                    print("👋 Exiting simulation system...")
                    self.running = False

                else:
                    print("❌ Invalid option. Please select 0-8.")

            except (KeyboardInterrupt, EOFError):
                print("\n🛑 Interrupted by user")
                self.running = False

        self.cleanup()

    def _single_prediction(self):
        """Send one write-heavy feature vector to /api/predict and show the model output"""
        features = {
            'read_write_ratio': 1.0, 'war_ratio': 0.5, 'wss': 50.0, 'entropy': 5.0,
            'read_pct': 0.5, 'write_pct': 0.5, 'repeat_ratio': 1.0,
            'read_entropy': 4.5, 'write_entropy': 4.5, 'total_ops': 400.0,
            'write_to_unique_ratio': 4.0, 'avg_offset_gap': 4096.0, 'burstiness': 1.5
        }
        result = self.client.predict_features(features)
        if result is None:
            print("❌ Prediction failed")
            return
        print(f"   CNN-LSTM: {result['prediction_primary']:.4f}  Hybrid: {result['prediction_hybrid']:.4f}  "
              f"Risk: {result['risk_level']}  Alert: {result['alert']}")

    def _show_system_status(self):
        """Show current system status"""
        status = self.client.status()
        if not status:
            print("❌ Failed to get system status")
            return
        system = status['system']
        print(f"\n📊 SYSTEM STATUS:")
        print(f"   Status: {system['status']}")
        print(f"   Uptime: {system['uptime']:.1f} seconds")
        print(f"   Alerts Triggered: {system['alerts_count']}")
        print(f"   Monitored Processes: {system['monitored_processes']}")
        print(f"   Recent detections: {len(self.client.detections(100))}")

    def _show_recent_alerts(self):
        """Show recent alerts"""
        alerts = [d for d in self.client.detections(100) if d.get('alert')]
        print(f"\n🚨 RECENT ALERTS ({len(alerts)} in last 100 detections):")
        for alert in alerts[-5:]:  # Show last 5
            print(f"   • {alert['process_name']} (PID: {alert['process_id']})")
            print(f"     Risk: {alert['risk_level']}, Score: {alert['prediction_hybrid']:.3f}")
            print(f"     Time: {datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')}")
            print()

    def cleanup(self):
        """Clean up resources"""
        self.simulator.trace_buffer.stop()
        print("✅ Simulation system stopped")


def main():
    """Main entry point"""
    print("Initializing Ransomware Detection Simulation System...")

    server_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER_URL
    controller = SimulationController(server_url)

    try:
        if not controller.start_simulation():
            print("\n❌ Failed to start simulation system")
            return 1
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        controller.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
