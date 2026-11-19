#!/usr/bin/env python3
"""
Master's Defense Demo Script
Automated demonstration of the ransomware detection system for the thesis defense.

1. System initialization and health checks
2. Baseline monitoring demonstration
3. Progressive (safe) ransomware simulations
4. Real-time detection showcase
5. Live statistics from the running server

Run from the repository root:  python -m simulation.defense_demo
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):  # allow `python simulation/defense_demo.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.api_client import DEFAULT_SERVER_URL, DetectionClient
from simulation.simulation_system import RansomwareSimulator

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"


class DefenseDemo:
    """Orchestrates the complete defense demonstration"""

    def __init__(self, server_url=DEFAULT_SERVER_URL):
        self.server_process = None
        self.monitor_process = None
        self.server_url = server_url
        self.client = DetectionClient(server_url)
        self.demo_running = False
        self.start_time = None

    def print_banner(self):
        """Display demo banner"""
        print("\n" + "="*80)
        print("🎓 RANSOMWARE DETECTION SYSTEM - MASTER'S DEFENSE DEMONSTRATION")
        print("   Hybrid Machine Learning Approach for Real-time Threat Detection")
        print("="*80)

    def check_prerequisites(self):
        """Check if all required files are present"""
        print("\n🔍 Checking Prerequisites...")

        required_files = [
            "app/main.py",
            "app/static/dashboard.html",
            "simulation/simulation_system.py",
            "monitor/windows_monitor.py",
        ]
        missing_files = [f for f in required_files if not (REPO_ROOT / f).exists()]
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False

        models_dir = REPO_ROOT / "models"
        model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.pkl"))
        if len(model_files) < 4:
            print("❌ Expected 4 model files (.h5 + 3 .pkl) in models/")
            return False

        print("✅ All prerequisites satisfied")
        print(f"   Found {len(model_files)} model files")
        return True

    def start_detection_server(self):
        """Start the FastAPI detection server (or reuse a running one)"""
        print("\n🚀 Starting Detection Server...")

        if self.client.is_ready():
            print("✅ Detection server already running")
            return True

        try:
            LOG_DIR.mkdir(exist_ok=True)
            port = self.server_url.rsplit(":", 1)[-1].strip("/")
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", port],
                cwd=REPO_ROOT,
                stdout=open(LOG_DIR / "server.log", "w"),
                stderr=subprocess.STDOUT
            )

            # Wait for server to start and load the models
            for attempt in range(60):
                if self.client.is_ready():
                    print("✅ Detection server started successfully")
                    return True
                time.sleep(1)

            print(f"❌ Failed to start detection server (see {LOG_DIR / 'server.log'})")
            return False

        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False

    def start_file_monitor(self):
        """Start the file system monitor"""
        print("\n👁️ Starting File System Monitor...")

        try:
            LOG_DIR.mkdir(exist_ok=True)
            self.monitor_process = subprocess.Popen(
                [sys.executable, "-m", "monitor.windows_monitor", self.server_url],
                cwd=REPO_ROOT,
                stdout=open(LOG_DIR / "monitor.log", "w"),
                stderr=subprocess.STDOUT
            )

            time.sleep(3)  # Give monitor time to initialize
            print("✅ File system monitor started")
            return True

        except Exception as e:
            print(f"❌ Error starting monitor: {e}")
            return False

    def _alerts(self, limit=100):
        return [d for d in self.client.detections(limit) if d.get('alert')]

    def demonstrate_baseline(self):
        """Demonstrate normal system operation"""
        print("\n📊 DEMONSTRATION PHASE 1: Baseline System Monitoring")
        print("-" * 60)

        status = self.client.status()
        if not status:
            print("❌ Error getting status")
            return False

        system, models = status['system'], status['models']
        print("Initial System State:")
        print(f"  • Status: {system['status']}")
        print(f"  • Models Loaded: {'yes' if models else 'no'}")
        print(f"  • Input shape: {models.get('input_shape')}")
        print(f"  • Alerts so far: {system['alerts_count']}")

        print("\n⏳ Monitoring normal file operations for 30 seconds...")
        print("   (This demonstrates the system's baseline behavior)")

        alerts_before = len(self._alerts())
        start_time = time.time()
        while time.time() - start_time < 30:
            time.sleep(5)
            new_alerts = len(self._alerts()) - alerts_before
            if new_alerts > 0:
                print(f"   ⚠️ {new_alerts} alert(s) during baseline")
            else:
                print("   📊 Baseline monitoring... (no alerts)")

        print("✅ Baseline demonstration completed")
        return True

    def demonstrate_crypto_ransomware(self):
        """Demonstrate CryptoLocker-style ransomware detection"""
        print("\n🔴 DEMONSTRATION PHASE 2: CryptoLocker Ransomware Simulation")
        print("-" * 60)
        print("Simulating CryptoLocker behavior (inside simulation/simulation_files/ only):")
        print("  • Sequential file 'encryption' traces")
        print("  • Rapid read-write I/O patterns")
        print("  • File extension changes")
        print("\n🚨 STARTING CRYPTO SIMULATION...")

        try:
            simulator = RansomwareSimulator(self.server_url)
            simulator.create_test_files(20)

            alerts_before = len(self._alerts())
            simulator.simulate_crypto_ransomware(45)

            alerts_detected = 0
            detection_time = None
            start_sim = time.time()

            while time.time() - start_sim < 50:
                alerts = self._alerts()
                current = len(alerts) - alerts_before
                if current > alerts_detected:
                    if detection_time is None:
                        detection_time = time.time() - start_sim
                        print(f"\n🎯 FIRST DETECTION at {detection_time:.1f} seconds!")
                    alerts_detected = current
                    latest = alerts[-1]
                    print(f"   🚨 ALERT: {latest['process_name']} - Risk: {latest['risk_level']}")
                    print(f"   🔢 Score: {latest['prediction_hybrid']:.4f}")
                time.sleep(2)

            simulator.trace_buffer.stop()
            print(f"\n✅ CryptoLocker simulation completed")
            print(f"   Total alerts: {alerts_detected}")
            if detection_time:
                print(f"   Detection time: {detection_time:.1f} seconds")
            return True

        except Exception as e:
            print(f"❌ Error in crypto simulation: {e}")
            return False

    def demonstrate_advanced_detection(self):
        """Demonstrate multiple simultaneous threats"""
        print("\n🧠 DEMONSTRATION PHASE 3: Advanced Detection Features")
        print("-" * 60)

        try:
            simulator = RansomwareSimulator(self.server_url)
            simulator.create_test_files(20)

            print("Testing multiple ransomware types simultaneously...")
            simulator.simulate_crypto_ransomware(30)
            time.sleep(5)
            simulator.simulate_locker_ransomware(25)
            time.sleep(5)
            simulator.simulate_wiper_ransomware(20)

            print("\n📊 Multi-threat detection in progress...")

            start_time = time.time()
            max_score = 0
            threat_types = set()

            while time.time() - start_time < 35:
                for alert in self._alerts()[-10:]:
                    process = alert['process_name'].lower()
                    if 'crypto' in process:
                        threat_types.add('CryptoLocker')
                    elif 'locker' in process:
                        threat_types.add('Screen Locker')
                    elif 'wiper' in process:
                        threat_types.add('Data Wiper')
                    max_score = max(max_score, alert['prediction_hybrid'])

                time.sleep(3)
                print(f"   🔍 Monitoring... Max Score: {max_score:.4f}, Types: {len(threat_types)}")

            simulator.trace_buffer.stop()
            print(f"\n✅ Advanced detection completed")
            print(f"   Threat types detected: {', '.join(sorted(threat_types)) or 'none'}")
            print(f"   Highest hybrid score: {max_score:.4f}")
            return True

        except Exception as e:
            print(f"❌ Error in advanced detection: {e}")
            return False

    def generate_performance_report(self):
        """Print live statistics measured by the running server"""
        print("\n📈 DEMONSTRATION PHASE 4: Live Statistics")
        print("-" * 60)

        status = self.client.status()
        if not status:
            print("❌ Error getting status")
            return False

        detections = self.client.detections(100)
        alerts = [d for d in detections if d.get('alert')]
        system = status['system']

        print("LIVE SERVER STATISTICS (last 100 detections):")
        print(f"  📊 Detections: {len(detections)}")
        print(f"  🚨 Alerts: {len(alerts)}  (total since start: {system['alerts_count']})")
        print(f"  🔴 Critical Alerts: {sum(1 for a in alerts if a.get('risk_level') == 'CRITICAL')}")
        print(f"  🟠 High Risk Alerts: {sum(1 for a in alerts if a.get('risk_level') == 'HIGH')}")
        print(f"  ⏱️ System Uptime: {system['uptime']:.1f} seconds")
        print("\nOffline test-set results of the model are reported in README.md (Results).")
        return True

    def run_complete_demo(self):
        """Run the complete defense demonstration"""
        self.print_banner()

        if not self.check_prerequisites():
            return False
        if not self.start_detection_server():
            return False
        if not self.start_file_monitor():
            return False

        print(f"\n🌐 Dashboard available at: {self.server_url}/dashboard")
        print("   Open this URL in your browser for real-time visualization")

        input("\nPress Enter when ready to begin demonstration...")

        self.start_time = time.time()
        self.demo_running = True

        try:
            if not self.demonstrate_baseline():
                return False

            input("\nPress Enter to continue to ransomware simulation...")
            if not self.demonstrate_crypto_ransomware():
                return False

            input("\nPress Enter to continue to advanced detection...")
            if not self.demonstrate_advanced_detection():
                return False

            self.generate_performance_report()

            demo_duration = time.time() - self.start_time
            print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print(f"   Total Duration: {demo_duration:.1f} seconds")
            print(f"   Dashboard: {self.server_url}/dashboard")

            input("\nPress Enter to end demonstration and cleanup...")

        except KeyboardInterrupt:
            print("\n🛑 Demonstration interrupted by user")

        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """Clean up processes and resources"""
        print("\n🧹 Cleaning up...")

        if self.server_process:
            self.server_process.terminate()
            print("  ✅ Detection server stopped")

        if self.monitor_process:
            self.monitor_process.terminate()
            print("  ✅ File monitor stopped")

        print("  ✅ Cleanup completed")

    def run_interactive_mode(self):
        """Run in interactive mode for manual control"""
        self.print_banner()
        print("\n🎮 INTERACTIVE DEMO MODE")
        print("   Manual control for flexible demonstration")

        if not self.check_prerequisites():
            return False
        if not self.start_detection_server():
            return False

        print(f"\n🌐 Dashboard: {self.server_url}/dashboard")
        print("✅ Server is ready for demonstration")

        while True:
            print("\n" + "-"*50)
            print("DEMO CONTROL MENU:")
            print("1. Check System Status")
            print("2. Start File System Monitor")
            print("3. Run Baseline Demo (30s)")
            print("4. Run Crypto Simulation (45s)")
            print("5. Run Multi-Threat Demo (35s)")
            print("6. Show Live Statistics")
            print("7. Open Dashboard Instructions")
            print("0. Exit and Cleanup")
            print("-"*50)

            try:
                choice = input("Select option (0-7): ").strip()

                if choice == "1":
                    self._show_system_status()
                elif choice == "2":
                    if not self.start_file_monitor():
                        print("❌ Failed to start file monitor")
                elif choice == "3":
                    self.demonstrate_baseline()
                elif choice == "4":
                    self.demonstrate_crypto_ransomware()
                elif choice == "5":
                    self.demonstrate_advanced_detection()
                elif choice == "6":
                    self.generate_performance_report()
                elif choice == "7":
                    self._show_dashboard_instructions()
                elif choice == "0":
                    print("👋 Exiting demonstration...")
                    break
                else:
                    print("❌ Invalid option. Please select 0-7.")

            except (KeyboardInterrupt, EOFError):
                print("\n🛑 Interrupted by user")
                break

        self.cleanup()
        return True

    def _show_system_status(self):
        """Show current system status"""
        status = self.client.status()
        if not status:
            print("❌ Failed to get system status")
            return
        system, models = status['system'], status['models']
        print(f"\n📊 SYSTEM STATUS:")
        print(f"   Status: {system['status']}")
        print(f"   Uptime: {system['uptime']:.1f} seconds")
        print(f"   Alerts Triggered: {system['alerts_count']}")
        print(f"   Monitored Processes: {system['monitored_processes']}")
        print(f"   Models: {'✅' if models else '❌'}")
        if models:
            print(f"   Loaded at: {models.get('loaded_at')}")

    def _show_dashboard_instructions(self):
        """Show dashboard instructions"""
        print(f"\n🌐 DASHBOARD INSTRUCTIONS:")
        print(f"   1. Open your web browser")
        print(f"   2. Navigate to: {self.server_url}/dashboard")
        print(f"   3. The dashboard shows:")
        print(f"      • Live system status")
        print(f"      • Real-time detection feed")
        print(f"      • Alert notifications")
        print(f"   4. API documentation: {self.server_url}/docs")


def main():
    """Main entry point"""
    server_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER_URL
    demo = DefenseDemo(server_url)

    print("Ransomware Detection System - Master's Defense Demo")
    print("Choose demonstration mode:")
    print("1. Automated Demo (recommended for defense)")
    print("2. Interactive Mode (manual control)")

    try:
        choice = input("\nSelect mode (1 or 2): ").strip()

        if choice == "1":
            print("\n🤖 Starting Automated Demo Mode...")
            if input("Continue? (y/N): ").lower().strip() == 'y':
                return 0 if demo.run_complete_demo() else 1
            print("Demo cancelled")
            return 0
        elif choice == "2":
            print("\n🎮 Starting Interactive Mode...")
            return 0 if demo.run_interactive_mode() else 1
        else:
            print("❌ Invalid selection. Please choose 1 or 2.")
            return 1

    except (KeyboardInterrupt, EOFError):
        print("\n👋 Demo cancelled by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
