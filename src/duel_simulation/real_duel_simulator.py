import os
import subprocess
import time
import socket
import json
import random
from pathlib import Path

class WindbotDockerSimulator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.ocgcore_lib_path = self.project_root / "src" / "duel_simulation" / "edopro" / "ocgcore" / "bin" / "release" / "libocgcore.so"
        self.docker_image = "windbot-server"
        self.port = 2399

    def is_docker_running(self):
        try:
            subprocess.check_output(["docker", "info"], stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

    def start_edopro_container(self):
        """
        Start the EDOPro server container in host mode.
        Assumes the EDOPro image is called 'edopro-server'.
        """
        print("\n🚀 Starting EDOPro server container...")
        try:
            subprocess.run([
                "docker", "run", "-d", "--rm",
                "--name", "edopro-server",
                "-p", "2399:2399",  # expose server port
                "edopro-server"     # replace with your actual image name if different
            ], check=True)
            time.sleep(5)
            print("✓ EDOPro server container is running.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start EDOPro container: {e}")


    def check_edopro_listening(self):
        """
        Check if the EDOPro server is listening on port 2399.
        """
        print("🔍 Checking if EDOPro server is listening on port 2399...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            result = sock.connect_ex(('127.0.0.1', 2399))
            if result == 0:
                print("✓ EDOPro server is up and accepting connections!")
                return True
            print("❌ EDOPro server not reachable at localhost:2399")
            return False

    def start_docker_container(self):
        print(f"\n🚀 Starting Windbot server container on port {self.port}...")
        try:
            subprocess.run([
                "docker", "run", "-d", "--rm",
                "-p", f"{self.port}:2399",
                "--name", "windbot-server",
                self.docker_image
            ], check=True)
            time.sleep(5)  # Wait for server to start
            print("✓ Windbot server container is running.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start Windbot container: {e}")

    def check_server_listening(self):
        print("🔍 Checking if Windbot server is listening...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            result = sock.connect_ex(('127.0.0.1', self.port))
            if result == 0:
                print("✓ Windbot server is up and accepting connections!")
                return True
            print("❌ Server not reachable at localhost:2399")
            return False

    def run_duel_simulation(self, deck1_path, deck2_path):
        print("\n🎮 Running duel simulation with EDOPro client...")
        print(f"   Deck 1: {deck1_path}")
        print(f"   Deck 2: {deck2_path}")

        edopro_exe = self.project_root / "src" / "duel_simulation" / "edopro" / "build" / "ygopro"
        if not edopro_exe.exists():
            print(f"❌ EDOPro binary not found at {edopro_exe}")
            return {
                "status": "edopro_not_found",
                "connected": self.check_server_listening()
            }

        log_path = self.project_root / "duel_logs" / f"log_{int(time.time())}.txt"
        os.makedirs(log_path.parent, exist_ok=True)

        # Start the match: WindBot listens, EDOPro connects
        cmd = [
            str(edopro_exe),
            "-j", f"{deck1_path}|{deck2_path}",
            "-s", "127.0.0.1",  # Connect to WindBot server
            "-w",  # No GUI
        ]

        try:
            print(f"🚀 Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

            # Log stdout for debugging
            with open(log_path, "wb") as f:
                f.write(result.stdout)

            if result.returncode == 0:
                print("✓ Duel completed successfully.")
            else:
                print("⚠️ Duel process exited with error.")

            return {
                "deck1": str(deck1_path),
                "deck2": str(deck2_path),
                "status": "completed" if result.returncode == 0 else "error",
                "connected": self.check_server_listening(),
                "log": str(log_path)
            }

        except subprocess.TimeoutExpired:
            print("❌ Duel process timed out.")
            return {
                "deck1": str(deck1_path),
                "deck2": str(deck2_path),
                "status": "timeout",
                "connected": self.check_server_listening()
            }


def main():
    sim = WindbotDockerSimulator()

    if not sim.is_docker_running():
        print("❌ Docker does not appear to be running. Please start Docker Desktop.")
        return

    #sim.start_docker_container()

    # Get 2 deck files for testing
    deck_dir = Path(__file__).parent.parent.parent / "decks" / "known"
    deck_files = list(deck_dir.glob("*.ydk"))

    if len(deck_files) < 2:
        print("❌ Need at least two .ydk deck files in decks/known/")
        return

    result = sim.run_duel_simulation(str(deck_files[0]), str(deck_files[1]))
    print("\n📊 Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

