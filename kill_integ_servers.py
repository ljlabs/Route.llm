"""
Kill any processes listening on integration test ports.
Run this if tests were interrupted and servers are still running.
"""
import subprocess
import sys

PORTS = [8081]

def kill_port(port):
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            pid = line.strip().split()[-1]
            try:
                subprocess.run(["taskkill", "/PID", pid, "/F"], check=True)
                print(f"Killed PID {pid} on port {port}")
            except subprocess.CalledProcessError:
                print(f"Failed to kill PID {pid} on port {port}")
            return
    print(f"Nothing listening on port {port}")

if __name__ == "__main__":
    for port in PORTS:
        kill_port(port)
    print("Done.")
