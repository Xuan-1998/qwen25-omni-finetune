#!/usr/bin/env python3
"""
Training Monitor for Qwen 2.5 Omni Fine-tuning
Monitors GPU usage, memory, and training progress
"""

import os
import time
import json
import psutil
import subprocess
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class TrainingMonitor:
    def __init__(self, log_dir="./logs", output_file="monitoring_report.json"):
        self.log_dir = log_dir
        self.output_file = output_file
        self.monitoring_data = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def get_gpu_info(self):
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info.append({
                            'name': parts[0],
                            'memory_total': int(parts[1]),
                            'memory_used': int(parts[2]),
                            'memory_free': int(parts[3]),
                            'utilization': int(parts[4]),
                            'temperature': int(parts[5])
                        })
                return gpu_info
        except Exception as e:
            print(f"Error getting GPU info: {e}")
        
        return []
    
    def get_system_info(self):
        """Get system information."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_total': memory.total,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'disk_total': disk.total,
                'disk_used': disk.used,
                'disk_percent': (disk.used / disk.total) * 100
            }
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {}
    
    def check_training_logs(self):
        """Check training logs for progress."""
        log_files = []
        if os.path.exists(self.log_dir):
            for file in os.listdir(self.log_dir):
                if file.endswith('.log') or 'events.out.tfevents' in file:
                    log_files.append(os.path.join(self.log_dir, file))
        
        return log_files
    
    def monitor_once(self):
        """Perform one monitoring cycle."""
        timestamp = datetime.now().isoformat()
        
        # Get GPU info
        gpu_info = self.get_gpu_info()
        
        # Get system info
        system_info = self.get_system_info()
        
        # Check log files
        log_files = self.check_training_logs()
        
        monitoring_entry = {
            'timestamp': timestamp,
            'gpu_info': gpu_info,
            'system_info': system_info,
            'log_files': log_files
        }
        
        self.monitoring_data.append(monitoring_entry)
        
        # Print current status
        print(f"\n📊 Monitoring Report - {timestamp}")
        print("=" * 50)
        
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                print(f"GPU {i}: {gpu['name']}")
                print(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
                print(f"  Utilization: {gpu['utilization']}%")
                print(f"  Temperature: {gpu['temperature']}°C")
        
        if system_info:
            print(f"CPU Usage: {system_info.get('cpu_percent', 0):.1f}%")
            print(f"Memory Usage: {system_info.get('memory_percent', 0):.1f}%")
            print(f"Disk Usage: {system_info.get('disk_percent', 0):.1f}%")
        
        print(f"Log Files: {len(log_files)}")
        
        return monitoring_entry
    
    def save_data(self):
        """Save monitoring data to file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.monitoring_data, f, indent=2)
        print(f"📁 Monitoring data saved to {self.output_file}")
    
    def generate_report(self):
        """Generate a monitoring report."""
        if not self.monitoring_data:
            print("No monitoring data available")
            return
        
        print("\n📈 Generating Monitoring Report...")
        
        # Extract data for plotting
        timestamps = [entry['timestamp'] for entry in self.monitoring_data]
        
        if self.monitoring_data[0]['gpu_info']:
            gpu_memory = [entry['gpu_info'][0]['memory_used'] for entry in self.monitoring_data]
            gpu_util = [entry['gpu_info'][0]['utilization'] for entry in self.monitoring_data]
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # GPU Memory usage
            ax1.plot(range(len(gpu_memory)), gpu_memory, 'b-', label='GPU Memory Used (MB)')
            ax1.set_title('GPU Memory Usage Over Time')
            ax1.set_ylabel('Memory (MB)')
            ax1.legend()
            ax1.grid(True)
            
            # GPU Utilization
            ax2.plot(range(len(gpu_util)), gpu_util, 'r-', label='GPU Utilization (%)')
            ax2.set_title('GPU Utilization Over Time')
            ax2.set_ylabel('Utilization (%)')
            ax2.set_xlabel('Time Steps')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('gpu_monitoring_report.png', dpi=150, bbox_inches='tight')
            print("📊 GPU monitoring charts saved as 'gpu_monitoring_report.png'")
        
        # Save summary statistics
        summary = {
            'monitoring_duration': len(self.monitoring_data),
            'start_time': self.monitoring_data[0]['timestamp'],
            'end_time': self.monitoring_data[-1]['timestamp'],
            'average_gpu_memory_usage': sum([entry['gpu_info'][0]['memory_used'] for entry in self.monitoring_data]) / len(self.monitoring_data) if self.monitoring_data and self.monitoring_data[0]['gpu_info'] else 0,
            'average_gpu_utilization': sum([entry['gpu_info'][0]['utilization'] for entry in self.monitoring_data]) / len(self.monitoring_data) if self.monitoring_data and self.monitoring_data[0]['gpu_info'] else 0,
        }
        
        with open('monitoring_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("📋 Summary saved as 'monitoring_summary.json'")
    
    def run_monitoring(self, duration_minutes=60, interval_seconds=30):
        """Run continuous monitoring."""
        print(f"🔍 Starting monitoring for {duration_minutes} minutes (interval: {interval_seconds}s)")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                self.monitor_once()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        
        # Save data and generate report
        self.save_data()
        self.generate_report()

def main():
    parser = argparse.ArgumentParser(description="Monitor Qwen 2.5 Omni training")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in minutes")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Training log directory")
    parser.add_argument("--output", type=str, default="monitoring_report.json", help="Output file for monitoring data")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(log_dir=args.log_dir, output_file=args.output)
    monitor.run_monitoring(duration_minutes=args.duration, interval_seconds=args.interval)

if __name__ == "__main__":
    main()
