#!/usr/bin/env python3

import time
import subprocess
import threading
import pandas as pd
import numpy as np
import os
from datetime import datetime

class SDNTrafficGenerator:
    def __init__(self, capture_interval=10, output_file='../live_traffic.csv'):
        self.capture_interval = capture_interval
        self.output_file = output_file
        self.is_capturing = False
        self.capture_thread = None
        
        # Initialize CSV file with headers matching UNSW-NB15 features
        self.initialize_csv()
    
    def initialize_csv(self):
        """Initialize CSV file with headers similar to UNSW-NB15"""
        # These are the most important features from UNSW-NB15 that we can capture in real-time
        headers = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
            'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
            'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
            'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
            'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'label'
        ]
        
        if not os.path.exists(self.output_file):
            pd.DataFrame(columns=headers).to_csv(self.output_file, index=False)
            print(f"Created new CSV file with headers: {self.output_file}")
    
    def simulate_flow_features(self):
        """Simulate flow features similar to UNSW-NB15"""
        # This is a simplified simulation - in a real scenario, you'd capture actual flow stats
        features = {}
        
        # Basic features
        features['dur'] = np.random.exponential(5)  # duration
        features['proto'] = np.random.choice([0, 1, 2])  # protocol (simplified)
        features['service'] = np.random.choice([0, 1, 2, 3])  # service (simplified)
        features['state'] = np.random.choice([0, 1, 2, 3, 4])  # state (simplified)
        
        # Packet features
        features['spkts'] = np.random.poisson(10)  # source packets
        features['dpkts'] = np.random.poisson(8)   # destination packets
        features['sbytes'] = np.random.poisson(1000)  # source bytes
        features['dbytes'] = np.random.poisson(800)   # destination bytes
        
        # Rate features
        features['rate'] = np.random.gamma(2, 2)  # rate
        features['sload'] = np.random.gamma(5, 10)  # source load
        features['dload'] = np.random.gamma(4, 8)   # destination load
        
        # Other features (simplified)
        features['sttl'] = np.random.randint(40, 250)  # source TTL
        features['dttl'] = np.random.randint(40, 250)  # destination TTL
        features['sloss'] = np.random.poisson(1)  # source loss
        features['dloss'] = np.random.poisson(1)  # destination loss
        
        # For demonstration, we'll mark some flows as potentially malicious
        # In real application, this would be determined by the model
        features['label'] = 0  # Default to normal
        
        # Mark as suspicious if certain conditions are met
        if (features['sbytes'] > 5000 or features['dbytes'] > 5000 or 
            features['rate'] > 20 or features['spkts'] > 50):
            features['label'] = 1  # Mark as potential attack
        
        return features
    
    def capture_flow_data(self):
        """Capture flow data and save to CSV"""
        try:
            # Simulate capturing multiple flows
            num_flows = np.random.randint(1, 5)
            flows = []
            
            for _ in range(num_flows):
                flow_features = self.simulate_flow_features()
                flows.append(flow_features)
            
            return flows
        except Exception as e:
            print(f"Error capturing flow data: {e}")
            return []
    
    def save_flows(self, flows):
        """Save flows to CSV file"""
        if flows:
            df = pd.DataFrame(flows)
            
            # Append to existing CSV file
            if os.path.exists(self.output_file):
                existing_df = pd.read_csv(self.output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(self.output_file, index=False)
            print(f"Saved {len(flows)} flows to {self.output_file}")
    
    def start_capture(self):
        """Start capturing traffic"""
        self.is_capturing = True
        
        def capture_loop():
            while self.is_capturing:
                flows = self.capture_flow_data()
                self.save_flows(flows)
                time.sleep(self.capture_interval)
        
        self.capture_thread = threading.Thread(target=capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print(f"Started capturing traffic every {self.capture_interval} seconds")
    
    def stop_capture(self):
        """Stop capturing traffic"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Stopped capturing traffic")

def main():
    # Create traffic generator instance
    traffic_generator = SDNTrafficGenerator(capture_interval=5)
    
    try:
        # Start traffic capture
        traffic_generator.start_capture()
        
        # Run for a while
        print("Running traffic simulation for 60 seconds...")
        time.sleep(60)
        
        # Stop capture
        traffic_generator.stop_capture()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        traffic_generator.stop_capture()

if __name__ == '__main__':
    main()
