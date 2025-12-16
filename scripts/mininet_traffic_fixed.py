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
        
        # Initialize CSV file with headers
        self.initialize_csv()
    
    def initialize_csv(self):
        """Initialize CSV file with headers matching UNSW-NB15 features"""
        headers = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 
            'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 
            'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 
            'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 
            'is_sm_ips_ports', 'label'
        ]
        
        if not os.path.exists(self.output_file):
            pd.DataFrame(columns=headers).to_csv(self.output_file, index=False)
            print(f"Created new CSV file with headers: {self.output_file}")
    
    def simulate_flow_features(self):
        """Simulate realistic network flow features with diverse traffic types"""
        features = {}
        
        # Basic features
        features['dur'] = np.random.exponential(8)
        features['proto'] = np.random.choice([0, 1, 2])  # 0: TCP, 1: UDP, 2: ICMP
        features['service'] = np.random.choice([0, 1, 2, 3])  # Different services
        features['state'] = np.random.choice([0, 1, 2, 3, 4])  # Connection states
        
        # Choose traffic type with realistic distribution
        traffic_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All 10 classes
        weights = [0.15, 0.05, 0.10, 0.05, 0.10, 0.05, 0.30, 0.10, 0.05, 0.05]  # Probabilities
        
        traffic_label = np.random.choice(traffic_types, p=weights)
        features['label'] = traffic_label
        
        # Make features distinctive for each traffic type
        if traffic_label == 0:  # Analysis
            features['spkts'] = np.random.poisson(20)
            features['dpkts'] = np.random.poisson(15)
            features['sbytes'] = np.random.poisson(1500)
            features['dbytes'] = np.random.poisson(1200)
            features['dur'] = np.random.exponential(12)  # Longer duration
            features['sttl'] = np.random.randint(100, 200)
            
        elif traffic_label == 6:  # Normal
            features['spkts'] = np.random.poisson(8)
            features['dpkts'] = np.random.poisson(6)
            features['sbytes'] = np.random.poisson(800)
            features['dbytes'] = np.random.poisson(600)
            features['dur'] = np.random.exponential(5)
            features['sttl'] = np.random.randint(50, 150)
            
        elif traffic_label == 7:  # Reconnaissance
            features['spkts'] = np.random.poisson(5)
            features['dpkts'] = np.random.poisson(3)
            features['sbytes'] = np.random.poisson(500)
            features['dbytes'] = np.random.poisson(400)
            features['dur'] = np.random.exponential(3)
            features['sttl'] = np.random.randint(40, 100)
            
        elif traffic_label == 2:  # DoS
            features['spkts'] = np.random.poisson(100)
            features['dpkts'] = np.random.poisson(80)
            features['sbytes'] = np.random.poisson(5000)
            features['dbytes'] = np.random.poisson(4000)
            features['dur'] = np.random.exponential(2)
            features['sttl'] = np.random.randint(30, 80)
            
        else:  # Other attack types
            features['spkts'] = np.random.poisson(15)
            features['dpkts'] = np.random.poisson(12)
            features['sbytes'] = np.random.poisson(1200)
            features['dbytes'] = np.random.poisson(900)
            features['dur'] = np.random.exponential(6)
            features['sttl'] = np.random.randint(60, 180)
        
        # Common features
        features['rate'] = np.random.gamma(3, 2)
        features['dttl'] = features['sttl'] - np.random.randint(5, 20)
        features['sload'] = np.random.gamma(8, 10)
        features['dload'] = np.random.gamma(6, 8)
        features['sloss'] = np.random.poisson(1)
        features['dloss'] = np.random.poisson(1)
        
        # Set missing features to 0
        all_features = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
            'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
            'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
            'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
            'is_sm_ips_ports'
        ]
        
        for feat in all_features:
            if feat not in features:
                features[feat] = 0
        
        return features
    
    def capture_flow_data(self):
        """Capture flow data and save to CSV"""
        try:
            # Simulate capturing multiple flows
            num_flows = np.random.randint(2, 6)
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
            
            # Print label distribution
            print("Label distribution:", df['label'].value_counts().sort_index().to_dict())
    
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
