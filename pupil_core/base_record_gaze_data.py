"""
base_record_gaze_data.py

Standalone script to record live gaze data from PupilCore's IPC Backbone.

How it works:
  1. Connects to the Pupil Remote (REQ socket on tcp://127.0.0.1:50020) to request
     the session's unique SUB_PORT (and PUB_PORT, if needed).
  2. Opens a SUB socket to the returned SUB_PORT and subscribes to topics starting with "gaze.".
  3. Receives msgpack-encoded messages, decodes them, and records them with a local timestamp.
  4. Upon Ctrl+C, saves all recorded messages to "gaze_data.json".

Setup Instructions:
  - Install Pupil Capture on your Mac and connect your PupilCore glasses.
  - In Pupil Capture, ensure that:
      • The "Pupil Remote" plugin is enabled (typically on port 50020).
      • The "ZMQ Pub" (or IPC Backbone) plugin is enabled.
      • Your eyes are being tracked (gaze markers are visible).
  - On your Mac, ensure your firewall permits incoming connections on the relevant ports.
  - Install dependencies:
        pip install pyzmq msgpack
  - Save this script as record_gaze_ipc.py and run it:
        python record_gaze_ipc.py
"""

import zmq
import msgpack
import time
import json
import sys

def get_ipc_ports(ip="127.0.0.1", remote_port=50020):
    """
    Connects to Pupil Remote and requests the IPC SUB and PUB ports.
    
    Returns:
        sub_port (str): The port for subscribing to IPC data.
        pub_port (str): The port for publishing (if needed).
    """
    context = zmq.Context.instance()
    req_socket = context.socket(zmq.REQ)
    try:
        req_socket.connect(f"tcp://{ip}:{remote_port}")
    except Exception as e:
        print(f"Error connecting to Pupil Remote at tcp://{ip}:{remote_port}: {e}")
        sys.exit(1)
    
    # Request the SUB port.
    req_socket.send_string("SUB_PORT")
    sub_port = req_socket.recv_string()
    
    # Request the PUB port (if needed).
    req_socket.send_string("PUB_PORT")
    pub_port = req_socket.recv_string()
    
    req_socket.close()
    return sub_port, pub_port

def record_gaze_data(ip="127.0.0.1", remote_port=50020, topic_filter="gaze.", out_file="gaze_data.json"):
    """
    Records real-time gaze data from the IPC Backbone.
    
    Args:
        ip (str): IP address where Pupil Capture is running.
        remote_port (int): Port for the Pupil Remote plugin (default 50020).
        topic_filter (str): Topic filter for gaze messages (usually "gaze.").
        out_file (str): File name to save recorded gaze data.
    """
    print("Requesting IPC Backbone ports from Pupil Remote...")
    sub_port, pub_port = get_ipc_ports(ip, remote_port)
    print(f"Received SUB_PORT: {sub_port}, PUB_PORT: {pub_port}")
    
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    # Connect to the unique SUB port provided by Pupil Remote.
    subscriber.connect(f"tcp://{ip}:{sub_port}")
    # Subscribe to all topics starting with "gaze."
    subscriber.setsockopt_string(zmq.SUBSCRIBE, topic_filter)
    
    print(f"Connected to IPC Backbone at tcp://{ip}:{sub_port} with topic filter '{topic_filter}'")
    print("Recording gaze data. Press Ctrl+C to stop and save data.")
    
    recorded_data = []
    
    try:
        while True:
            # Receive a multipart message: [topic, payload]
            msg_parts = subscriber.recv_multipart()
            if len(msg_parts) < 2:
                continue
            topic = msg_parts[0].decode('utf-8')
            payload = msg_parts[1]
            try:
                # Unpack using msgpack (the IPC Backbone uses msgpack for serialization)
                data = msgpack.unpackb(payload, raw=False)
            except Exception as e:
                print("Error unpacking message:", e)
                continue
            
            # Add a local timestamp (for debugging purposes)
            data["local_timestamp"] = time.time()
            recorded_data.append({"topic": topic, "data": data})
            print(f"Recorded message from topic '{topic}'.")
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Saving gaze data...")
        try:
            with open(out_file, "w") as f:
                json.dump(recorded_data, f, indent=4)
            print(f"Gaze data successfully saved to '{out_file}'.")
        except Exception as e:
            print("Error saving gaze data:", e)
    finally:
        subscriber.close()
        context.term()
        print("Exiting program.")

if __name__ == "__main__":
    record_gaze_data()
