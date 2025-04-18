"""
stream_and_log_gaze.py

- Subscribes to Pupil Core's IPC Backbone (gaze.* topics).
- Extracts only norm_pos, gaze_point_3d, confidence, timestamp.
- Streams each sample via UDP to a downstream process.
- Logs all samples to a JSON file named with the session start datetime.
"""

"""
norm_pos
Two floats between 0 and 1 indicating where on the world‐camera image the user is looking (x=0 left…1=right; y=0 top…1=bottom) 
https://docs.pupil-labs.com/neon/data-collection/data-streams/
.
gaze_point_3d
The 3D coordinates in the world camera’s reference frame. Useful when you have a calibrated 3D scene and want metric gaze intersection points 
https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/raw_data_exporter.py
.
confidence
How reliable the estimate is; you may wish to discard samples with confidence < 0.6 or similar

.
pupil_timestamp
The Pupil Core’s internal time (in seconds since session start). Use it for precise synchronization with other streams (e.g. audio) 
https://docs.pupil-labs.com/core/developer/network-api/
.
local_timestamp
The time (in UNIX epoch seconds) at which your script received the sample—handy for logging/debugging but not strictly necessary for real‑time steering.
"""

import zmq
import msgpack
import socket
import time
import json
from datetime import datetime

# CONFIGURATION
PUPIL_IP       = "127.0.0.1"
PUPIL_REMOTE_PORT = 50020       # Port for Pupil Remote REQ
TOPIC_FILTER   = "gaze."        # Subscribe to all gaze.* topics
UDP_TARGET_IP  = "127.0.0.1"    # Where to stream filtered gaze data
UDP_TARGET_PORT= 2400           # Port for downstream process

def get_ipc_sub_port(ip=PUPIL_IP, remote_port=PUPIL_REMOTE_PORT):
    """Request the IPC Backbone SUB_PORT from Pupil Remote."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{ip}:{remote_port}")
    sock.send_string("SUB_PORT")
    sub_port = sock.recv_string()
    sock.close()
    return sub_port

def main():
    # 1) Determine subscription port
    sub_port = get_ipc_sub_port()
    print(f"[INFO] IPC SUB_PORT = {sub_port}")

    # 2) Open ZMQ SUB socket for gaze data
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{PUPIL_IP}:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_FILTER)

    # 3) Prepare UDP socket for streaming
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 4) Prepare log file
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"gaze_log_{start_time}.json"
    log = []

    print("[INFO] Streaming to UDP port", UDP_TARGET_PORT)
    print("[INFO] Logging to file", log_filename)
    print("Press Ctrl+C to exit and save.")

    try:
        while True:
            topic, payload = sub.recv_multipart()
            # Unpack msgpack payload
            data = msgpack.unpackb(payload, raw=False)

            # Extract the fields we care about
            filtered = {
                "topic": topic.decode("utf-8"),
                "pupil_timestamp": data.get("timestamp"),
                "confidence":     data.get("confidence"),
                "norm_pos":       data.get("norm_pos"),
                "gaze_point_3d":  data.get("gaze_point_3d"),
                # Optionally add local reception timestamp:
                "local_timestamp": time.time()
            }

            # 5) Stream via UDP
            msg = json.dumps(filtered).encode("utf-8")
            udp.sendto(msg, (UDP_TARGET_IP, UDP_TARGET_PORT))

            # 6) Append to log
            log.append(filtered)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Saving log to", log_filename)
        with open(log_filename, "w") as f:
            json.dump(log, f, indent=2)
        print("[INFO] Done.")

    finally:
        sub.close()
        ctx.term()
        udp.close()

if __name__ == "__main__":
    main()
