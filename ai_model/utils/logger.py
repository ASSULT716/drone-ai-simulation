import csv
import time

def save_log(file_path, state, action):
    # state: [rel_x, rel_y, rel_z, threat, hp, is_damaged]
    # action: [out_x, out_y, out_z, yaw, pitch]
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([time.time()] + state + action)