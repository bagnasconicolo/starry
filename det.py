#!/usr/bin/env python3
"""
Script 1: Real-Time Detection & Logging (with Patches + Surroundings), with User-Selected Initial Run Number

Features:
  - Detects ionizing radiation events via background subtraction & DBSCAN.
  - Saves each event to:
      * A CSV (event_log_runX.csv) with bounding-box info, sum/avg intensities, etc.
      * A patches_runX/ folder containing each event patch as .npy
        BUT with extra "margin" around the minimal bounding box.
  - Asks the user for an initial run number before data acquisition.
  - Multiple runs support (press 'n' to start a new run after the first).
  - On-screen trackbars for thresholds, plus bounding box labeling.
  - 'p' to show 3D plot of the last frame, 'd' for sum_int distribution (current run).
  - Press 'q' to quit.

You can later load these patches in Script 2 (aggregator).

Requirements:
    pip install opencv-python numpy matplotlib scikit-learn
"""

import cv2
import numpy as np
import os
import csv
import time
import collections
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import sys

# --------------- Globals ---------------
CAMERA_INDEX            = 0
FRAME_WIDTH             = None   # e.g. 1280
FRAME_HEIGHT            = None   # e.g. 720
FRAME_RESIZE_FACTOR     = 1.0

BACKGROUND_LEARNING_RATE= 0.01
EPS_DISTANCE            = 2.0   # DBSCAN
MORPH_KERNEL_SIZE       = (3,3)
MAX_DISPLAY_FRAMES      = 30

# Trackbar defaults
THRESHOLD_DEFAULT       = 100
MIN_CLUSTER_SIZE_DEFAULT= 5
MIN_CLUSTER_INT_DEFAULT = 500

# **Margin** for bounding box expansions
MARGIN                  = 10  # extra surroundings in each dimension

# --------------------------------------

def nothing(x):
    pass

def initialize_camera(index=0, w=None, h=None):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera index={index}")
    if w is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    if h is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

def rolling_avg_bg(bg_frame, alpha, new_frame):
    return cv2.addWeighted(bg_frame, alpha, new_frame, 1 - alpha, 0)

def detect_events(gray, background, thr_val, min_clust_size, min_clust_int):
    """
    Finds clusters of bright pixels. Returns a list of event dicts, each with bounding_box + intensities.
    """
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))
    _, bin_mask = cv2.threshold(diff, thr_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)

    coords = np.column_stack(np.where(bin_mask > 0))
    if len(coords) < 1:
        return []

    db = DBSCAN(eps=EPS_DISTANCE, min_samples=min_clust_size).fit(coords)
    labels = db.labels_

    events = []
    for lbl in set(labels):
        if lbl == -1:
            # noise
            continue
        cluster_mask = (labels == lbl)
        cluster_pts  = coords[cluster_mask]

        sum_intensity = np.sum(gray[cluster_pts[:, 0], cluster_pts[:, 1]])
        if sum_intensity < min_clust_int:
            continue
        max_intensity = np.max(gray[cluster_pts[:, 0], cluster_pts[:, 1]])
        size = len(cluster_pts)

        rmin, cmin = np.min(cluster_pts, axis=0)
        rmax, cmax = np.max(cluster_pts, axis=0)

        events.append({
            "label": lbl,
            "sum_intensity": sum_intensity,
            "max_intensity": max_intensity,
            "cluster_size": size,
            "bounding_box": (int(rmin), int(cmin), int(rmax), int(cmax))
        })
    return events

def plot_3d_frame(gray_frame, events):
    """
    Creates a 3D surface plot of the last grayscale frame,
    plus bounding-box corners in red.
    """
    plt.ion()
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    rows, cols = gray_frame.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    ax.plot_surface(X, Y, gray_frame, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title("3D Last Frame + Events")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensity")

    # optional bounding-box corners
    for evt in events:
        (rmin, cmin, rmax, cmax) = evt["bounding_box"]
        corners = [(rmin,cmin),(rmin,cmax),(rmax,cmin),(rmax,cmax)]
        Zvals   = [gray_frame[r,c] for (r,c) in corners]
        Xvals   = [c for (r,c) in corners]
        Yvals   = [r for (r,c) in corners]
        ax.scatter(Xvals, Yvals, Zvals, c='r', s=20, alpha=0.6)

    plt.show(block=False)

def plot_distribution(sum_list):
    """
    Quick histogram of sum_intensity for the current run.
    """
    plt.ion()
    if not sum_list:
        print("No events in this run.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(sum_list, bins=20, color='blue', alpha=0.7)
    ax.set_title("Sum Intensity Distribution (current run)")
    ax.set_xlabel("sum_intensity")
    ax.set_ylabel("Frequency")
    plt.show(block=False)

# --------------- MultiRunDetector ---------------
class MultiRunDetector:
    def __init__(self, initial_run_id=1):
        """
        Pass in initial_run_id. The script will start from that run number,
        and increment for subsequent runs if user presses 'n'.
        """
        # trackbars
        cv2.namedWindow("Controls", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Threshold",     "Controls", THRESHOLD_DEFAULT, 255, nothing)
        cv2.createTrackbar("MinClusterSize","Controls", MIN_CLUSTER_SIZE_DEFAULT, 100, nothing)
        cv2.createTrackbar("MinClusterInt", "Controls", MIN_CLUSTER_INT_DEFAULT, 5000, nothing)

        # init camera
        self.cap = initialize_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Cannot read initial frame from camera.")

        if FRAME_RESIZE_FACTOR != 1.0:
            frame = cv2.resize(frame, None, fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR,
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.background = gray.astype(np.float32)

        self.frame_history = collections.deque(maxlen=MAX_DISPLAY_FRAMES)

        # multi-run
        self.run_id         = initial_run_id - 1  # so that first start_new_run => user_run_id
        self.run_event_count= 0
        self.run_sum_list   = []
        self.run_start_time = None
        self.csv_file       = None
        self.csv_writer     = None
        self.frame_index    = 0

        # start the first run
        self.start_new_run()

    def start_new_run(self):
        if self.csv_file:
            self.csv_file.close()

        self.run_id += 1
        self.run_event_count = 0
        self.run_sum_list    = []
        self.run_start_time  = time.time()

        csv_name = f"event_log_run{self.run_id}.csv"
        self.csv_file = open(csv_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "event_id","frame_index",
            "rmin","cmin","rmax","cmax",
            "sum_intensity","max_intensity","cluster_size",
            "avg_intensity","patch_filename"
        ])

        patches_dir = f"patches_run{self.run_id}"
        os.makedirs(patches_dir, exist_ok=True)

        print(f"=== Started RUN #{self.run_id}, writing to {csv_name}, patches => {patches_dir} ===")

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def main_loop(self):
        global MARGIN
        event_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No frame from camera.")
                break

            if FRAME_RESIZE_FACTOR != 1.0:
                frame = cv2.resize(frame, None, fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR,
                                   interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            alpha = 1 - BACKGROUND_LEARNING_RATE
            self.background = rolling_avg_bg(self.background, alpha, gray.astype(np.float32))

            thr  = cv2.getTrackbarPos("Threshold","Controls")
            msz  = cv2.getTrackbarPos("MinClusterSize","Controls")
            mint = cv2.getTrackbarPos("MinClusterInt","Controls")

            events = detect_events(gray, self.background, thr, msz, mint)

            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            h, w  = gray.shape
            for evt in events:
                self.run_event_count += 1
                event_id += 1
                (rmin, cmin, rmax, cmax) = evt["bounding_box"]

                # Expand bounding box by MARGIN
                rmin_m = max(rmin - MARGIN, 0)
                cmin_m = max(cmin - MARGIN, 0)
                rmax_m = min(rmax + MARGIN, h-1)
                cmax_m = min(cmax + MARGIN, w-1)

                cv2.rectangle(disp, (cmin_m, rmin_m), (cmax_m, rmax_m), (0,0,255), 1)
                label_text = f"Evt{event_id}:S={int(evt['sum_intensity'])}"
                cv2.putText(disp, label_text, (cmin_m, rmax_m+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)

                avg_intensity = evt["sum_intensity"]/evt["cluster_size"]
                self.run_sum_list.append(evt["sum_intensity"])

                # Save patch
                patch = gray[rmin_m:rmax_m+1, cmin_m:cmax_m+1]
                patch_fname= f"evt_{event_id}.npy"
                run_patches_dir = f"patches_run{self.run_id}"
                patch_path= os.path.join(run_patches_dir, patch_fname)
                np.save(patch_path, patch)

                # Log
                self.csv_writer.writerow([
                    event_id,
                    self.frame_index,
                    rmin_m, cmin_m, rmax_m, cmax_m,
                    evt["sum_intensity"],
                    evt["max_intensity"],
                    evt["cluster_size"],
                    f"{avg_intensity:.2f}",
                    patch_fname
                ])

            elapsed = time.time() - self.run_start_time
            cv2.putText(disp, f"RUN #{self.run_id}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
            cv2.putText(disp, f"Time: {int(elapsed)}s", (10,45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
            cv2.putText(disp, f"Events: {self.run_event_count}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

            cv2.imshow("Real-Time Detection (Press 'n' new run, 'q' quit)", disp)
            self.frame_history.append((gray.copy(), events))
            self.frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.start_new_run()
                event_id = 0
            elif key == ord('p'):
                if self.frame_history:
                    lf, le = self.frame_history[-1]
                    plot_3d_frame(lf, le)
            elif key == ord('d'):
                plot_distribution(self.run_sum_list)

        self.close()

def main():
    # ask user for initial run number
    default_run_id = 1
    user_in = input(f"Enter initial run number [default={default_run_id}]: ")
    try:
        if user_in.strip():
            initial_run_id= int(user_in)
        else:
            initial_run_id= default_run_id
    except ValueError:
        initial_run_id= default_run_id

    plt.ion()
    app = MultiRunDetector(initial_run_id=initial_run_id)
    try:
        app.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()

if __name__ == "__main__":
    main()
