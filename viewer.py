#!/usr/bin/env python3
"""
Event Gallery GUI for Visualizing Radiation Patches

Allows a user to:
1) Select a CSV file of event metadata
2) Select a directory containing .npy patches for each event
3) Display a table (gallery) of all events
4) Choose a plot style (2D gray, 3D surface, 3D wireframe, 3D scatter, 2D contour)
5) Plot the selected event patch in a separate Matplotlib window

Requirements:
    pip install pyqt5 opencv-python numpy matplotlib
"""

import sys
import os
import csv
import numpy as np
import cv2

import matplotlib
matplotlib.use("Qt5Agg")  # or "TkAgg", either is fine for separate pop-up windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QLabel
)
from PyQt5.QtCore import Qt

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------

def load_events_from_csv(csv_path, patches_dir):
    """
    Reads the CSV and returns a list of dict entries (one per event).
    Each dict has keys like: 'event_id','patch_path','sum_intensity','avg_intensity', etc.
    """
    events = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We'll handle the essential columns. Adjust as needed for your CSV structure
            try:
                event_id = int(row["event_id"])
                sum_intensity = float(row["sum_intensity"])
                avg_intensity = float(row["avg_intensity"])
                patch_filename = row["patch_filename"]
                # bounding box columns
                rmin = int(row["rmin"])
                cmin = int(row["cmin"])
                rmax = int(row["rmax"])
                cmax = int(row["cmax"])
            except KeyError:
                # Skip rows missing required fields
                continue
            except (ValueError, TypeError):
                continue

            patch_path = os.path.join(patches_dir, patch_filename)
            event_dict = {
                "event_id": event_id,
                "sum_intensity": sum_intensity,
                "avg_intensity": avg_intensity,
                "rmin": rmin,
                "cmin": cmin,
                "rmax": rmax,
                "cmax": cmax,
                "patch_path": patch_path
            }
            events.append(event_dict)
    return events

def load_patch(patch_path):
    """
    Loads the .npy patch and returns a 2D numpy array (float32).
    """
    if not os.path.exists(patch_path):
        print(f"Patch not found: {patch_path}")
        return None
    patch = np.load(patch_path)
    return patch.astype(np.float32)

def plot_patch(patch, style="2D Gray", title="Event Patch"):
    """
    Creates a new pop-up Matplotlib window showing the patch in the desired style.
    Non-blocking so you can open multiple windows.
    """
    if patch is None or patch.size == 0:
        print("Invalid patch data. Cannot plot.")
        return

    # turn off blocking so user can keep the main GUI open
    plt.ion()

    # Prepare data for 3D
    rows, cols = patch.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z    = patch

    if style == "2D Gray":
        fig = plt.figure()
        plt.title(title + " (2D Gray)")
        plt.imshow(patch, cmap='gray', origin='upper')
        plt.colorbar(label='Intensity')
        plt.show(block=False)

    elif style == "3D Surface":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
        ax.set_title(title + " (3D Surface)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        plt.tight_layout()
        plt.show(block=False)

    elif style == "3D Wireframe":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, color='green')
        ax.set_title(title + " (3D Wireframe)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        plt.tight_layout()
        plt.show(block=False)

    elif style == "3D Scatter":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z.flatten()
        scat = ax.scatter(Xf, Yf, Zf, c=Zf, cmap='jet', s=10, alpha=0.7)
        ax.set_title(title + " (3D Scatter)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        fig.colorbar(scat, label='Intensity')
        plt.tight_layout()
        plt.show(block=False)

    elif style == "2D Contour":
        fig = plt.figure()
        plt.title(title + " (2D Contour)")
        cs = plt.contour(Z, levels=15, cmap='jet')
        plt.clabel(cs, inline=True, fontsize=8)
        plt.colorbar()
        plt.show(block=False)

    else:
        print(f"Unknown style: {style}")

# ------------------------------------------------
# Main Window
# ------------------------------------------------

class EventGalleryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Gallery GUI")
        self.resize(900, 500)

        # We'll store events in a list of dicts
        self.events = []
        self.csv_path = ""
        self.patches_dir = ""

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top buttons to load CSV / folder
        btn_layout = QHBoxLayout()
        self.btn_load_csv = QPushButton("Load CSV")
        self.btn_load_csv.clicked.connect(self.on_load_csv)
        btn_layout.addWidget(self.btn_load_csv)

        self.btn_load_patches = QPushButton("Load Patches Folder")
        self.btn_load_patches.clicked.connect(self.on_load_patches)
        btn_layout.addWidget(self.btn_load_patches)

        self.btn_load_events = QPushButton("Parse Events")
        self.btn_load_events.clicked.connect(self.on_parse_events)
        btn_layout.addWidget(self.btn_load_events)

        main_layout.addLayout(btn_layout)

        # Table of events
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Event ID", "Sum Int", "Avg Int", "BBox", "Patch Path"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        main_layout.addWidget(self.table)

        # Bottom row: style + plot button
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel("Plot Style:"))
        self.combo_style = QComboBox()
        self.combo_style.addItems([
            "2D Gray",
            "2D Contour",
            "3D Surface",
            "3D Wireframe",
            "3D Scatter"
        ])
        bottom_layout.addWidget(self.combo_style)

        self.btn_plot = QPushButton("Plot Selected Event")
        self.btn_plot.clicked.connect(self.on_plot_event)
        bottom_layout.addWidget(self.btn_plot)

        main_layout.addLayout(bottom_layout)

    def on_load_csv(self):
        """
        Ask user for a CSV file.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.csv_path = path
            print("Selected CSV:", path)

    def on_load_patches(self):
        """
        Ask user for a folder containing .npy patches.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select patches folder")
        if folder:
            self.patches_dir = folder
            print("Selected patches dir:", folder)

    def on_parse_events(self):
        """
        Load events from CSV + patches dir -> display in table.
        """
        if not self.csv_path or not self.patches_dir:
            print("CSV or patches folder not selected.")
            return

        # Load
        self.events = load_events_from_csv(self.csv_path, self.patches_dir)
        print(f"Loaded {len(self.events)} events from CSV.")

        # Populate table
        self.table.setRowCount(len(self.events))
        for row_idx, ev in enumerate(self.events):
            # event_id
            item_id = QTableWidgetItem(str(ev["event_id"]))
            self.table.setItem(row_idx, 0, item_id)
            # sum_int
            item_sum = QTableWidgetItem(f"{ev['sum_intensity']:.1f}")
            self.table.setItem(row_idx, 1, item_sum)
            # avg_int
            item_avg = QTableWidgetItem(f"{ev['avg_intensity']:.1f}")
            self.table.setItem(row_idx, 2, item_avg)
            # bounding box
            rmin,cmin,rmax,cmax = ev["rmin"], ev["cmin"], ev["rmax"], ev["cmax"]
            item_bbox = QTableWidgetItem(f"({rmin},{cmin})-({rmax},{cmax})")
            self.table.setItem(row_idx, 3, item_bbox)
            # patch path
            item_path = QTableWidgetItem(ev["patch_path"])
            self.table.setItem(row_idx, 4, item_path)

    def on_plot_event(self):
        """
        Plot whichever event is selected in the table, using the chosen style.
        """
        sel_row = self.table.currentRow()
        if sel_row < 0:
            print("No event selected.")
            return
        if sel_row >= len(self.events):
            print("Invalid selection.")
            return

        ev = self.events[sel_row]
        patch = load_patch(ev["patch_path"])
        style = self.combo_style.currentText()
        title = f"Event {ev['event_id']} - Sum={ev['sum_intensity']:.1f}, Avg={ev['avg_intensity']:.1f}"
        plot_patch(patch, style=style, title=title)

# --------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = EventGalleryWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
