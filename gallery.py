#!/usr/bin/env python3
"""
Gallery-Style GUI for Radiation Event Patches (PyQt5)

Displays a scrollable grid of event thumbnails. Each event is shown as:
  - A 64x64 grayscale thumbnail
  - A small label with ID, sum_int, avg_int
  - A "Plot" button that opens a Matplotlib window in the chosen style.

Requirements:
    pip install pyqt5 opencv-python numpy matplotlib

CSV columns assumed:
    event_id, sum_intensity, avg_intensity, rmin, cmin, rmax, cmax, patch_filename
"""

import sys, os, csv
import numpy as np
import cv2

import matplotlib
matplotlib.use("Qt5Agg")  # Non-blocking separate windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QScrollArea, QGridLayout, QLabel,
    QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage

# ------------------ Patch Plotting ------------------

def plot_patch(patch, style="2D Gray", title="Event Patch"):
    """
    Creates a Matplotlib window showing `patch` in the chosen style.
    Non-blocking so user can keep the GUI open.
    """
    if patch is None or patch.size==0:
        print("Invalid patch. Cannot plot.")
        return

    plt.ion()
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
        xf = X.flatten()
        yf = Y.flatten()
        zf = Z.flatten()
        scat = ax.scatter(xf, yf, zf, c=zf, cmap='jet', alpha=0.7, s=10)
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

# ------------------ CSV / Patches Reading ------------------

def load_events(csv_path, patches_dir):
    """
    Reads the CSV, merges with patch paths, returns a list of dicts:
      {
        'event_id': int,
        'sum_intensity': float,
        'avg_intensity': float,
        'rmin': int, 'cmin': int, 'rmax': int, 'cmax': int,
        'patch_path': str
      }
    """
    events = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                event_id = int(row["event_id"])
                sum_int  = float(row["sum_intensity"])
                avg_int  = float(row["avg_intensity"])
                rmin     = int(row["rmin"])
                cmin     = int(row["cmin"])
                rmax     = int(row["rmax"])
                cmax     = int(row["cmax"])
                patch_fname = row["patch_filename"]
            except (KeyError, ValueError):
                continue
            patch_path = os.path.join(patches_dir, patch_fname)
            e = {
                "event_id": event_id,
                "sum_intensity": sum_int,
                "avg_intensity": avg_int,
                "rmin": rmin, "cmin": cmin, "rmax": rmax, "cmax": cmax,
                "patch_path": patch_path
            }
            events.append(e)
    return events

def load_patch(patch_path):
    if not os.path.exists(patch_path):
        print(f"Missing patch: {patch_path}")
        return None
    arr = np.load(patch_path)
    return arr.astype(np.float32)

def patch_to_thumbnail(patch, thumb_size=64):
    """
    Convert patch -> grayscale QPixmap (thumb_size x thumb_size).
    We'll do a min-max normalize -> 0..255, then scale to thumb_size.
    """
    if patch is None or patch.size==0:
        return QPixmap()

    vmin, vmax = patch.min(), patch.max()
    if vmax == vmin:
        # all pixels same => zero range
        norm = np.zeros_like(patch, dtype=np.uint8)
    else:
        norm = (patch - vmin)/(vmax - vmin)*255.0
    norm = norm.astype(np.uint8)

    # resize to create a 2D thumbnail
    h, w = norm.shape
    if (h!=thumb_size or w!=thumb_size):
        # use OpenCV to resize
        norm_resized = cv2.resize(norm, (thumb_size, thumb_size),
                                  interpolation=cv2.INTER_AREA)
    else:
        norm_resized = norm

    # Convert to QImage
    qimg = QImage(norm_resized.data, thumb_size, thumb_size,
                  thumb_size, QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap

# ------------------ "Thumbnail" Widget ------------------

from PyQt5.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

class EventThumbWidget(QFrame):
    """
    Displays:
      - A 64x64 grayscale thumbnail
      - A small label: "ID=..., sum=..., avg=..."
      - A "Plot" button
    """
    def __init__(self, event_data, parent_gallery):
        super().__init__()
        self.event_data = event_data
        self.parent_gallery = parent_gallery  # to call parent's "plot_event" method

        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5,5,5,5)
        layout.setSpacing(3)

        # Load patch + create thumbnail
        patch = load_patch(event_data["patch_path"])
        self.thumb_pixmap = patch_to_thumbnail(patch, thumb_size=64)
        self.label_img = QLabel()
        self.label_img.setPixmap(self.thumb_pixmap)
        layout.addWidget(self.label_img, alignment=Qt.AlignCenter)

        # Label with ID, sum, avg
        eid   = event_data["event_id"]
        s_int = event_data["sum_intensity"]
        a_int = event_data["avg_intensity"]
        lbl_text = f"ID={eid}\nSum={s_int:.1f}\nAvg={a_int:.1f}"
        self.label_info = QLabel(lbl_text)
        self.label_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_info)

        # Plot button
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.on_plot_clicked)
        layout.addWidget(self.btn_plot, alignment=Qt.AlignCenter)

    def on_plot_clicked(self):
        """
        Calls parent's plot_event with this event's data.
        """
        self.parent_gallery.plot_event(self.event_data)

# ------------------ Main Gallery Window ------------------

class EventGalleryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Gallery (Grid View)")
        self.resize(1000, 600)

        # We'll store a list of events
        self.events = []
        self.csv_path = ""
        self.patches_dir = ""

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # top bar
        top_layout = QHBoxLayout()
        self.btn_load_csv = QPushButton("Load CSV")
        self.btn_load_csv.clicked.connect(self.on_load_csv)
        top_layout.addWidget(self.btn_load_csv)

        self.btn_load_patches = QPushButton("Load Patches Folder")
        self.btn_load_patches.clicked.connect(self.on_load_patches)
        top_layout.addWidget(self.btn_load_patches)

        self.btn_parse = QPushButton("Parse Events")
        self.btn_parse.clicked.connect(self.on_parse_events)
        top_layout.addWidget(self.btn_parse)

        # global style combo
        self.combo_style = QComboBox()
        self.combo_style.addItems(["2D Gray", "3D Surface", "3D Wireframe", "3D Scatter", "2D Contour"])
        top_layout.addWidget(QLabel("Plot Style:"))
        top_layout.addWidget(self.combo_style)

        main_layout.addLayout(top_layout)

        # scroll area for the grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        # a container widget in the scroll area
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(10)
        self.gallery_layout.setContentsMargins(10,10,10,10)

        self.scroll_area.setWidget(self.gallery_widget)

    def on_load_csv(self):
        """
        Select CSV file
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.csv_path = path

    def on_load_patches(self):
        """
        Select directory with .npy patches
        """
        folder = QFileDialog.getExistingDirectory(self, "Select patches folder")
        if folder:
            self.patches_dir = folder

    def on_parse_events(self):
        """
        Load events from CSV + patches, then build the gallery grid.
        """
        if not self.csv_path or not self.patches_dir:
            print("CSV or patches folder not selected.")
            return
        self.events = load_events(self.csv_path, self.patches_dir)
        print(f"Loaded {len(self.events)} events. Building gallery...")

        # clear existing items in the grid
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # put each event as a thumbnail in a grid
        cols = 5  # number of columns in the gallery
        row = 0
        col = 0
        for ev in self.events:
            thumb = EventThumbWidget(ev, self)
            self.gallery_layout.addWidget(thumb, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1

    def plot_event(self, event_data):
        """
        Called by each thumbnail's "Plot" button -> load patch, do plot with current style
        """
        patch = load_patch(event_data["patch_path"])
        style = self.combo_style.currentText()
        title = f"Event {event_data['event_id']} (Sum={event_data['sum_intensity']:.1f}, Avg={event_data['avg_intensity']:.1f})"
        plot_patch(patch, style=style, title=title)

# -------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    window = EventGalleryWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
