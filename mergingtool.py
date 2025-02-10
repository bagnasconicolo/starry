#!/usr/bin/env python3
"""
Merge Multiple Runs into One (GUI)

Allows you to:
  1) Add multiple runs, each with a CSV (event_log_runX.csv) + patches_runX folder.
  2) See them listed in a table.
  3) "Combine & Save" => produce a single merged CSV (event_log_merged.csv)
     plus a patches_merged/ folder containing all .npy patches renamed consistently.

Requirements:
    pip install pyqt5 opencv-python numpy

Usage:
  - python merge_runs_gui.py
  - A window appears with "Add Run," "Remove Selected," and "Combine & Save" buttons.
  - Click "Add Run" to select the CSV and then its patches folder.
  - Table updates with the number of events.
  - After adding all desired runs, click "Combine & Save" -> pick an output folder.
  - The script copies all patches to patches_merged/ and writes event_log_merged.csv there.

If you see no window at all, ensure you are running in a normal Python GUI environment
and have PyQt5 installed. Otherwise, the script will quietly exit.
"""

import sys
import os
import csv
import shutil

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QMessageBox, QTableWidget
)
from PyQt5.QtCore import Qt

class RunInfo:
    """Simple container for one run's metadata."""
    def __init__(self, csv_path, patches_dir):
        self.csv_path    = csv_path
        self.patches_dir = patches_dir
        self.events      = []  # each event is a dict

def load_events_from_csv(csv_path, patches_dir):
    """
    Read CSV, store each event as a dict, plus a key "source_patches_dir" for copying patches.
    """
    events = []
    if not os.path.exists(csv_path):
        return events
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = dict(row)
            e["source_patches_dir"] = patches_dir
            events.append(e)
    return events

class MergeRunsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Merge Multiple Runs into One")
        self.resize(900, 500)

        self.runs = []  # list of RunInfo

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # A welcome label at the top
        self.label_info = QLabel(
            "Welcome! This tool merges multiple runs (CSV+patches) into one.\n"
            "Click 'Add Run' to select a CSV file and its patches folder.\n"
            "They will appear in the table below. Then press 'Combine & Save' to produce a single merged run."
        )
        self.label_info.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.label_info)

        # Top Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_run = QPushButton("Add Run")
        self.btn_add_run.clicked.connect(self.on_add_run)
        btn_layout.addWidget(self.btn_add_run)

        self.btn_remove_run = QPushButton("Remove Selected")
        self.btn_remove_run.clicked.connect(self.on_remove_run)
        btn_layout.addWidget(self.btn_remove_run)

        self.btn_combine = QPushButton("Combine & Save")
        self.btn_combine.clicked.connect(self.on_combine)
        btn_layout.addWidget(self.btn_combine)

        main_layout.addLayout(btn_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["CSV Path", "Patches Folder", "Num Events"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        main_layout.addWidget(self.table)

    def on_add_run(self):
        """
        Let user pick a CSV file, then a patches folder, load events.
        """
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not csv_path:
            return
        patches_dir = QFileDialog.getExistingDirectory(self, "Select patches folder")
        if not patches_dir:
            return

        # load events
        evs = load_events_from_csv(csv_path, patches_dir)
        if not evs:
            QMessageBox.warning(self, "No Events", f"No events found in {csv_path}")
            return

        runinfo = RunInfo(csv_path, patches_dir)
        runinfo.events = evs
        self.runs.append(runinfo)
        self.update_table()

    def update_table(self):
        """
        Refresh the QTableWidget to show each run's CSV, patch folder, and # events.
        """
        self.table.setRowCount(len(self.runs))
        for row_idx, runinfo in enumerate(self.runs):
            item_csv = QTableWidgetItem(runinfo.csv_path)
            self.table.setItem(row_idx, 0, item_csv)

            item_patches = QTableWidgetItem(runinfo.patches_dir)
            self.table.setItem(row_idx, 1, item_patches)

            item_num = QTableWidgetItem(str(len(runinfo.events)))
            self.table.setItem(row_idx, 2, item_num)

    def on_remove_run(self):
        """
        Remove the run that is currently selected in the table.
        """
        sel_row = self.table.currentRow()
        if sel_row < 0 or sel_row >= len(self.runs):
            return
        self.runs.pop(sel_row)
        self.update_table()

    def on_combine(self):
        """
        Merge the runs => single run. Let user pick output folder => produce:
          event_log_merged.csv
          patches_merged/
        """
        if not self.runs:
            QMessageBox.warning(self, "No Runs", "No runs to combine.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder for Merged Run")
        if not out_dir:
            return

        merged_csv_path = os.path.join(out_dir, "event_log_merged.csv")
        merged_patches_dir = os.path.join(out_dir, "patches_merged")
        os.makedirs(merged_patches_dir, exist_ok=True)

        # gather all events
        all_events = []
        for runinfo in self.runs:
            all_events.extend(runinfo.events)
        if not all_events:
            QMessageBox.warning(self, "No Events", "All runs appear empty.")
            return

        csv_columns = [
            "event_id","frame_index","rmin","cmin","rmax","cmax",
            "sum_intensity","max_intensity","cluster_size",
            "avg_intensity","patch_filename"
        ]

        event_counter = 0
        with open(merged_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()

            for evt in all_events:
                event_counter += 1
                new_id = event_counter

                old_patch_fname = evt.get("patch_filename","")
                source_dir = evt.get("source_patches_dir","")
                old_patch_path = os.path.join(source_dir, old_patch_fname)
                new_patch_fname = f"evt_{new_id}.npy"
                new_patch_path = os.path.join(merged_patches_dir, new_patch_fname)

                # copy patch if it exists
                if os.path.exists(old_patch_path):
                    shutil.copy2(old_patch_path, new_patch_path)

                row_dict = {}
                row_dict["event_id"]       = str(new_id)
                row_dict["frame_index"]    = evt.get("frame_index","0")
                row_dict["rmin"]           = evt.get("rmin","0")
                row_dict["cmin"]           = evt.get("cmin","0")
                row_dict["rmax"]           = evt.get("rmax","0")
                row_dict["cmax"]           = evt.get("cmax","0")
                row_dict["sum_intensity"]  = evt.get("sum_intensity","0")
                row_dict["max_intensity"]  = evt.get("max_intensity","0")
                row_dict["cluster_size"]   = evt.get("cluster_size","0")
                row_dict["avg_intensity"]  = evt.get("avg_intensity","0")
                row_dict["patch_filename"] = new_patch_fname

                writer.writerow(row_dict)

        QMessageBox.information(self, "Done",
            f"Merged {event_counter} events into:\n{merged_csv_path}\nand {merged_patches_dir}")

def main():
    app = QApplication(sys.argv)
    window = MergeRunsWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
