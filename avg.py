#!/usr/bin/env python3
"""
Expanded Aggregator with Margin, Peak Alignment, Normalization,
Variance Map, Max Projection, and Dot Overlay of Event Centers

Features:
  1) Reads a CSV (event_log_runX.csv) + patch folder (patches_runX/).
  2) Filters events by min_avg_int. Loads each .npy patch, applies optional:
       - alignment ("peak" or "center")
       - normalization ("none","peak","sum")
  3) Computes aggregator average (2D array), plus:
       - Pixelwise variance map
       - Max-projection overlay (all events stacked in aggregator space)
  4) Calculates "active-pixel radius" ignoring margins.
  5) Visualizes:
       (a) 2D/3D aggregator average
       (b) 2D contour aggregator
       (c) bounding-box overlay if you want
       (d) **Dot overlay** (each event's bounding-box center is a small dot)
       (e) Max-projection overlay
       (f) Variance map
       (g) Distributions: sum_int, avg_int, active radius

Requires:
    pip install opencv-python numpy matplotlib
    (plus tkinter for file dialogs)
"""

import os
import sys
import math
import csv
import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")  # non-blocking windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------- DEFAULTS ---------------------------
MIN_AVG_INT_DEFAULT = 100.0
AGG_DIM_DEFAULT     = 80

ALIGN_MODES = ["peak","center"]
NORM_MODES  = ["none","peak","sum"]

# For the "active radius" ignoring margin
ACTIVE_FRAC         = 0.2  # threshold fraction of patch.max()
# ---------------------------------------------------------------

def compute_active_radius(aligned_patch, frac=ACTIVE_FRAC):
    """
    1) threshold = frac * patch.max()
    2) bounding box of those "active" pixels => diagonal => radius = half diagonal
    ignoring blank margins.
    """
    mx = aligned_patch.max()
    if mx<=0:
        return 0.0
    thr = frac*mx
    mask= (aligned_patch>thr)
    coords= np.column_stack(np.where(mask))
    if coords.size<2:
        return 0.0
    rmin,cmin = coords.min(axis=0)
    rmax,cmax = coords.max(axis=0)
    h = (rmax-rmin+1)
    w = (cmax-cmin+1)
    diag= math.sqrt(h*h + w*w)
    return diag*0.5

def align_patch_by_peak(patch, aggregator_dim):
    ph, pw = patch.shape
    if ph>aggregator_dim or pw>aggregator_dim:
        return None
    # find brightest pixel
    pr, pc = np.unravel_index(np.argmax(patch), patch.shape)
    center_r= aggregator_dim//2
    center_c= aggregator_dim//2
    shift_r = center_r - pr
    shift_c = center_c - pc
    sr= shift_r
    sc= shift_c
    er= sr+ ph
    ec= sc+ pw
    if sr<0 or sc<0 or er>aggregator_dim or ec>aggregator_dim:
        return None
    buf= np.zeros((aggregator_dim, aggregator_dim), dtype=patch.dtype)
    buf[sr:er, sc:ec]= patch
    return buf

def align_patch_center(patch, aggregator_dim):
    ph, pw= patch.shape
    if ph>aggregator_dim or pw>aggregator_dim:
        return None
    buf= np.zeros((aggregator_dim, aggregator_dim), dtype=patch.dtype)
    center_r= aggregator_dim//2
    center_c= aggregator_dim//2
    patch_cr= ph//2
    patch_cc= pw//2
    sr= center_r - patch_cr
    sc= center_c - patch_cc
    er= sr+ ph
    ec= sc+ pw
    if sr<0 or sc<0 or er>aggregator_dim or ec>aggregator_dim:
        return None
    buf[sr:er, sc:ec]= patch
    return buf

def normalize_patch_none(patch):
    return patch

def normalize_patch_peak(patch):
    mx = patch.max()
    if mx>0:
        return patch/mx
    return patch

def normalize_patch_sum(patch):
    s= patch.sum()
    if s>0:
        return patch/s
    return patch

def main():
    root= tk.Tk()
    root.withdraw()

    # 1) CSV
    csv_path= filedialog.askopenfilename(
        title="Select CSV",
        filetypes=[("CSV","*.csv"),("All","*.*")]
    )
    if not csv_path:
        messagebox.showinfo("Cancelled","No CSV selected.")
        sys.exit(0)

    # 2) patches folder
    patch_dir= filedialog.askdirectory(title="Select patches folder")
    if not patch_dir:
        messagebox.showinfo("Cancelled","No folder selected.")
        sys.exit(0)

    # aggregator dimension
    user_agg= simpledialog.askstring("Aggregator Dim",
        f"Default={AGG_DIM_DEFAULT}")
    if user_agg:
        try:
            aggregator_dim= int(user_agg)
        except:
            aggregator_dim= AGG_DIM_DEFAULT
    else:
        aggregator_dim= AGG_DIM_DEFAULT

    # min avg int
    user_min_avg= simpledialog.askstring("Min Avg Intensity",
        f"Default={MIN_AVG_INT_DEFAULT}")
    if user_min_avg:
        try:
            min_avg_int= float(user_min_avg)
        except:
            min_avg_int= MIN_AVG_INT_DEFAULT
    else:
        min_avg_int= MIN_AVG_INT_DEFAULT

    # alignment
    align_mode= simpledialog.askstring("Alignment Mode",
        f"Options: {ALIGN_MODES} (default={ALIGN_MODES[0]})")
    if align_mode not in ALIGN_MODES:
        align_mode= ALIGN_MODES[0]

    # normalization
    norm_mode= simpledialog.askstring("Normalization Mode",
        f"Options: {NORM_MODES} (default={NORM_MODES[0]})")
    if norm_mode not in NORM_MODES:
        norm_mode= NORM_MODES[0]

    root.destroy()

    if not os.path.isfile(csv_path):
        print("Invalid CSV path.")
        sys.exit(1)
    if not os.path.isdir(patch_dir):
        print("Invalid patches folder.")
        sys.exit(1)

    # define alignment function
    if align_mode=="peak":
        align_func= align_patch_by_peak
    else:
        align_func= align_patch_center

    # define normalization
    if norm_mode=="peak":
        norm_func= normalize_patch_peak
    elif norm_mode=="sum":
        norm_func= normalize_patch_sum
    else:
        norm_func= normalize_patch_none

    # aggregator arrays
    sum_array   = np.zeros((aggregator_dim, aggregator_dim), dtype=np.float32)
    sqsum_array = np.zeros((aggregator_dim, aggregator_dim), dtype=np.float32)
    max_array   = np.zeros((aggregator_dim, aggregator_dim), dtype=np.float32)

    aggregator_sum= []
    aggregator_avg= []
    aggregator_rad= []
    aggregator_bboxes= []  # store bounding boxes to do dot overlay
    aggregator_patches= []
    n_ev=0

    max_rmax=0
    max_cmax=0

    # parse CSV
    with open(csv_path,'r', newline='') as f:
        rd= csv.DictReader(f)
        for row in rd:
            try:
                avg_i= float(row["avg_intensity"])
                if avg_i< min_avg_int:
                    continue
                sum_i= float(row["sum_intensity"])
                rmin= int(row["rmin"])
                cmin= int(row["cmin"])
                rmax= int(row["rmax"])
                cmax= int(row["cmax"])
                patch_fname= row["patch_filename"]
            except:
                continue

            patch_path= os.path.join(patch_dir, patch_fname)
            if not os.path.exists(patch_path):
                continue

            patch = np.load(patch_path).astype(np.float32)
            patch = norm_func(patch)
            aligned= align_func(patch, aggregator_dim)
            if aligned is None:
                continue

            # compute active radius
            rad = compute_active_radius(aligned)
            aggregator_rad.append(rad)
            aggregator_sum.append(sum_i)
            aggregator_avg.append(avg_i)

            aggregator_bboxes.append((rmin,cmin,rmax,cmax))

            if rmax> max_rmax: max_rmax= rmax
            if cmax> max_cmax: max_cmax= cmax

            # aggregator stats
            sum_array  += aligned
            sqsum_array+= aligned*aligned
            max_array  = np.maximum(max_array, aligned)
            aggregator_patches.append(aligned)
            n_ev+=1

    if n_ev<1:
        print(f"No events pass filter (avg_int >= {min_avg_int}). Exiting.")
        sys.exit(0)

    avg_array= sum_array / n_ev
    mean_sq  = sqsum_array / n_ev
    var_array= mean_sq - (avg_array*avg_array)

    plt.ion()

    # 1) aggregator 2D heatmap
    fig1= plt.figure()
    plt.title(f"Aggregator Average (n={n_ev})\nAlign={align_mode}, Norm={norm_mode}")
    plt.imshow(avg_array, cmap='gray', origin='upper')
    plt.colorbar(label='Intensity')
    plt.show(block=False)

    # 2) aggregator 3D surface
    rows, cols= avg_array.shape
    X, Y= np.meshgrid(np.arange(cols), np.arange(rows))
    fig2= plt.figure()
    ax2= fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, avg_array, cmap='jet', linewidth=0, antialiased=False)
    ax2.set_title("3D Surface - Aggregator Average")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Intensity")
    plt.tight_layout()
    plt.show(block=False)

    # 3) aggregator 3D scatter
    fig3= plt.figure()
    ax3= fig3.add_subplot(111, projection='3d')
    xf= X.flatten()
    yf= Y.flatten()
    zf= avg_array.flatten()
    sc= ax3.scatter(xf, yf, zf, c=zf, cmap='jet', alpha=0.7, s=10)
    ax3.set_title("3D Scatter - Aggregator Average")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Intensity")
    fig3.colorbar(sc,label='Intensity')
    plt.show(block=False)

    # 4) aggregator 2D contour
    fig4= plt.figure()
    plt.title("2D Contour - Aggregator Average")
    cset= plt.contour(avg_array, levels=15, cmap='jet')
    plt.clabel(cset, inline=True, fontsize=8)
    plt.colorbar()
    plt.show(block=False)

    # 5) bounding box overlay (optional - rectangles)
    max_rmax+=2
    max_cmax+=2
    if max_rmax<1: max_rmax=1
    if max_cmax<1: max_cmax=1
    overlay_bbox= np.zeros((max_rmax, max_cmax,3), dtype=np.uint8)
    for (rm,cm,rx,cx) in aggregator_bboxes:
        if rx<max_rmax and cx<max_cmax:
            cv2.rectangle(overlay_bbox, (cm,rm),(cx,rx),(0,0,255),1)
    ov_bbox_rgb= cv2.cvtColor(overlay_bbox, cv2.COLOR_BGR2RGB)
    fig5= plt.figure()
    plt.title(f"BBox Overlay (n={n_ev}) - with rectangles")
    plt.imshow(ov_bbox_rgb)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=False)

    # ================ NEW: Dot Overlay ================
    # We'll do the same final shape, but place a small dot at the bounding box center.
    overlay_dot= np.zeros((max_rmax, max_cmax,3), dtype=np.uint8)
    for (rm,cm,rx,cx) in aggregator_bboxes:
        if rx<max_rmax and cx<max_cmax:
            # bounding box center
            cr = (rm + rx)//2
            cc = (cm + cx)//2
            cv2.circle(overlay_dot, (cc, cr), 3, (255,255,255), -1)  #  dot
    ov_dot_rgb= cv2.cvtColor(overlay_dot, cv2.COLOR_BGR2RGB)
    figD= plt.figure()
    plt.title(f"Dot Overlay of {n_ev} events (BBox center in green)")
    plt.imshow(ov_dot_rgb)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=False)

    # A) Max-projection overlay
    figA= plt.figure()
    plt.title("All-Events Overlay (Max Projection in aggregator space)")
    plt.imshow(max_array, cmap='gray', origin='upper')
    plt.colorbar(label='Intensity')
    plt.show(block=False)

    # B) Variance map
    var_array= var_array.clip(min=0)  # just in case
    figB= plt.figure()
    plt.title("Variance Map (pixelwise across aggregator patches)")
    plt.imshow(var_array, cmap='gray', origin='upper')
    plt.colorbar(label='Variance')
    plt.show(block=False)

    # distributions: sum_int, avg_int, active radius
    sums_arr= []
    avgs_arr= []
    rads_arr= []
    for i in range(n_ev):
        sums_arr.append( aggregator_sum[i] )
        avgs_arr.append( aggregator_avg[i] )
        rads_arr.append( aggregator_rad[i] )
    sums_arr= np.array(sums_arr, dtype=np.float32)
    avgs_arr= np.array(avgs_arr, dtype=np.float32)
    rads_arr= np.array(rads_arr, dtype=np.float32)

    fig_dist, (axA, axB, axC)= plt.subplots(1,3, figsize=(15,4))

    # sum
    hs, es= np.histogram(sums_arr,bins=20)
    hs_pct= (hs/ n_ev)*100
    axA.bar(es[:-1], hs_pct, width=(es[1]-es[0]), color='blue', alpha=0.7)
    axA.set_title("Sum Int Dist")
    axA.set_xlabel("sum_intensity")
    axA.set_ylabel("Freq (%)")

    # avg
    ha, ea= np.histogram(avgs_arr,bins=20)
    ha_pct= (ha/n_ev)*100
    axB.bar(ea[:-1], ha_pct, width=(ea[1]-ea[0]), color='magenta', alpha=0.7)
    axB.set_title("Avg Int Dist")
    axB.set_xlabel("avg_intensity")
    axB.set_ylabel("Freq (%)")

    # active radius
    hr, er= np.histogram(rads_arr,bins=20)
    hr_pct= (hr/n_ev)*100
    axC.bar(er[:-1], hr_pct, width=(er[1]-er[0]), color='green', alpha=0.7)
    axC.set_title("Active Radius Dist")
    axC.set_xlabel("radius (pixels)")
    axC.set_ylabel("Freq (%)")

    plt.tight_layout()
    plt.show(block=False)

    print("\nAll aggregator plots open. Press ENTER in terminal to close.")
    input()
    plt.close('all')

if __name__=="__main__":
    main()
