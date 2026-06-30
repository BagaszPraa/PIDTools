#!/usr/bin/env python3
"""
ArduPilot .bin Log Viewer
==========================
Aplikasi GUI sederhana berbasis PySide6 untuk membaca file log ArduPilot
(.bin / .log) dan menampilkan plot dari pesan-pesan (messages) di
dalamnya, contohnya ATT (Attitude), GPS, BARO, IMU, BAT, dll.

Cara pakai:
    1. Install dependensi:
       pip install PySide6 pymavlink matplotlib numpy

    2. Jalankan:
       python ardupilot_log_viewer.py

Fitur:
    - Buka file .bin / .log ArduPilot (format dataflash)
    - Daftar semua tipe pesan (message type) yang ada di log
    - Daftar field/kolom numerik dari tipe pesan yang dipilih
    - Plot satu atau beberapa field sekaligus (overlay) terhadap waktu
    - Zoom, pan, dan save gambar plot (toolbar matplotlib bawaan)
    - Export data pesan yang sedang ditampilkan ke CSV
"""

import sys
import os
import csv
from collections import defaultdict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QFileDialog,
    QSplitter, QStatusBar, QMessageBox, QAbstractItemView, QLineEdit,
    QToolBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


# --------------------------------------------------------------------------
# Worker thread: parsing file .bin agar GUI tidak freeze
# --------------------------------------------------------------------------
class LogLoaderThread(QThread):
    progress = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        if mavutil is None:
            self.failed.emit(
                "Modul 'pymavlink' tidak ditemukan.\n"
                "Install dengan: pip install pymavlink"
            )
            return

        try:
            self.progress.emit("Membuka file log...")
            mlog = mavutil.mavlink_connection(self.filepath, dialect="ardupilotmega")

            # data[msg_type][field_name] = list nilai
            # juga simpan timestamp per msg_type di key '__time__'
            data = defaultdict(lambda: defaultdict(list))

            count = 0
            while True:
                msg = mlog.recv_match(blocking=False)
                if msg is None:
                    break

                msg_type = msg.get_type()
                if msg_type in ("BAD_DATA", "UNKNOWN"):
                    continue

                msg_dict = msg.to_dict()
                msg_dict.pop("mavpackettype", None)

                # Ambil timestamp: pakai field TimeUS / time_boot_ms kalau ada,
                # kalau tidak pakai urutan index
                t = None
                for tkey in ("TimeUS", "time_boot_ms", "time_usec"):
                    if tkey in msg_dict:
                        t = msg_dict[tkey]
                        break
                if t is None:
                    t = count

                bucket = data[msg_type]
                bucket["__time__"].append(t)
                for k, v in msg_dict.items():
                    if isinstance(v, (int, float)):
                        bucket[k].append(v)

                count += 1
                if count % 20000 == 0:
                    self.progress.emit(f"Membaca pesan... ({count} pesan)")

            self.progress.emit(f"Selesai membaca {count} pesan.")
            self.finished_ok.emit(dict(data))

        except Exception as e:
            self.failed.emit(f"Gagal membaca file log:\n{e}")


# --------------------------------------------------------------------------
# Main Window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArduPilot .bin Log Viewer")
        self.resize(1200, 750)

        self.log_data = {}   # msg_type -> {field: [values]}
        self.current_filepath = None
        self.loader_thread = None

        self._build_ui()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("Buka Log (.bin/.log)", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        export_action = QAction("Export CSV", self)
        export_action.triggered.connect(self.export_csv)
        toolbar.addAction(export_action)

        clear_action = QAction("Bersihkan Plot", self)
        clear_action.triggered.connect(self.clear_plot)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        self.box_zoom_action = QAction("Box Zoom", self)
        self.box_zoom_action.setCheckable(True)
        self.box_zoom_action.toggled.connect(self.toggle_box_zoom)
        toolbar.addAction(self.box_zoom_action)

        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.reset_zoom)
        toolbar.addAction(reset_zoom_action)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Panel kiri: daftar message type & field -----------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("Tipe Pesan (Message Type):"))
        self.msg_type_list = QListWidget()
        self.msg_type_list.itemSelectionChanged.connect(self.on_msg_type_selected)
        left_layout.addWidget(self.msg_type_list)

        left_layout.addWidget(QLabel("Filter Field:"))
        self.field_filter = QLineEdit()
        self.field_filter.setPlaceholderText("ketik untuk filter field...")
        self.field_filter.textChanged.connect(self.refresh_field_list)
        left_layout.addWidget(self.field_filter)

        left_layout.addWidget(QLabel("Field (pilih satu/lebih):"))
        self.field_list = QListWidget()
        self.field_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self.field_list)

        self.plot_btn = QPushButton("Plot Field Terpilih")
        self.plot_btn.clicked.connect(self.plot_selected_fields)
        left_layout.addWidget(self.plot_btn)

        splitter.addWidget(left_panel)

        # --- Panel kanan: canvas matplotlib ----------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure = Figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.nav_toolbar)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # Rectangle selector untuk box-zoom (mirip UAVLogViewer)
        self.rect_selector = RectangleSelector(
            self.ax,
            self.on_box_zoom_select,
            useblit=True,
            button=[1],  # klik kiri mouse
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=False,
            props=dict(facecolor="orange", edgecolor="orange", alpha=0.2, fill=True),
        )
        self.rect_selector.set_active(False)
        self._default_xlim = None
        self._default_ylim = None

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Siap. Buka file .bin untuk mulai.")

    # ------------------------------------------------------------ Logic --
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Buka File Log ArduPilot", "",
            "ArduPilot Log (*.bin *.BIN *.log *.LOG);;Semua File (*)"
        )
        if not filepath:
            return

        self.current_filepath = filepath
        self.status.showMessage(f"Membaca {os.path.basename(filepath)} ...")
        self.plot_btn.setEnabled(False)

        self.loader_thread = LogLoaderThread(filepath)
        self.loader_thread.progress.connect(self.status.showMessage)
        self.loader_thread.finished_ok.connect(self.on_log_loaded)
        self.loader_thread.failed.connect(self.on_log_failed)
        self.loader_thread.start()

    def on_log_loaded(self, data):
        self.log_data = data
        self.plot_btn.setEnabled(True)

        self.msg_type_list.clear()
        for msg_type in sorted(data.keys()):
            n_fields = len(data[msg_type]) - 1  # minus __time__
            n_samples = len(data[msg_type].get("__time__", []))
            item = QListWidgetItem(f"{msg_type}  ({n_fields} field, {n_samples} sampel)")
            item.setData(Qt.UserRole, msg_type)
            self.msg_type_list.addItem(item)

        self.status.showMessage(
            f"Berhasil memuat {os.path.basename(self.current_filepath)} — "
            f"{len(data)} tipe pesan ditemukan."
        )

    def on_log_failed(self, error_msg):
        self.plot_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)
        self.status.showMessage("Gagal membaca file.")

    def on_msg_type_selected(self):
        self.refresh_field_list()

    def refresh_field_list(self):
        self.field_list.clear()
        items = self.msg_type_list.selectedItems()
        if not items:
            return
        msg_type = items[0].data(Qt.UserRole)
        fields = [f for f in self.log_data.get(msg_type, {}).keys() if f != "__time__"]

        keyword = self.field_filter.text().strip().lower()
        if keyword:
            fields = [f for f in fields if keyword in f.lower()]

        for f in sorted(fields):
            self.field_list.addItem(f)

    def plot_selected_fields(self):
        msg_items = self.msg_type_list.selectedItems()
        field_items = self.field_list.selectedItems()

        if not msg_items or not field_items:
            QMessageBox.information(
                self, "Info",
                "Pilih satu tipe pesan dan minimal satu field terlebih dahulu."
            )
            return

        msg_type = msg_items[0].data(Qt.UserRole)
        bucket = self.log_data.get(msg_type, {})
        t = bucket.get("__time__", [])

        self.ax.clear()
        for item in field_items:
            field_name = item.text()
            values = bucket.get(field_name, [])
            n = min(len(t), len(values))
            if n == 0:
                continue
            self.ax.plot(t[:n], values[:n], label=field_name, linewidth=1.0)

        self.ax.set_title(f"{msg_type}")
        self.ax.set_xlabel("Waktu (raw timestamp dari log)")
        self.ax.set_ylabel("Nilai")
        self.ax.legend(loc="best", fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

        # Simpan batas tampilan default agar bisa direset setelah box-zoom
        self._default_xlim = self.ax.get_xlim()
        self._default_ylim = self.ax.get_ylim()

        # RectangleSelector terikat ke Axes lama; buat ulang agar tetap aktif
        # di Axes yang baru saja di-clear/redraw.
        was_active = self.box_zoom_action.isChecked()
        self.rect_selector = RectangleSelector(
            self.ax,
            self.on_box_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=False,
            props=dict(facecolor="orange", edgecolor="orange", alpha=0.2, fill=True),
        )
        self.rect_selector.set_active(was_active)

        self.status.showMessage(
            f"Plot {len(field_items)} field dari pesan {msg_type}."
        )

    def clear_plot(self):
        self.ax.clear()
        self.canvas.draw()

    def toggle_box_zoom(self, checked):
        # Matikan tool pan/zoom bawaan matplotlib agar tidak bentrok
        if checked:
            mode = str(self.nav_toolbar.mode).upper()
            if "PAN" in mode:
                self.nav_toolbar.pan()
            elif "ZOOM" in mode:
                self.nav_toolbar.zoom()

        self.rect_selector.set_active(checked)
        if checked:
            self.status.showMessage(
                "Mode Box Zoom aktif: klik & seret untuk membuat kotak area zoom."
            )
        else:
            self.status.showMessage("Mode Box Zoom nonaktif.")

    def on_box_zoom_select(self, eclick, erelease):
        """Dipanggil saat user selesai menyeret kotak seleksi (mirip UAVLogViewer)."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        if x1 == x2 or y1 == y2:
            return

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
        self.status.showMessage(
            f"Zoom ke area X: [{xmin:.3f}, {xmax:.3f}]  Y: [{ymin:.3f}, {ymax:.3f}]"
        )

    def reset_zoom(self):
        if self._default_xlim is not None and self._default_ylim is not None:
            self.ax.set_xlim(self._default_xlim)
            self.ax.set_ylim(self._default_ylim)
            self.canvas.draw_idle()
            self.status.showMessage("Zoom direset ke tampilan awal.")

    def export_csv(self):
        msg_items = self.msg_type_list.selectedItems()
        if not msg_items:
            QMessageBox.information(self, "Info", "Pilih tipe pesan terlebih dahulu.")
            return

        msg_type = msg_items[0].data(Qt.UserRole)
        bucket = self.log_data.get(msg_type, {})
        if not bucket:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan sebagai CSV", f"{msg_type}.csv", "CSV Files (*.csv)"
        )
        if not save_path:
            return

        fieldnames = ["time"] + [k for k in bucket.keys() if k != "__time__"]
        rows = len(bucket["__time__"])

        try:
            with open(save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                for i in range(rows):
                    row = [bucket["__time__"][i]]
                    for k in fieldnames[1:]:
                        vals = bucket.get(k, [])
                        row.append(vals[i] if i < len(vals) else "")
                    writer.writerow(row)
            self.status.showMessage(f"Data {msg_type} disimpan ke {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menyimpan CSV:\n{e}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()