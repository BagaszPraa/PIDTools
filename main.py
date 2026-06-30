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
    - Plot satu atau beberapa field sekaligus (overlay) terhadap waktu (ms)
    - Pengkali/skala nilai per field sebelum di-plot
    - Zoom box (drag kotak), reset zoom, pan, dan save gambar plot
    - Hover di garis plot untuk menampilkan nilai (titik & tooltip)
    - Export data pesan yang sedang ditampilkan ke CSV
    - Halaman PID Tune: plot Target vs Actual dan komponen P/I/D/FF dari
      pesan PIDR/PIDP/PIDY, plus analisa & saran penyesuaian gain berbasis
      heuristik dari data log
    - Halaman Live Tuning: terhubung langsung ke drone via telemetri
      MAVLink (UDP/TCP/Serial), plot PID real-time (PID_TUNING), serta
      baca & tulis parameter PID (mis. ATC_RAT_RLL_P) saat drone terbang
"""

import sys
import os
import csv
import time
import queue
from collections import defaultdict, deque

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QFileDialog,
    QSplitter, QStatusBar, QMessageBox, QAbstractItemView, QLineEdit,
    QToolBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedWidget, QComboBox, QTextEdit, QDoubleSpinBox, QGroupBox,
    QFormLayout, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QActionGroup

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

            # Simpan nilai parameter terakhir (mis. ATC_RAT_RLL_P) dari pesan
            # PARM, dipakai untuk fitur analisa/saran PID.
            param_values = {}

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

                if msg_type == "PARM":
                    pname = msg_dict.get("Name")
                    pval = msg_dict.get("Value")
                    if pname is not None and pval is not None:
                        param_values[pname] = pval

                # Ambil timestamp dan konversi ke milidetik (ms) secara konsisten.
                # - TimeUS / time_usec : satuan mikrodetik -> dibagi 1000
                # - time_boot_ms       : sudah dalam milidetik
                t = None
                for tkey in ("TimeUS", "time_usec"):
                    if tkey in msg_dict:
                        t = msg_dict[tkey] / 1000.0
                        break
                if t is None and "time_boot_ms" in msg_dict:
                    t = float(msg_dict["time_boot_ms"])
                if t is None:
                    t = float(count)

                bucket = data[msg_type]
                bucket["__time__"].append(t)
                for k, v in msg_dict.items():
                    if isinstance(v, (int, float)):
                        bucket[k].append(v)

                count += 1
                if count % 20000 == 0:
                    self.progress.emit(f"Membaca pesan... ({count} pesan)")

            result = dict(data)
            result["__PARAMS__"] = param_values

            self.progress.emit(f"Selesai membaca {count} pesan.")
            self.finished_ok.emit(result)

        except Exception as e:
            self.failed.emit(f"Gagal membaca file log:\n{e}")


# --------------------------------------------------------------------------
# Live Telemetry Worker (untuk fitur Live Tuning)
# --------------------------------------------------------------------------
class MavlinkLiveWorker(QThread):
    """Mengelola koneksi MAVLink real-time ke drone (UDP/TCP/Serial) untuk
    fitur Live Tuning: membaca pesan PID_TUNING & HEARTBEAT, serta
    membaca/menulis parameter (mis. ATC_RAT_RLL_P) saat drone sedang terbang.

    Semua komunikasi MAVLink (baca/tulis) dijalankan di thread ini agar
    tidak memblokir GUI dan agar koneksi soket hanya diakses dari satu thread.
    """

    connected = Signal(str)
    disconnected = Signal()
    error = Signal(str)
    heartbeat = Signal(dict)          # {"armed": bool, "mode": str}
    param_value = Signal(str, float)  # nama param, nilai
    pid_sample = Signal(str, float, float, float, float, float, float)
    # axis_name, t_relatif(s), desired, achieved, P, I, D

    # Mapping enum axis MAVLink PID_TUNING -> nama yang dipakai di GUI
    AXIS_MAP = {1: "Roll", 2: "Pitch", 3: "Yaw", 4: "AccZ", 5: "VelXY", 6: "VelZ"}

    def __init__(self, conn_str):
        super().__init__()
        self.conn_str = conn_str
        self.master = None
        self._running = False
        self._cmd_queue = queue.Queue()

    def request_param(self, name):
        """Minta nilai parameter terbaru dari vehicle (dipanggil dari GUI thread)."""
        self._cmd_queue.put(("READ", name, None))

    def set_param(self, name, value):
        """Kirim perubahan nilai parameter ke vehicle (dipanggil dari GUI thread)."""
        self._cmd_queue.put(("WRITE", name, value))

    def stop(self):
        self._running = False

    def run(self):
        if mavutil is None:
            self.error.emit(
                "Modul 'pymavlink' tidak ditemukan.\nInstall dengan: pip install pymavlink"
            )
            return

        try:
            self.master = mavutil.mavlink_connection(self.conn_str)
            self.master.wait_heartbeat(timeout=10)
        except Exception as e:
            self.error.emit(f"Gagal terhubung ke '{self.conn_str}':\n{e}")
            return

        self.connected.emit(self.conn_str)
        self._running = True

        # Minta vehicle mengirim semua stream telemetri (termasuk PID_TUNING bila
        # GCS_PID_MASK pada vehicle sudah mengaktifkan axis yang relevan)
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
            )
        except Exception:
            pass

        start_t = time.time()

        while self._running:
            # Proses perintah baca/tulis parameter yang antri dari GUI thread
            while not self._cmd_queue.empty():
                try:
                    action, name, value = self._cmd_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if action == "READ":
                        self.master.mav.param_request_read_send(
                            self.master.target_system, self.master.target_component,
                            name.encode("utf-8"), -1
                        )
                    elif action == "WRITE":
                        self.master.mav.param_set_send(
                            self.master.target_system, self.master.target_component,
                            name.encode("utf-8"), float(value),
                            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                        )
                except Exception as e:
                    self.error.emit(f"Gagal mengirim parameter {name}: {e}")

            try:
                msg = self.master.recv_match(blocking=True, timeout=0.3)
            except Exception as e:
                self.error.emit(f"Koneksi terputus: {e}")
                break

            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "HEARTBEAT":
                try:
                    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                except Exception:
                    armed = False
                try:
                    mode = mavutil.mode_string_v10(msg)
                except Exception:
                    mode = str(getattr(msg, "custom_mode", "?"))
                self.heartbeat.emit({"armed": armed, "mode": mode})

            elif mtype == "PARAM_VALUE":
                pname = msg.param_id
                if isinstance(pname, bytes):
                    pname = pname.decode("utf-8", errors="ignore")
                pname = pname.strip("\x00")
                self.param_value.emit(pname, float(msg.param_value))

            elif mtype == "PID_TUNING":
                axis_name = self.AXIS_MAP.get(msg.axis, str(msg.axis))
                t_rel = time.time() - start_t
                self.pid_sample.emit(
                    axis_name, t_rel,
                    float(msg.desired), float(msg.achieved),
                    float(msg.P), float(msg.I), float(msg.D)
                )

        try:
            if self.master is not None:
                self.master.close()
        except Exception:
            pass
        self.disconnected.emit()


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

        # State untuk fitur Live Tuning
        self.live_worker = None
        self.live_start_time = None
        self.live_buffers = {
            axis: {"t": deque(maxlen=1500), "des": deque(maxlen=1500), "act": deque(maxlen=1500)}
            for axis in ("Roll", "Pitch", "Yaw")
        }
        self.live_max_points = 1500
        self.live_param_labels = {}   # nama param -> QLabel nilai saat ini
        self.live_param_inputs = {}   # nama param -> QDoubleSpinBox nilai baru
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(250)
        self.live_timer.timeout.connect(self.update_live_plot)

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

        toolbar.addSeparator()

        # --- Toolbar navigasi halaman: Plot <-> PID Tune ---------------
        self.page_group = QActionGroup(self)
        self.page_group.setExclusive(True)

        self.page_plot_action = QAction("Plot", self)
        self.page_plot_action.setCheckable(True)
        self.page_plot_action.setChecked(True)
        self.page_plot_action.triggered.connect(lambda: self.switch_page(0))
        self.page_group.addAction(self.page_plot_action)
        toolbar.addAction(self.page_plot_action)

        self.page_pid_action = QAction("PID Tune", self)
        self.page_pid_action.setCheckable(True)
        self.page_pid_action.triggered.connect(lambda: self.switch_page(1))
        self.page_group.addAction(self.page_pid_action)
        toolbar.addAction(self.page_pid_action)

        self.page_live_action = QAction("Live Tuning", self)
        self.page_live_action.setCheckable(True)
        self.page_live_action.triggered.connect(lambda: self.switch_page(2))
        self.page_group.addAction(self.page_live_action)
        toolbar.addAction(self.page_live_action)

        # Central layout: QStackedWidget berisi halaman Plot dan PID Tune
        self.stacked_pages = QStackedWidget()
        self.setCentralWidget(self.stacked_pages)

        self.stacked_pages.addWidget(self._build_plot_page())
        self.stacked_pages.addWidget(self._build_pid_tune_page())
        self.stacked_pages.addWidget(self._build_live_tuning_page())

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Siap. Buka file .bin untuk mulai.")

    def switch_page(self, index):
        self.stacked_pages.setCurrentIndex(index)

    def _build_plot_page(self):
        central = QWidget()
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
        self.field_list.itemSelectionChanged.connect(self.refresh_scale_table)
        left_layout.addWidget(self.field_list)

        left_layout.addWidget(QLabel("Skala / Pengkali per Field:"))
        self.scale_table = QTableWidget(0, 2)
        self.scale_table.setHorizontalHeaderLabels(["Field", "Pengkali"])
        self.scale_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.scale_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.scale_table.verticalHeader().setVisible(False)
        self.scale_table.setMaximumHeight(160)
        left_layout.addWidget(self.scale_table)

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

        # --- Hover tooltip: tampilkan nilai saat cursor dekat dengan garis --
        self._plotted_lines = []  # list of dict: {line, x, y, label}
        self._hover_marker, = self.ax.plot(
            [], [], "o", color="red", markersize=6, zorder=10, visible=False
        )
        self._hover_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.85),
            arrowprops=dict(arrowstyle="->"),
            zorder=11,
            visible=False,
        )
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        return central

    def _build_pid_tune_page(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)

        # --- Panel kiri: kontrol PID Tune -----------------------------
        left_panel = QWidget()
        left_panel.setMaximumWidth(260)
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("Axis:"))
        self.pid_axis_combo = QComboBox()
        # Mapping nama axis -> prefix message dataflash ArduPilot (PIDR/PIDP/PIDY)
        self.pid_axis_map = {
            "Roll (PIDR)": "PIDR",
            "Pitch (PIDP)": "PIDP",
            "Yaw (PIDY)": "PIDY",
        }
        self.pid_axis_combo.addItems(list(self.pid_axis_map.keys()))
        left_layout.addWidget(self.pid_axis_combo)

        left_layout.addWidget(QLabel("Komponen (centang untuk plot):"))
        self.pid_component_list = QListWidget()
        self.pid_component_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self.pid_component_list)

        self.pid_plot_btn = QPushButton("Plot PID")
        self.pid_plot_btn.clicked.connect(self.plot_pid_tune)
        left_layout.addWidget(self.pid_plot_btn)

        self.pid_analyze_btn = QPushButton("Analisa && Sarankan PID")
        self.pid_analyze_btn.clicked.connect(self.analyze_pid_tuning)
        left_layout.addWidget(self.pid_analyze_btn)

        left_layout.addWidget(QLabel(
            "Catatan: Grafik atas menampilkan Target vs Actual,\n"
            "grafik bawah menampilkan komponen P/I/D/FF yang dipilih."
        ))
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # --- Panel kanan: dua subplot (Target/Actual & komponen PID) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.pid_figure = Figure(figsize=(8, 6))
        self.pid_ax_top = self.pid_figure.add_subplot(211)
        self.pid_ax_bottom = self.pid_figure.add_subplot(212, sharex=self.pid_ax_top)
        self.pid_canvas = FigureCanvas(self.pid_figure)
        self.pid_nav_toolbar = NavigationToolbar(self.pid_canvas, self)

        right_layout.addWidget(self.pid_nav_toolbar)
        right_layout.addWidget(self.pid_canvas)

        right_layout.addWidget(QLabel("Hasil Analisa & Saran PID:"))
        self.pid_analysis_output = QTextEdit()
        self.pid_analysis_output.setReadOnly(True)
        self.pid_analysis_output.setMaximumHeight(180)
        self.pid_analysis_output.setPlaceholderText(
            "Klik 'Plot PID' lalu 'Analisa & Sarankan PID' untuk melihat saran nilai PID "
            "berdasarkan parameter saat ini di log dan data respons aktual."
        )
        right_layout.addWidget(self.pid_analysis_output)

        main_layout.addWidget(right_panel, stretch=1)

        # Update daftar komponen setiap kali axis diganti
        self.pid_axis_combo.currentTextChanged.connect(self.refresh_pid_component_list)

        return central

    def _build_live_tuning_page(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)

        # --- Panel kiri: koneksi & parameter live tuning ----------------
        left_panel = QWidget()
        left_panel.setMaximumWidth(320)
        left_layout = QVBoxLayout(left_panel)

        # Grup koneksi telemetri
        conn_group = QGroupBox("Koneksi Telemetri")
        conn_layout = QVBoxLayout(conn_group)

        conn_layout.addWidget(QLabel(
            "Connection string MAVLink, contoh:\n"
            "  udp:127.0.0.1:14550  (SITL / radio via Mission Planner bridge)\n"
            "  udpin:0.0.0.0:14550  (listen UDP)\n"
            "  tcp:192.168.4.1:5760\n"
            "  /dev/ttyUSB0,57600   (serial Linux)\n"
            "  COM5,57600           (serial Windows)"
        ))
        self.live_conn_input = QLineEdit("udp:127.0.0.1:14550")
        conn_layout.addWidget(self.live_conn_input)

        conn_btn_row = QHBoxLayout()
        self.live_connect_btn = QPushButton("Connect")
        self.live_connect_btn.clicked.connect(self.live_connect)
        self.live_disconnect_btn = QPushButton("Disconnect")
        self.live_disconnect_btn.clicked.connect(self.live_disconnect)
        self.live_disconnect_btn.setEnabled(False)
        conn_btn_row.addWidget(self.live_connect_btn)
        conn_btn_row.addWidget(self.live_disconnect_btn)
        conn_layout.addLayout(conn_btn_row)

        self.live_status_label = QLabel("Status: belum terhubung")
        self.live_status_label.setWordWrap(True)
        conn_layout.addWidget(self.live_status_label)

        left_layout.addWidget(conn_group)

        # Grup live PID tuning (baca & tulis parameter saat terbang)
        tune_group = QGroupBox("Live PID Tuning")
        tune_layout = QVBoxLayout(tune_group)

        tune_layout.addWidget(QLabel(
            "⚠ Perubahan langsung memengaruhi perilaku drone saat terbang.\n"
            "Ubah nilai sedikit demi sedikit, siap-siap ambil alih kendali manual,\n"
            "dan pastikan area terbang aman."
        ))

        tune_layout.addWidget(QLabel("Axis:"))
        self.live_axis_combo = QComboBox()
        self.live_axis_map = {
            "Roll": "ATC_RAT_RLL",
            "Pitch": "ATC_RAT_PIT",
            "Yaw": "ATC_RAT_YAW",
        }
        self.live_axis_combo.addItems(list(self.live_axis_map.keys()))
        self.live_axis_combo.currentTextChanged.connect(self.rebuild_live_param_rows)
        tune_layout.addWidget(self.live_axis_combo)

        self.live_param_form_widget = QWidget()
        self.live_param_form = QFormLayout(self.live_param_form_widget)
        tune_layout.addWidget(self.live_param_form_widget)

        self.live_read_params_btn = QPushButton("Baca Nilai Parameter Saat Ini")
        self.live_read_params_btn.clicked.connect(self.live_read_params)
        tune_layout.addWidget(self.live_read_params_btn)

        left_layout.addWidget(tune_group)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # --- Panel kanan: plot live Desired vs Achieved -------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        plot_axis_row = QHBoxLayout()
        plot_axis_row.addWidget(QLabel("Tampilkan plot axis:"))
        self.live_plot_axis_combo = QComboBox()
        self.live_plot_axis_combo.addItems(["Roll", "Pitch", "Yaw"])
        plot_axis_row.addWidget(self.live_plot_axis_combo)

        plot_axis_row.addSpacing(16)
        plot_axis_row.addWidget(QLabel("Maks. titik data:"))
        self.live_max_points_input = QSpinBox()
        self.live_max_points_input.setRange(50, 50000)
        self.live_max_points_input.setSingleStep(50)
        self.live_max_points_input.setValue(self.live_max_points)
        self.live_max_points_input.setToolTip(
            "Jumlah maksimum sampel terakhir yang disimpan & ditampilkan per axis "
            "(buffer rolling). Nilai lebih kecil = plot lebih ringan & cepat."
        )
        self.live_max_points_input.valueChanged.connect(self.set_live_max_points)
        plot_axis_row.addWidget(self.live_max_points_input)

        plot_axis_row.addStretch()
        right_layout.addLayout(plot_axis_row)

        self.live_figure = Figure(figsize=(8, 5))
        self.live_ax = self.live_figure.add_subplot(111)
        self.live_canvas = FigureCanvas(self.live_figure)
        right_layout.addWidget(self.live_canvas)

        main_layout.addWidget(right_panel, stretch=1)

        self.rebuild_live_param_rows()

        return central

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
            if msg_type == "__PARAMS__":
                continue
            n_fields = len(data[msg_type]) - 1  # minus __time__
            n_samples = len(data[msg_type].get("__time__", []))
            item = QListWidgetItem(f"{msg_type}  ({n_fields} field, {n_samples} sampel)")
            item.setData(Qt.UserRole, msg_type)
            self.msg_type_list.addItem(item)

        self.status.showMessage(
            f"Berhasil memuat {os.path.basename(self.current_filepath)} — "
            f"{len(data)} tipe pesan ditemukan."
        )

        self.refresh_pid_component_list()

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

        self.refresh_scale_table()

    def refresh_scale_table(self):
        """Sinkronkan baris tabel pengkali dengan field yang sedang dipilih,
        sambil mempertahankan nilai pengkali yang sudah diisi user."""
        # Simpan nilai pengkali yang sudah ada agar tidak hilang
        existing_scales = {}
        for row in range(self.scale_table.rowCount()):
            name_item = self.scale_table.item(row, 0)
            scale_item = self.scale_table.item(row, 1)
            if name_item is not None:
                try:
                    existing_scales[name_item.text()] = float(scale_item.text())
                except (ValueError, AttributeError):
                    existing_scales[name_item.text()] = 1.0

        selected_fields = [item.text() for item in self.field_list.selectedItems()]

        self.scale_table.setRowCount(0)
        for field_name in selected_fields:
            row = self.scale_table.rowCount()
            self.scale_table.insertRow(row)

            name_item = QTableWidgetItem(field_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.scale_table.setItem(row, 0, name_item)

            scale_value = existing_scales.get(field_name, 1.0)
            scale_item = QTableWidgetItem(f"{scale_value:g}")
            self.scale_table.setItem(row, 1, scale_item)

    def get_field_scale(self, field_name):
        """Ambil nilai pengkali untuk field tertentu dari tabel (default 1.0)."""
        for row in range(self.scale_table.rowCount()):
            name_item = self.scale_table.item(row, 0)
            if name_item is not None and name_item.text() == field_name:
                scale_item = self.scale_table.item(row, 1)
                try:
                    return float(scale_item.text())
                except (ValueError, AttributeError, TypeError):
                    return 1.0
        return 1.0

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
        self._plotted_lines = []
        for item in field_items:
            field_name = item.text()
            values = bucket.get(field_name, [])
            n = min(len(t), len(values))
            if n == 0:
                continue
            scale = self.get_field_scale(field_name)
            x_arr = np.asarray(t[:n], dtype=float)
            y_arr = np.asarray(values[:n], dtype=float) * scale

            label = field_name if scale == 1 else f"{field_name} (x{scale:g})"
            line, = self.ax.plot(x_arr, y_arr, label=label, linewidth=1.0)
            self._plotted_lines.append({"line": line, "x": x_arr, "y": y_arr, "label": label})

        self.ax.set_title(f"{msg_type}")
        self.ax.set_xlabel("Waktu (ms)")
        self.ax.set_ylabel("Nilai")
        self.ax.legend(loc="best", fontsize=8)
        self.ax.grid(True, alpha=0.3)

        # ax.clear() menghapus semua artist termasuk marker/annotation hover,
        # jadi perlu dibuat ulang di Axes yang baru.
        self._hover_marker, = self.ax.plot(
            [], [], "o", color="red", markersize=6, zorder=10, visible=False
        )
        self._hover_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.85),
            arrowprops=dict(arrowstyle="->"),
            zorder=11,
            visible=False,
        )

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

    # --------------------------------------------------------- PID Tune --
    def refresh_pid_component_list(self):
        """Isi daftar komponen (Tar, Act, P, I, D, FF, dll) sesuai axis PID
        yang dipilih, berdasarkan field numerik apa saja yang tersedia
        di pesan PIDR/PIDP/PIDY pada log yang sedang dibuka."""
        self.pid_component_list.clear()

        axis_label = self.pid_axis_combo.currentText()
        msg_prefix = self.pid_axis_map.get(axis_label)
        if not msg_prefix:
            return

        bucket = self.log_data.get(msg_prefix, {})
        fields = [f for f in bucket.keys() if f != "__time__"]
        if not fields:
            return

        # Urutkan agar field umum (Tar/Des, Act) muncul lebih dulu
        priority = ["Tar", "Des", "Act"]
        fields_sorted = sorted(
            fields,
            key=lambda f: (priority.index(f) if f in priority else len(priority), f)
        )

        for f in fields_sorted:
            item = QListWidgetItem(f)
            self.pid_component_list.addItem(item)
            # Default-centang komponen Target/Actual agar langsung terlihat
            if f in ("Tar", "Des", "Act"):
                item.setSelected(True)

    def plot_pid_tune(self):
        axis_label = self.pid_axis_combo.currentText()
        msg_prefix = self.pid_axis_map.get(axis_label)
        if not msg_prefix:
            return

        bucket = self.log_data.get(msg_prefix, {})
        t = bucket.get("__time__", [])
        if not t:
            QMessageBox.information(
                self, "Info",
                f"Tidak ditemukan pesan {msg_prefix} pada log ini.\n"
                "Pastikan logging PID (LOG_BITMASK) diaktifkan saat terbang."
            )
            return

        selected = [item.text() for item in self.pid_component_list.selectedItems()]
        if not selected:
            QMessageBox.information(self, "Info", "Pilih minimal satu komponen untuk di-plot.")
            return

        # Pisahkan komponen Target/Actual (subplot atas) dari komponen P/I/D/FF (subplot bawah)
        top_fields = [f for f in selected if f in ("Tar", "Des", "Act")]
        bottom_fields = [f for f in selected if f not in ("Tar", "Des", "Act")]

        self.pid_ax_top.clear()
        self.pid_ax_bottom.clear()

        t_arr_full = np.asarray(t, dtype=float)

        def plot_into(ax, field_names):
            for field_name in field_names:
                values = bucket.get(field_name, [])
                n = min(len(t_arr_full), len(values))
                if n == 0:
                    continue
                ax.plot(t_arr_full[:n], np.asarray(values[:n], dtype=float),
                        label=field_name, linewidth=1.0)

        plot_into(self.pid_ax_top, top_fields)
        plot_into(self.pid_ax_bottom, bottom_fields)

        self.pid_ax_top.set_title(f"{axis_label} — Target vs Actual")
        self.pid_ax_top.set_ylabel("Nilai")
        self.pid_ax_top.grid(True, alpha=0.3)
        if top_fields:
            self.pid_ax_top.legend(loc="best", fontsize=8)

        self.pid_ax_bottom.set_title(f"{axis_label} — Komponen P/I/D/FF")
        self.pid_ax_bottom.set_xlabel("Waktu (ms)")
        self.pid_ax_bottom.set_ylabel("Nilai")
        self.pid_ax_bottom.grid(True, alpha=0.3)
        if bottom_fields:
            self.pid_ax_bottom.legend(loc="best", fontsize=8)

        self.pid_figure.tight_layout()
        self.pid_canvas.draw()

        self.status.showMessage(f"Plot PID Tune untuk {axis_label} ({msg_prefix}).")

    def analyze_pid_tuning(self):
        """Analisa heuristik sederhana: bandingkan Target vs Actual dari pesan
        PIDR/PIDP/PIDY, lalu sarankan penyesuaian gain P/I/D relatif terhadap
        nilai parameter (ATC_RAT_xxx_P/I/D) yang tercatat di log (pesan PARM).

        PERINGATAN: ini adalah saran awal berbasis heuristik (deteksi osilasi,
        overshoot, dan steady-state error), BUKAN auto-tune yang presisi.
        Tetap perlu diuji coba bertahap di lapangan / SITL.
        """
        axis_label = self.pid_axis_combo.currentText()
        msg_prefix = self.pid_axis_map.get(axis_label)
        if not msg_prefix:
            return

        bucket = self.log_data.get(msg_prefix, {})
        t = bucket.get("__time__", [])

        tar_key = "Tar" if "Tar" in bucket else ("Des" if "Des" in bucket else None)
        act_key = "Act" if "Act" in bucket else None

        if not t or tar_key is None or act_key is None:
            QMessageBox.information(
                self, "Info",
                f"Data Target/Actual untuk {msg_prefix} tidak lengkap pada log ini,\n"
                "tidak bisa melakukan analisa."
            )
            return

        tar = np.asarray(bucket[tar_key], dtype=float)
        act = np.asarray(bucket[act_key], dtype=float)
        t_arr = np.asarray(t, dtype=float)
        n = min(len(t_arr), len(tar), len(act))
        if n < 10:
            QMessageBox.information(self, "Info", "Data terlalu sedikit untuk dianalisa.")
            return

        t_arr, tar, act = t_arr[:n], tar[:n], act[:n]
        err = act - tar

        duration_s = max((t_arr[-1] - t_arr[0]) / 1000.0, 1e-6)
        rms_err = float(np.sqrt(np.mean(err ** 2)))
        tar_range = float(np.max(np.abs(tar))) if np.max(np.abs(tar)) > 1e-6 else 1.0
        rel_rms = rms_err / tar_range

        # Deteksi osilasi dari jumlah perlintasan nol (zero-crossing) sinyal error
        signs = np.sign(err)
        signs[signs == 0] = 1
        crossings = int(np.sum(signs[:-1] != signs[1:]))
        osc_freq_hz = crossings / (2.0 * duration_s)

        # Steady-state error: rata-rata error pada 20% data terakhir
        tail = err[int(0.8 * n):]
        steady_err_rel = abs(float(np.mean(tail))) / tar_range if len(tail) else 0.0

        # Ambil gain saat ini dari parameter log (pesan PARM), jika tersedia
        params = self.log_data.get("__PARAMS__", {})
        axis_code = {"PIDR": "RLL", "PIDP": "PIT", "PIDY": "YAW"}[msg_prefix]
        p_name = f"ATC_RAT_{axis_code}_P"
        i_name = f"ATC_RAT_{axis_code}_I"
        d_name = f"ATC_RAT_{axis_code}_D"
        ff_name = f"ATC_RAT_{axis_code}_FF"

        cur_p = params.get(p_name)
        cur_i = params.get(i_name)
        cur_d = params.get(d_name)
        cur_ff = params.get(ff_name)

        # --- Heuristik penentuan arah penyesuaian -----------------------
        notes = []
        p_factor, i_factor, d_factor = 1.0, 1.0, 1.0

        oscillating = osc_freq_hz > 4.0 and rel_rms > 0.05
        sluggish = (not oscillating) and rel_rms > 0.15

        if oscillating:
            p_factor = 0.85
            d_factor = 0.85
            notes.append(
                f"Terdeteksi osilasi pada respons (~{osc_freq_hz:.1f} Hz, "
                f"RMS error relatif {rel_rms*100:.1f}%). Disarankan turunkan P dan D "
                "sekitar 15% untuk meredam osilasi."
            )
        elif sluggish:
            p_factor = 1.15
            notes.append(
                f"Respons tampak lambat / kurang responsif (RMS error relatif "
                f"{rel_rms*100:.1f}%, osilasi rendah ~{osc_freq_hz:.1f} Hz). "
                "Disarankan naikkan P sekitar 15% agar respons lebih cepat mengejar target."
            )
        else:
            notes.append(
                f"Respons cukup baik (RMS error relatif {rel_rms*100:.1f}%, "
                f"osilasi ~{osc_freq_hz:.1f} Hz). Tidak ada perubahan besar yang disarankan."
            )

        if steady_err_rel > 0.05:
            i_factor = 1.15
            notes.append(
                f"Terdeteksi steady-state error ~{steady_err_rel*100:.1f}% pada akhir data. "
                "Disarankan naikkan I sekitar 15% agar error menetap berkurang."
            )

        # --- Susun teks hasil --------------------------------------------
        lines = [f"=== Analisa PID — {axis_label} ({msg_prefix}) ==="]
        lines.append(f"Durasi data dianalisa : {duration_s:.2f} s")
        lines.append(f"RMS error relatif      : {rel_rms*100:.2f}%")
        lines.append(f"Estimasi frek. osilasi : {osc_freq_hz:.2f} Hz")
        lines.append(f"Steady-state error     : {steady_err_rel*100:.2f}%")
        lines.append("")

        def fmt_suggestion(name, current, factor):
            if current is None:
                return f"{name:<20}: tidak ditemukan di parameter log"
            suggested = current * factor
            arrow = "→ naik" if factor > 1.0 else ("→ turun" if factor < 1.0 else "(tetap)")
            return f"{name:<20}: {current:.5f}  ->  {suggested:.5f}   {arrow}"

        lines.append(fmt_suggestion(p_name, cur_p, p_factor))
        lines.append(fmt_suggestion(i_name, cur_i, i_factor))
        lines.append(fmt_suggestion(d_name, cur_d, d_factor))
        if cur_ff is not None:
            lines.append(fmt_suggestion(ff_name, cur_ff, 1.0))
        lines.append("")
        lines.append("Catatan analisa:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
        lines.append(
            "PERINGATAN: ini saran awal berbasis heuristik dari data log, BUKAN "
            "auto-tune presisi. Ubah satu gain dalam jumlah kecil setiap kali, "
            "uji ulang, dan selalu utamakan keselamatan saat tuning di lapangan."
        )

        self.pid_analysis_output.setPlainText("\n".join(lines))
        self.status.showMessage(f"Analisa PID untuk {axis_label} selesai.")

    # --------------------------------------------------------- Live Tuning --
    def set_live_max_points(self, value):
        """Ubah jumlah maksimum titik data (buffer rolling) yang disimpan &
        ditampilkan untuk plot Live Tuning, sambil mempertahankan data
        terakhir yang sudah terkumpul (data lama di luar batas baru dibuang)."""
        self.live_max_points = int(value)
        for axis_name, buf in self.live_buffers.items():
            self.live_buffers[axis_name] = {
                "t": deque(buf["t"], maxlen=self.live_max_points),
                "des": deque(buf["des"], maxlen=self.live_max_points),
                "act": deque(buf["act"], maxlen=self.live_max_points),
            }
        self.status.showMessage(
            f"Maksimum titik data plot Live Tuning diatur ke {self.live_max_points}."
        )

    def rebuild_live_param_rows(self):
        """Bangun ulang baris parameter (P/I/D/FF) di form sesuai axis yang
        dipilih untuk Live Tuning, masing-masing dengan label nilai saat ini
        dan input nilai baru beserta tombol tulis."""
        # Bersihkan form lama
        while self.live_param_form.rowCount():
            self.live_param_form.removeRow(0)
        self.live_param_labels.clear()
        self.live_param_inputs.clear()

        axis_label = self.live_axis_combo.currentText()
        prefix = self.live_axis_map.get(axis_label)
        if not prefix:
            return

        for suffix in ("P", "I", "D", "FF"):
            param_name = f"{prefix}_{suffix}"

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            current_label = QLabel("—")
            current_label.setMinimumWidth(70)
            row_layout.addWidget(current_label)

            value_input = QDoubleSpinBox()
            value_input.setDecimals(5)
            value_input.setRange(-1000.0, 1000.0)
            value_input.setSingleStep(0.001)
            row_layout.addWidget(value_input)

            write_btn = QPushButton("Tulis")
            write_btn.clicked.connect(
                lambda checked=False, n=param_name, w=value_input: self.live_write_param(n, w.value())
            )
            row_layout.addWidget(write_btn)

            self.live_param_form.addRow(f"{param_name}:", row_widget)
            self.live_param_labels[param_name] = current_label
            self.live_param_inputs[param_name] = value_input

        # Jika sudah terhubung, langsung minta nilai terbaru untuk axis ini
        if self.live_worker is not None:
            self.live_read_params()

    def live_connect(self):
        if self.live_worker is not None:
            return

        conn_str = self.live_conn_input.text().strip()
        if not conn_str:
            QMessageBox.information(self, "Info", "Isi connection string telemetri terlebih dahulu.")
            return

        self.live_status_label.setText(f"Status: menghubungkan ke {conn_str} ...")
        self.live_connect_btn.setEnabled(False)

        self.live_worker = MavlinkLiveWorker(conn_str)
        self.live_worker.connected.connect(self.on_live_connected)
        self.live_worker.disconnected.connect(self.on_live_disconnected)
        self.live_worker.error.connect(self.on_live_error)
        self.live_worker.heartbeat.connect(self.on_live_heartbeat)
        self.live_worker.param_value.connect(self.on_live_param_value)
        self.live_worker.pid_sample.connect(self.on_live_pid_sample)
        self.live_worker.start()

    def live_disconnect(self):
        if self.live_worker is not None:
            self.live_worker.stop()
        self.live_timer.stop()
        self.live_disconnect_btn.setEnabled(False)

    def on_live_connected(self, conn_str):
        self.live_status_label.setText(f"Status: terhubung ke {conn_str}")
        self.live_connect_btn.setEnabled(False)
        self.live_disconnect_btn.setEnabled(True)
        self.live_timer.start()
        self.status.showMessage(f"Live Tuning terhubung ke {conn_str}.")
        # Ambil nilai parameter axis yang sedang aktif begitu terhubung
        self.live_read_params()

    def on_live_disconnected(self):
        self.live_status_label.setText("Status: terputus")
        self.live_connect_btn.setEnabled(True)
        self.live_disconnect_btn.setEnabled(False)
        self.live_timer.stop()
        self.live_worker = None
        self.status.showMessage("Live Tuning terputus dari drone.")

    def on_live_error(self, message):
        self.live_status_label.setText(f"Status: error — {message}")
        self.live_connect_btn.setEnabled(True)
        self.live_disconnect_btn.setEnabled(False)
        self.live_timer.stop()
        self.live_worker = None
        QMessageBox.warning(self, "Live Tuning", message)

    def on_live_heartbeat(self, info):
        armed_text = "ARMED" if info.get("armed") else "DISARMED"
        mode_text = info.get("mode", "?")
        self.live_status_label.setText(
            f"Status: terhubung — mode {mode_text}, {armed_text}"
        )

    def on_live_param_value(self, name, value):
        label = self.live_param_labels.get(name)
        if label is not None:
            label.setText(f"{value:.5f}")
        spin = self.live_param_inputs.get(name)
        if spin is not None and not spin.hasFocus():
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def on_live_pid_sample(self, axis_name, t_rel, desired, achieved, p, i, d):
        buf = self.live_buffers.get(axis_name)
        if buf is None:
            return
        buf["t"].append(t_rel)
        buf["des"].append(desired)
        buf["act"].append(achieved)

    def live_read_params(self):
        if self.live_worker is None:
            QMessageBox.information(self, "Info", "Hubungkan ke drone terlebih dahulu.")
            return
        for param_name in self.live_param_labels.keys():
            self.live_worker.request_param(param_name)

    def live_write_param(self, param_name, value):
        if self.live_worker is None:
            QMessageBox.information(self, "Info", "Hubungkan ke drone terlebih dahulu.")
            return

        confirm = QMessageBox.question(
            self, "Konfirmasi Live Tuning",
            f"Tulis parameter {param_name} = {value:.5f} ke drone SEKARANG?\n\n"
            "Pastikan ini aman dilakukan pada kondisi terbang saat ini.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        self.live_worker.set_param(param_name, value)
        self.status.showMessage(f"Mengirim {param_name} = {value:.5f} ke drone...")

    def update_live_plot(self):
        axis_name = self.live_plot_axis_combo.currentText()
        buf = self.live_buffers.get(axis_name)
        if not buf or not buf["t"]:
            return

        self.live_ax.clear()
        t_arr = np.asarray(buf["t"])
        # Tampilkan relatif terhadap waktu terbaru agar sumbu X mudah dibaca
        t_rel = (t_arr - t_arr[-1]) * 1000.0  # ms, 0 = saat ini
        self.live_ax.plot(t_rel, np.asarray(buf["des"]), label="Desired", linewidth=1.2)
        self.live_ax.plot(t_rel, np.asarray(buf["act"]), label="Achieved", linewidth=1.2)
        self.live_ax.set_title(f"Live PID Tuning — {axis_name}")
        self.live_ax.set_xlabel("Waktu relatif (ms, 0 = sekarang)")
        self.live_ax.set_ylabel("Nilai")
        self.live_ax.grid(True, alpha=0.3)
        self.live_ax.legend(loc="best", fontsize=8)
        self.live_figure.tight_layout()
        self.live_canvas.draw_idle()

    def closeEvent(self, event):
        if self.live_worker is not None:
            self.live_worker.stop()
            self.live_worker.wait(1000)
        super().closeEvent(event)

    def clear_plot(self):
        self.ax.clear()
        self._plotted_lines = []
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

    def on_hover(self, event):
        if not self._plotted_lines:
            return
        if event.inaxes != self.ax or event.xdata is None:
            if self._hover_annotation.get_visible():
                self._hover_annotation.set_visible(False)
                self._hover_marker.set_visible(False)
                self.canvas.draw_idle()
            return

        # Cari titik terdekat (dalam satuan data X) di antara semua garis yang di-plot
        best = None
        best_dist_px = None

        # Konversi posisi mouse (data coords) ke pixel agar threshold konsisten
        mouse_px = self.ax.transData.transform((event.xdata, event.ydata))

        for entry in self._plotted_lines:
            x_arr, y_arr = entry["x"], entry["y"]
            if len(x_arr) == 0:
                continue
            idx = np.searchsorted(x_arr, event.xdata)
            candidates = [i for i in (idx - 1, idx) if 0 <= i < len(x_arr)]
            for i in candidates:
                pt_px = self.ax.transData.transform((x_arr[i], y_arr[i]))
                dist_px = np.hypot(pt_px[0] - mouse_px[0], pt_px[1] - mouse_px[1])
                if best_dist_px is None or dist_px < best_dist_px:
                    best_dist_px = dist_px
                    best = (entry["label"], x_arr[i], y_arr[i])

        threshold_px = 25  # jarak maksimum (pixel) agar tooltip muncul
        if best is not None and best_dist_px is not None and best_dist_px <= threshold_px:
            label, x_val, y_val = best
            self._hover_marker.set_data([x_val], [y_val])
            self._hover_marker.set_visible(True)
            self._hover_annotation.xy = (x_val, y_val)
            self._hover_annotation.set_text(f"{label}\nt = {x_val:.2f} ms\ny = {y_val:.4g}")
            self._hover_annotation.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self._hover_annotation.get_visible():
                self._hover_annotation.set_visible(False)
                self._hover_marker.set_visible(False)
                self.canvas.draw_idle()

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

        fieldnames = ["time_ms"] + [k for k in bucket.keys() if k != "__time__"]
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