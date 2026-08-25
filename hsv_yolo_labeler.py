#!/usr/bin/env python3
"""Portable Qt GUI for HSV-assisted YOLO dataset labeling."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox,
    QFileDialog, QDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QLineEdit, QMessageBox, QPushButton, QCheckBox, QShortcut, QSlider, QSpinBox,
    QHeaderView, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

import cv2
import numpy as np

DEFAULTS = (170, 10, 100, 255, 100, 255, 500, 5)


def parse_args():
    p = argparse.ArgumentParser(description="Live HSV to YOLO label GUI")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("dataset"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


class HueRing(QWidget):
    def __init__(self):
        super().__init__(); self.start_h, self.end_h = 170, 10
        self.setMinimumSize(210, 205)

    def set_range(self, start, end):
        self.start_h, self.end_h = start, end; self.update()

    def paintEvent(self, _):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 48
        rect = QRectF((self.width() - side) / 2, 20, side, side)
        selected = lambda h: (self.start_h <= h <= self.end_h if self.start_h <= self.end_h
                              else h >= self.start_h or h <= self.end_h)
        pen = QPen()
        painter.setPen(QPen(QColor(245, 245, 245), 28, Qt.SolidLine, Qt.FlatCap))
        for h in range(180):
            if selected(h): painter.drawArc(rect, (90 - h * 2) * 16, -2 * 16)
        for h in range(180):
            color = QColor.fromHsv(h * 2, 255, 255)
            if not selected(h): color = color.darker(260)
            pen = QPen(color, 22 if selected(h) else 18, Qt.SolidLine, Qt.FlatCap)
            painter.setPen(pen)
            painter.drawArc(rect, (90 - h * 2) * 16, -2 * 16)
        span = (self.end_h - self.start_h) % 180
        painter.setPen(Qt.black)
        wrap = "wrap 179 / 0" if self.start_h > self.end_h else "current range"
        painter.drawText(rect, Qt.AlignCenter,
                         f"H {self.start_h} → {self.end_h}\n{wrap}")


class SvSpectrum(QWidget):
    def __init__(self):
        super().__init__(); self.values = (0, 100, 255, 100, 255)
        self.setMinimumSize(240, 180)

    def set_range(self, hue, low_s, high_s, low_v, high_v):
        self.values = (hue, low_s, high_s, low_v, high_v); self.update()

    def paintEvent(self, _):
        painter = QPainter(self); hue, low_s, high_s, low_v, high_v = self.values
        area = self.rect().adjusted(30, 20, -12, -28)
        spectrum = QImage(256, 256, QImage.Format_RGB32)
        for y in range(256):
            for saturation in range(256):
                spectrum.setPixelColor(saturation, y,
                    QColor.fromHsv(hue * 2, saturation, 255 - y))
        painter.drawImage(area, spectrum)
        x = lambda s: area.left() + round((area.width() - 1) * s / 255)
        y = lambda v: area.bottom() - round((area.height() - 1) * v / 255)
        selected = QRectF(x(low_s), y(high_v), max(1, x(high_s)-x(low_s)),
                          max(1, y(low_v)-y(high_v)))
        painter.setPen(QPen(Qt.white, 3)); painter.drawRect(selected)
        painter.setPen(QPen(Qt.black, 1, Qt.DashLine)); painter.drawRect(selected.adjusted(2,2,-2,-2))
        painter.drawText(2, 18, "255"); painter.drawText(5, area.bottom(), "V 0")
        painter.drawText(area.left(), self.height()-6, "S 0")
        painter.drawText(area.right()-20, self.height()-6, "255")


class ValueControl(QWidget):
    def __init__(self, low, high, value):
        super().__init__(); row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0)
        self.slider, self.spin = QSlider(Qt.Horizontal), QSpinBox()
        for widget in (self.slider, self.spin):
            widget.setRange(low, high); widget.setValue(value)
        self.spin.setFixedWidth(70)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        row.addWidget(self.slider, 1); row.addWidget(self.spin)

    def value(self): return self.slider.value()
    def setValue(self, value): self.slider.setValue(value)


class Preview(QLabel):
    def __init__(self, text):
        super().__init__(text); self._image = None
        self.setAlignment(Qt.AlignCenter); self.setMinimumSize(400, 400)
        self.setStyleSheet("background:black;color:white;border:1px solid #777")

    def show_bgr(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); h, w, c = rgb.shape
        self._image = QPixmap.fromImage(QImage(rgb.data, w, h, c*w, QImage.Format_RGB888).copy())
        self.refresh()

    def show_mask(self, image):
        h, w = image.shape
        self._image = QPixmap.fromImage(QImage(image.data, w, h, w, QImage.Format_Grayscale8).copy())
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event); self.refresh()

    def refresh(self):
        if self._image:
            self.setPixmap(self._image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ReviewDialog(QDialog):
    def __init__(self, dataset_dir, class_names, parent=None, initial_class=None):
        super().__init__(parent); self.dataset_dir = dataset_dir; self.class_names = class_names
        self.all_images = sorted(image for split in ("train","valid","test")
                                 for image in (dataset_dir/split/"images").glob("*"))
        self.images = self.all_images[:]
        self.index = 0; self.filter_class = None
        self.setWindowTitle("YOLO 라벨 검수"); self.resize(1100, 760)
        layout = QVBoxLayout(self); self.info = QLabel(); self.info.setAlignment(Qt.AlignCenter)
        self.preview = Preview("저장된 이미지가 없습니다")
        layout.addWidget(self.info); layout.addWidget(self.preview, 1)
        filter_row = QHBoxLayout(); self.filter_combo = QComboBox()
        self.filter_combo.addItem("전체 검수", -1)
        for class_id,name in enumerate(class_names): self.filter_combo.addItem(f"{class_id}: {name}", class_id)
        self.filter_combo.currentIndexChanged.connect(self.change_filter)
        if initial_class is not None: self.filter_combo.setCurrentIndex(initial_class + 1)
        filter_row.addWidget(QLabel("검수 범위")); filter_row.addWidget(self.filter_combo); filter_row.addStretch()
        layout.addLayout(filter_row)
        row = QHBoxLayout(); previous = QPushButton("◀ 이전"); following = QPushButton("다음 ▶")
        self.class_combo = QComboBox(); self.class_combo.addItems(class_names)
        apply_button = QPushButton("현재 이미지의 모든 박스에 클래스 적용")
        previous.clicked.connect(lambda: self.move(-1)); following.clicked.connect(lambda: self.move(1))
        apply_button.clicked.connect(self.apply_class)
        row.addWidget(previous); row.addWidget(following); row.addStretch()
        row.addWidget(QLabel("클래스 지정")); row.addWidget(self.class_combo); row.addWidget(apply_button)
        layout.addLayout(row)
        self.previous_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.previous_shortcut.setAutoRepeat(True); self.next_shortcut.setAutoRepeat(True)
        self.previous_shortcut.activated.connect(lambda: self.move(-1))
        self.next_shortcut.activated.connect(lambda: self.move(1))
        self.show_current()

    def label_path(self, image):
        return image.parent.parent / "labels" / f"{image.stem}.txt"

    def read_labels(self, image):
        path = self.label_path(image); result = []
        if not path.exists(): return result
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                try: result.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError: pass
        return result

    def show_current(self):
        if not self.images:
            self.preview.clear(); self.preview.setText("선택한 범위에 해당하는 이미지가 없습니다")
            self.info.setText("검수할 이미지가 없습니다"); return
        path = self.images[self.index]; image = cv2.imread(str(path))
        if image is None: return
        height, width = image.shape[:2]; labels = self.read_labels(path)
        shown_labels = labels if self.filter_class is None else [v for v in labels if v[0] == self.filter_class]
        colors = ((0,255,0),(255,160,0),(0,180,255),(255,0,180),(180,255,0))
        for class_id,cx,cy,bw,bh in shown_labels:
            x1,y1=int((cx-bw/2)*width),int((cy-bh/2)*height)
            x2,y2=int((cx+bw/2)*width),int((cy+bh/2)*height); color=colors[class_id%len(colors)]
            name=self.class_names[class_id] if class_id<len(self.class_names) else f"class {class_id}"
            cv2.rectangle(image,(x1,y1),(x2,y2),color,4)
            text=f"{class_id}: {name}"; font=cv2.FONT_HERSHEY_SIMPLEX; scale=.8; thickness=3
            (text_w,text_h),_=cv2.getTextSize(text,font,scale,thickness)
            text_y=max(text_h+8,y1)
            cv2.rectangle(image,(x1,text_y-text_h-8),(x1+text_w+10,text_y+4),color,-1)
            cv2.putText(image,text,(x1+5,text_y-3),font,scale,(0,0,0),thickness,cv2.LINE_AA)
        self.preview.show_bgr(image)
        self.info.setText(f"{self.index+1} / {len(self.images)} | {path.name} | 표시 박스 {len(shown_labels)}개")

    def change_filter(self):
        value = self.filter_combo.currentData(); self.filter_class = None if value == -1 else value
        if self.filter_class is None: self.images = self.all_images[:]
        else: self.images = [image for image in self.all_images
                             if any(label[0] == self.filter_class for label in self.read_labels(image))]
        self.index = 0; self.show_current()

    def move(self, delta):
        if self.images: self.index=(self.index+delta)%len(self.images); self.show_current()

    def apply_class(self):
        if not self.images: return
        path=self.images[self.index]; labels=self.read_labels(path); class_id=self.class_combo.currentIndex()
        lines=[f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for _,cx,cy,bw,bh in labels]
        self.label_path(path).write_text("\n".join(lines),encoding="utf-8"); self.show_current()


class DataInfoDialog(QDialog):
    COLORS = ("#7c3aed","#e91e63","#14d9b5","#ff7a00","#10a5cf","#f3e600",
              "#e515d7","#1687ef","#ffaaaa","#1515ed","#a95732","#ec18e5")

    def __init__(self, dataset_dir, class_names, counts, review_callback, parent=None):
        super().__init__(parent); self.setWindowTitle("현재 데이터 정보"); self.resize(850, 560)
        layout=QVBoxLayout(self); total=QLabel(f"전체 바운딩 박스  {sum(counts):,}개")
        total.setStyleSheet("font-size:20px;font-weight:bold;padding:8px")
        review_all=QPushButton("전체 검수"); review_all.clicked.connect(lambda: review_callback(None))
        top=QHBoxLayout(); top.addWidget(total); top.addStretch(); top.addWidget(review_all); layout.addLayout(top)
        table=QTableWidget(len(class_names),4); table.setHorizontalHeaderLabels(("색상","클래스","박스 개수","검수"))
        table.verticalHeader().setVisible(False); table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers); table.setSelectionBehavior(QTableWidget.SelectRows)
        for class_id,name in enumerate(class_names):
            color=QTableWidgetItem("●"); color.setForeground(QColor(self.COLORS[class_id%len(self.COLORS)]))
            color.setTextAlignment(Qt.AlignCenter); table.setItem(class_id,0,color)
            table.setItem(class_id,1,QTableWidgetItem(f"{class_id}  {name}"))
            count=QTableWidgetItem(f"{counts[class_id]:,}"); count.setTextAlignment(Qt.AlignCenter)
            table.setItem(class_id,2,count); button=QPushButton("클래스 검수")
            button.clicked.connect(lambda _,cid=class_id: review_callback(cid)); table.setCellWidget(class_id,3,button)
            table.setRowHeight(class_id,42)
        table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeToContents)
        layout.addWidget(table)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__(); self.args = args; self.capture = None
        self.frame, self.last_raw_frame, self.boxes = None, None, []
        self.last_move_hue = 170
        self.dataset_dir = args.output.expanduser()
        self.class_names = ["object"]
        self.frame_count = 0
        self.source_mode = "camera"
        self.video_path = None
        self.video_saving = False
        self.video_paused = False
        self.video_ended = False
        self.rotation = 0
        self.setWindowTitle("HSV YOLO Label Tool"); self.resize(1480, 880)
        self.setAcceptDrops(True)
        self.make_menu()
        root = QWidget(); outer = QHBoxLayout(root)
        self.controls_scroll = QScrollArea(); self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(390); self.controls_scroll.setMaximumWidth(390)
        self.controls_scroll.setWidget(self.make_controls())
        outer.addWidget(self.controls_scroll)
        self.preview_panel = QWidget(); right = QVBoxLayout(self.preview_panel); views = QHBoxLayout()
        for title, attr in (("HSV 필터 결과", "mask_view"),
                            ("원본 + YOLO 바운딩 박스", "original_view")):
            col = QVBoxLayout(); heading = QLabel(title); heading.setAlignment(Qt.AlignCenter)
            view = Preview("카메라 영상을 기다리는 중..."); setattr(self, attr, view)
            col.addWidget(heading); col.addWidget(view, 1); views.addLayout(col, 1)
        right.addLayout(views, 1)
        player = QHBoxLayout()
        self.play_button = QPushButton("▶ 재생"); self.rotate_button = QPushButton("↻ 90° 회전")
        self.speed_combo = QComboBox(); self.speed_combo.addItems(("0.5×","1×","2×"))
        self.speed_combo.setCurrentText("1×"); self.speed_combo.currentIndexChanged.connect(self.update_playback_speed)
        self.timeline = QSlider(Qt.Horizontal); self.timeline.setRange(0, 1000)
        self.time_label = QLabel("00:00 / 00:00"); self.time_label.setMinimumWidth(110)
        self.play_button.clicked.connect(self.toggle_video_pause)
        self.rotate_button.clicked.connect(self.rotate_video)
        self.timeline.sliderMoved.connect(self.seek_timeline)
        player.addWidget(self.play_button); player.addWidget(self.rotate_button)
        player.addWidget(QLabel("배속")); player.addWidget(self.speed_combo)
        player.addWidget(self.timeline, 1); player.addWidget(self.time_label)
        right.addLayout(player); outer.addWidget(self.preview_panel, 1); self.setCentralWidget(root)
        self.load_class_names()
        self.change_workflow_tab(0)
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_space_action)
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.left_shortcut.setAutoRepeat(True); self.right_shortcut.setAutoRepeat(True)
        self.left_shortcut.activated.connect(lambda: self.seek_seconds(-5))
        self.right_shortcut.activated.connect(lambda: self.seek_seconds(5))
        self.timer = QTimer(self); self.timer.timeout.connect(self.update_frame)
        self.open_camera(args.camera); self.timer.start(30)

    def make_menu(self):
        dataset_menu = self.menuBar().addMenu("데이터셋")
        choose = dataset_menu.addAction("폴더 선택 / 생성...")
        add_class = dataset_menu.addAction("클래스 추가...")
        rename_class = dataset_menu.addAction("현재 클래스 이름 변경...")
        delete_class = dataset_menu.addAction("현재 클래스 삭제...")
        make_yaml = dataset_menu.addAction("data.yaml 생성")
        review = dataset_menu.addAction("저장 라벨 검수...")
        dataset_menu.addSeparator()
        open_info = dataset_menu.addAction("현재 데이터 정보...")
        choose.triggered.connect(self.choose_dataset_folder)
        add_class.triggered.connect(self.add_class)
        rename_class.triggered.connect(self.rename_class)
        delete_class.triggered.connect(self.delete_class)
        make_yaml.triggered.connect(self.write_data_yaml)
        review.triggered.connect(lambda: self.open_review(None))
        open_info.triggered.connect(self.show_dataset_info)

    def make_controls(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        self.main_title = QLabel("HSV → YOLO"); self.main_title.setAlignment(Qt.AlignCenter)
        self.main_title.setStyleSheet("font-size:20px;font-weight:bold"); layout.addWidget(self.main_title)
        self.workflow_tabs = QTabWidget()
        camera = QWidget(); form = QFormLayout(camera)
        self.camera_combo = QComboBox(); self.camera_combo.addItems(map(str, range(6)))
        self.camera_combo.setCurrentText(str(self.args.camera))
        form.addRow("카메라 번호", self.camera_combo)
        camera_button = QPushButton("실시간 카메라로 전환")
        camera_button.clicked.connect(lambda: self.open_camera(int(self.camera_combo.currentText())))
        form.addRow("", camera_button)

        video = QWidget(); video_form = QFormLayout(video)
        self.video_path_edit = QLineEdit(); self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("영상을 창에 드롭하거나 선택")
        choose_video = QPushButton("영상 선택..."); preview_video = QPushButton("영상 미리보기 실행")
        save_video = QPushButton("현재 설정으로 영상 라벨링 저장")
        self.target_box_count = QSpinBox(); self.target_box_count.setRange(0,999); self.target_box_count.setValue(1)
        self.pause_on_count_mismatch = QCheckBox("목표와 다르면 자동 일시정지")
        self.pause_on_count_mismatch.setChecked(True)
        choose_video.clicked.connect(self.choose_video)
        preview_video.clicked.connect(lambda: self.start_video(False))
        save_video.clicked.connect(lambda: self.start_video(True))
        video_form.addRow("영상", self.video_path_edit); video_form.addRow("", choose_video)
        video_form.addRow("현재 클래스 목표 수", self.target_box_count)
        video_form.addRow("", self.pause_on_count_mismatch)
        video_form.addRow("", preview_video); video_form.addRow("", save_video)

        self.labeling_panel = QGroupBox("데이터셋 / 라벨 저장"); data_form = QFormLayout(self.labeling_panel)
        self.folder_label = QLabel(str(self.dataset_dir)); self.folder_label.setWordWrap(True)
        self.class_combo = QComboBox(); self.class_combo.addItems(self.class_names)
        self.interval = QSpinBox(); self.interval.setRange(1, 100000); self.interval.setValue(30)
        self.auto_save = QCheckBox("활성화")
        self.box_count_label = QLabel("0개")
        self.box_count_label.setStyleSheet("font-size:18px;font-weight:bold;color:#16a34a")
        folder_button = QPushButton("폴더 선택 / 생성")
        class_button = QPushButton("클래스 추가")
        rename_button = QPushButton("이름 변경"); delete_button = QPushButton("삭제")
        folder_button.clicked.connect(self.choose_dataset_folder)
        class_button.clicked.connect(self.add_class)
        rename_button.clicked.connect(self.rename_class); delete_button.clicked.connect(self.delete_class)
        data_form.addRow("Dataset", self.folder_label)
        data_form.addRow("", folder_button)
        data_form.addRow("현재 클래스", self.class_combo)
        data_form.addRow("", class_button)
        class_edit_widget = QWidget(); class_edit_row = QHBoxLayout(class_edit_widget)
        class_edit_row.setContentsMargins(0,0,0,0); class_edit_row.addWidget(rename_button); class_edit_row.addWidget(delete_button)
        data_form.addRow("", class_edit_widget)
        data_form.addRow("현재 박스 수", self.box_count_label)
        data_form.addRow("저장 간격(프레임)", self.interval)
        data_form.addRow("자동 저장", self.auto_save)
        review_page = QWidget(); review_layout = QVBoxLayout(review_page)
        self.review_summary = QLabel("데이터 통계를 불러오는 중...")
        self.review_summary.setStyleSheet("font-size:18px;font-weight:bold;padding:8px")
        self.review_table = QTableWidget(0,4)
        self.review_table.setHorizontalHeaderLabels(("색상","클래스","박스 개수","검수"))
        self.review_table.verticalHeader().setVisible(False); self.review_table.setAlternatingRowColors(True)
        self.review_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.review_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeToContents)
        refresh_info = QPushButton("통계 새로고침")
        refresh_info.clicked.connect(self.refresh_review_table)
        review_layout.addWidget(self.review_summary); review_layout.addWidget(self.review_table,1)
        review_layout.addWidget(refresh_info)
        self.workflow_tabs.addTab(camera,"실시간 카메라")
        self.workflow_tabs.addTab(video,"영상")
        self.workflow_tabs.addTab(review_page,"검수")
        layout.addWidget(self.workflow_tabs)
        layout.addWidget(self.labeling_panel)
        self.workflow_tabs.currentChanged.connect(self.change_workflow_tab)
        hue = QGroupBox("Hue 범위 (원형)"); self.hue_panel=hue; hue_box = QVBoxLayout(hue)
        self.ring = HueRing(); hue_box.addWidget(self.ring)
        hue_form = QFormLayout(); self.start_h = ValueControl(0,179,170); self.end_h = ValueControl(0,179,10)
        self.move_h = ValueControl(0,179,170)
        hue_form.addRow("Start H", self.start_h); hue_form.addRow("End H", self.end_h)
        hue_form.addRow("Move range", self.move_h)
        hue_box.addLayout(hue_form); layout.addWidget(hue)
        sv = QGroupBox("Saturation / Value"); self.sv_panel=sv; sv_form = QFormLayout(sv)
        self.sv_spectrum = SvSpectrum(); sv_form.addRow(self.sv_spectrum)
        self.s_low = ValueControl(0,255,100); self.s_high = ValueControl(0,255,255)
        self.v_low = ValueControl(0,255,100); self.v_high = ValueControl(0,255,255)
        for name, control in (("Lower S",self.s_low),("Upper S",self.s_high),
                              ("Lower V",self.v_low),("Upper V",self.v_high)):
            sv_form.addRow(name, control)
        layout.addWidget(sv)
        filters = QGroupBox("바운딩 박스 필터"); self.filter_panel=filters; filter_form = QFormLayout(filters)
        self.min_area = ValueControl(0,100000,500); self.kernel = ValueControl(1,31,5)
        self.remove_nested = QCheckBox("큰 박스 안의 작은 박스 제거")
        filter_form.addRow("최소 면적",self.min_area); filter_form.addRow("노이즈 제거",self.kernel)
        filter_form.addRow("", self.remove_nested)
        layout.addWidget(filters)
        self.action_panel=QWidget(); buttons = QHBoxLayout(self.action_panel)
        buttons.setContentsMargins(0,0,0,0); reset = QPushButton("기본값"); save = QPushButton("이미지 + 라벨 저장 (S)")
        reset.clicked.connect(self.reset_defaults); save.clicked.connect(self.save_sample)
        buttons.addWidget(reset); buttons.addWidget(save,1); layout.addWidget(self.action_panel)
        self.status = QLabel("준비 중..."); self.status.setWordWrap(True)
        self.status.setStyleSheet("padding:6px;background:#20242a;color:#e8e8e8")
        layout.addWidget(self.status); layout.addStretch()
        self.start_h.slider.valueChanged.connect(self.update_ring)
        self.end_h.slider.valueChanged.connect(self.update_ring)
        self.move_h.slider.valueChanged.connect(self.move_hue_range)
        for control in (self.start_h,self.end_h,self.s_low,self.s_high,self.v_low,self.v_high):
            control.slider.valueChanged.connect(self.update_spectrum)
        self.camera_combo.currentTextChanged.connect(lambda text: self.open_camera(int(text)))
        return panel

    def update_ring(self): self.ring.set_range(self.start_h.value(), self.end_h.value())

    def update_spectrum(self):
        self.sv_spectrum.set_range(self.start_h.value(), self.s_low.value(),
            self.s_high.value(), self.v_low.value(), self.v_high.value())

    def move_hue_range(self, value):
        delta = (value - self.last_move_hue + 180) % 180
        if delta > 90: delta -= 180
        self.last_move_hue = value
        self.start_h.setValue((self.start_h.value() + delta) % 180)
        self.end_h.setValue((self.end_h.value() + delta) % 180)

    def open_camera(self, index):
        if self.capture is not None: self.capture.release()
        self.source_mode = "camera"; self.video_saving = False; self.video_paused = False; self.video_ended = False
        self.play_button.setText("▶ 재생"); self.timeline.setValue(0); self.time_label.setText("00:00 / 00:00")
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,self.args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,self.args.height)
        self.status.setText(f"카메라 {index} 연결됨" if self.capture.isOpened()
                            else f"카메라 {index}을 열 수 없습니다")

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 파일 선택", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)")
        if path: self.set_video_path(path)

    def set_video_path(self, path):
        self.video_path = Path(path).resolve(); self.video_path_edit.setText(str(self.video_path))
        self.status.setText(f"영상 선택됨: {self.video_path.name}")

    def start_video(self, save_labels):
        if self.video_path is None or not self.video_path.exists():
            QMessageBox.warning(self, "영상 없음", "영상 파일을 드롭하거나 선택하세요"); return
        if self.capture is not None: self.capture.release()
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            QMessageBox.warning(self, "열기 실패", str(self.video_path)); return
        self.source_mode = "video"; self.video_saving = save_labels
        self.video_paused = False; self.video_ended = False; self.frame_count = 0
        self.play_button.setText("⏸ 일시정지")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.update_playback_speed()
        action = "라벨링 저장" if save_labels else "미리보기"
        self.status.setText(f"영상 {action} 시작: {self.video_path.name}")

    @staticmethod
    def format_time(seconds):
        seconds = max(0, int(seconds)); return f"{seconds//60:02d}:{seconds%60:02d}"

    def toggle_space_action(self):
        if self.source_mode == "video": self.toggle_video_pause()
        else: self.toggle_auto_save()

    def toggle_video_pause(self):
        if self.source_mode != "video": return
        if self.video_ended:
            self.start_video(False); return
        if self.capture is None or not self.capture.isOpened(): return
        self.video_paused = not self.video_paused
        self.play_button.setText("▶ 재생" if self.video_paused else "⏸ 일시정지")

    def playback_rate(self):
        return (0.5,1.0,2.0)[self.speed_combo.currentIndex()]

    def update_playback_speed(self):
        if not hasattr(self,"timer") or self.source_mode != "video" or self.capture is None: return
        fps=self.capture.get(cv2.CAP_PROP_FPS)
        self.timer.setInterval(max(1,round(1000/(fps*self.playback_rate()))) if fps>0 else 30)

    def rotate_video(self):
        self.rotation = (self.rotation + 90) % 360
        self.rotate_button.setText(f"↻ 회전 {self.rotation}°")

    def seek_seconds(self, seconds):
        if self.source_mode != "video" or self.capture is None or not self.capture.isOpened(): return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30
        current = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total-1, current + seconds*fps)))

    def seek_timeline(self, value):
        if self.source_mode != "video" or self.capture is None or not self.capture.isOpened(): return
        total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, total * value / 1000)

    def update_video_clock(self):
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30
        current = self.capture.get(cv2.CAP_PROP_POS_FRAMES); total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.timeline.blockSignals(True)
        self.timeline.setValue(round(1000 * current / total) if total > 0 else 0)
        self.timeline.blockSignals(False)
        self.time_label.setText(f"{self.format_time(current/fps)} / {self.format_time(total/fps)}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.set_video_path(files[0]); event.acceptProposedAction()

    def make_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sv = cv2.inRange(hsv, np.array([0,self.s_low.value(),self.v_low.value()],np.uint8),
                         np.array([179,self.s_high.value(),self.v_high.value()],np.uint8))
        h, start, end = hsv[:,:,0], self.start_h.value(), self.end_h.value()
        hue = cv2.inRange(h,start,end) if start <= end else cv2.bitwise_or(
            cv2.inRange(h,start,179),cv2.inRange(h,0,end))
        mask = cv2.bitwise_and(sv,hue); size = self.kernel.value() | 1
        kernel = np.ones((size,size),np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel),
                                cv2.MORPH_CLOSE,kernel)

    def update_frame(self):
        if self.capture is None or not self.capture.isOpened(): return
        paused_refresh = self.source_mode == "video" and self.video_paused
        if paused_refresh:
            if self.last_raw_frame is None: return
            ok, frame = True, self.last_raw_frame.copy()
        else:
            ok, frame = self.capture.read()
        if not ok:
            if self.source_mode == "video":
                finished = "영상 라벨링 저장 완료" if self.video_saving else "영상 미리보기 종료"
                self.video_saving = False; self.video_paused = True; self.video_ended = True
                self.timer.setInterval(30); self.capture.release()
                self.play_button.setText("▶ 재생"); self.timeline.setValue(1000)
                self.status.setText(finished)
            else: self.status.setText("카메라 프레임을 읽지 못했습니다")
            return
        if not paused_refresh: self.last_raw_frame = frame.copy()
        if self.source_mode == "video":
            if self.rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.update_video_clock()
        self.frame = frame; mask = self.make_mask(frame)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.boxes = sorted([cv2.boundingRect(c) for c in contours
                             if cv2.contourArea(c)>=self.min_area.value()])
        if self.remove_nested.isChecked():
            self.boxes = self.without_nested_boxes(self.boxes)
        preview = frame.copy()
        for x,y,w,h in self.boxes:
            cv2.rectangle(preview,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(preview,str(self.class_combo.currentIndex()),(x,max(18,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
        self.mask_view.show_mask(mask); self.original_view.show_bgr(preview)
        self.box_count_label.setText(f"{len(self.boxes)}개")
        count_mismatch = (self.source_mode == "video" and
                          not self.video_saving and
                          self.pause_on_count_mismatch.isChecked() and
                          len(self.boxes) != self.target_box_count.value())
        if count_mismatch and not paused_refresh:
            self.video_paused = True; self.play_button.setText("▶ 재생")
        camera_save = self.source_mode == "camera" and self.auto_save.isChecked()
        video_save = self.source_mode == "video" and self.video_saving
        if not paused_refresh:
            self.frame_count += 1
            if not count_mismatch and (camera_save or video_save) and self.frame_count % self.interval.value() == 0:
                self.save_sample(automatic=True)
        source = (f"카메라 {self.camera_combo.currentText()}" if self.source_mode == "camera"
                  else f"영상 {self.video_path.name} | {self.frame_count} 프레임")
        if count_mismatch:
            state = f" | 목표 {self.target_box_count.value()}개 불일치 — 일시정지"
        else:
            state = " | 저장 중" if video_save else ""
        self.status.setText(f"{source}{state} | 박스 {len(self.boxes)}개\n저장: {self.dataset_dir}")

    @staticmethod
    def without_nested_boxes(boxes):
        result = []
        for index, (x,y,w,h) in enumerate(boxes):
            x2,y2=x+w,y+h
            contained = any(index != other_index and ox <= x and oy <= y and
                ox+ow >= x2 and oy+oh >= y2 and ow*oh > w*h
                for other_index,(ox,oy,ow,oh) in enumerate(boxes))
            if not contained: result.append((x,y,w,h))
        return result

    def ensure_dataset(self):
        for split in ("train","valid","test"):
            for kind in ("images","labels"):
                (self.dataset_dir/split/kind).mkdir(parents=True,exist_ok=True)

    def choose_dataset_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "YOLO 데이터셋 폴더 선택", str(self.dataset_dir),
                                                     QFileDialog.ShowDirsOnly)
        if not selected: return
        selected_path = Path(selected).resolve()
        try: self.dataset_dir = selected_path.relative_to(Path.cwd().resolve())
        except ValueError: self.dataset_dir = selected_path
        self.ensure_dataset()
        self.folder_label.setText(str(self.dataset_dir)); self.load_class_names()
        if not (self.dataset_dir / "data.yaml").exists(): self.write_data_yaml(silent=True)
        self.refresh_review_table()
        self.status.setText(f"데이터셋 폴더 준비 완료: {self.dataset_dir}")

    def add_class(self):
        name, ok = QInputDialog.getText(self, "클래스 추가", "클래스 이름")
        name = name.strip()
        if not ok or not name: return
        if name in self.class_names:
            self.class_combo.setCurrentText(name); return
        self.class_names.append(name); self.class_combo.addItem(name)
        self.class_combo.setCurrentIndex(len(self.class_names)-1)
        self.write_data_yaml(silent=True)
        self.refresh_review_table()

    def load_class_names(self):
        path = self.dataset_dir / "data.yaml"
        if not path.exists() or not hasattr(self, "class_combo"): return
        text = path.read_text(encoding="utf-8"); names = []
        inline = re.search(r"^names:\s*(\[.*\])\s*$", text, re.MULTILINE)
        if inline:
            try: names = [str(value) for value in ast.literal_eval(inline.group(1))]
            except (ValueError, SyntaxError): names = []
        if not names:
            matches = re.findall(r"^\s+(\d+):\s*['\"]?(.*?)['\"]?\s*$", text, re.MULTILINE)
            names = [name.rstrip("'\"") for _,name in sorted(matches,key=lambda item:int(item[0]))]
        if names:
            self.class_names = names; self.class_combo.clear(); self.class_combo.addItems(names)

    def rename_class(self):
        source = self.class_combo.currentIndex()
        if source < 0: return
        old = self.class_names[source]
        name, ok = QInputDialog.getText(self,"클래스 이름 변경","새 이름",text=old)
        name = name.strip()
        if not ok or not name or name == old: return
        if name in self.class_names:
            target = self.class_names.index(name)
            answer = QMessageBox.question(self,"클래스 병합",
                f"'{name}' 클래스가 이미 있습니다.\n'{old}'의 모든 박스를 '{name}'으로 병합할까요?",
                QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if answer != QMessageBox.Yes: return
            self.remap_class_ids(source,target)
        else:
            self.class_names[source]=name; self.class_combo.setItemText(source,name)
        self.write_data_yaml(silent=True)
        self.refresh_review_table()

    def delete_class(self):
        source = self.class_combo.currentIndex()
        if source < 0: return
        if len(self.class_names)==1:
            QMessageBox.warning(self,"삭제 불가","클래스는 최소 한 개 필요합니다"); return
        name=self.class_names[source]
        answer=QMessageBox.question(self,"클래스 삭제",
            f"'{name}' 클래스와 해당 클래스의 모든 바운딩 박스를 삭제할까요?",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer != QMessageBox.Yes: return
        self.remap_class_ids(source,None); self.write_data_yaml(silent=True)
        self.refresh_review_table()

    def remap_class_ids(self, source, target):
        for split in ("train","valid","test"):
            for path in (self.dataset_dir/split/"labels").glob("*.txt"):
                output=[]
                for line in path.read_text(encoding="utf-8").splitlines():
                    parts=line.split()
                    if len(parts)!=5: continue
                    try: class_id=int(parts[0])
                    except ValueError: continue
                    if class_id==source:
                        if target is None: continue
                        class_id=target
                    if class_id>source: class_id-=1
                    output.append(" ".join([str(class_id),*parts[1:]]))
                path.write_text("\n".join(output),encoding="utf-8")
        selected_name=self.class_names[target] if target is not None else None
        self.class_names.pop(source); self.class_combo.clear(); self.class_combo.addItems(self.class_names)
        if selected_name in self.class_names: self.class_combo.setCurrentText(selected_name)

    def write_data_yaml(self, checked=False, silent=False):
        self.ensure_dataset()
        escaped = [name.replace("'", "''") for name in self.class_names]
        # Keep the YAML relocatable with the dataset directory. Ultralytics
        # resolves this path relative to data.yaml.
        lines = ["# Ultralytics YOLO11 dataset", "path: .",
                 "train: train/images", "val: valid/images", "test: test/images",
                 "", f"nc: {len(escaped)}", "names:"]
        lines.extend(f"  {i}: '{name}'" for i, name in enumerate(escaped))
        path = self.dataset_dir / "data.yaml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not silent: self.status.setText(f"생성 완료: {path}")

    def show_dataset_info(self):
        self.workflow_tabs.setCurrentIndex(2)
        self.refresh_review_table()

    def change_workflow_tab(self, index):
        review = index == 2
        self.preview_panel.setVisible(not review)
        for panel in (self.labeling_panel,self.hue_panel,self.sv_panel,self.filter_panel,
                      self.action_panel,self.status):
            panel.setVisible(not review)
        self.main_title.setText("데이터셋 검수" if review else "HSV → YOLO")
        if review:
            self.controls_scroll.setMinimumWidth(0); self.controls_scroll.setMaximumWidth(16777215)
            self.workflow_tabs.setMinimumHeight(520); self.workflow_tabs.setMaximumHeight(16777215)
            self.refresh_review_table()
        else:
            self.controls_scroll.setMinimumWidth(390); self.controls_scroll.setMaximumWidth(390)
            page_height=self.workflow_tabs.widget(index).sizeHint().height()+42
            self.workflow_tabs.setMinimumHeight(page_height); self.workflow_tabs.setMaximumHeight(page_height)

    def refresh_review_table(self):
        if not hasattr(self,"review_table"): return
        counts,invalid=self.class_box_counts(); self.review_table.setRowCount(len(self.class_names))
        self.review_summary.setText(f"전체 바운딩 박스  {sum(counts):,}개" +
            (f"  |  잘못된 라벨 {invalid}개" if invalid else ""))
        colors=DataInfoDialog.COLORS
        for class_id,name in enumerate(self.class_names):
            color=QTableWidgetItem("●"); color.setForeground(QColor(colors[class_id%len(colors)]))
            color.setTextAlignment(Qt.AlignCenter); self.review_table.setItem(class_id,0,color)
            self.review_table.setItem(class_id,1,QTableWidgetItem(f"{class_id}  {name}"))
            count=QTableWidgetItem(f"{counts[class_id]:,}"); count.setTextAlignment(Qt.AlignCenter)
            self.review_table.setItem(class_id,2,count)
            button=QPushButton("클래스 검수")
            button.clicked.connect(lambda _,cid=class_id:self.open_review(cid))
            self.review_table.setCellWidget(class_id,3,button); self.review_table.setRowHeight(class_id,42)

    def class_box_counts(self):
        counts = [0] * len(self.class_names); invalid = 0
        for split in ("train","valid","test"):
            for path in (self.dataset_dir/split/"labels").glob("*.txt"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    try: class_id=int(line.split()[0])
                    except (ValueError,IndexError): invalid+=1; continue
                    if 0<=class_id<len(counts): counts[class_id]+=1
                    else: invalid+=1
        return counts, invalid

    def toggle_auto_save(self):
        if self.source_mode != "camera":
            self.open_camera(int(self.camera_combo.currentText()))
        enabled = not self.auto_save.isChecked(); self.auto_save.setChecked(enabled)
        if enabled:
            self.frame_count = 0
            self.status.setText(f"자동 저장 시작: {self.interval.value()} 프레임마다 저장")
        else:
            self.status.setText("자동 저장 종료 | 저장 라벨 검수에서 결과를 확인하세요")

    def open_review(self, class_id=None):
        self.ensure_dataset()
        ReviewDialog(self.dataset_dir,self.class_names,self,initial_class=class_id).exec_()

    def save_sample(self, checked=False, automatic=False):
        if self.frame is None:
            if not automatic: QMessageBox.warning(self,"저장 실패","저장할 영상이 없습니다")
            return
        self.ensure_dataset(); self.write_data_yaml(silent=True)
        images, labels = self.dataset_dir/"train/images", self.dataset_dir/"train/labels"
        stem=datetime.now().strftime("frame_%Y%m%d_%H%M%S_%f")
        image_path, label_path=images/f"{stem}.jpg",labels/f"{stem}.txt"
        cv2.imwrite(str(image_path),self.frame); height,width=self.frame.shape[:2]
        lines=[f"{self.class_combo.currentIndex()} {(x+w/2)/width:.6f} {(y+h/2)/height:.6f} {w/width:.6f} {h/height:.6f}"
               for x,y,w,h in self.boxes]
        label_path.write_text("\n".join(lines),encoding="utf-8")
        self.status.setText(f"저장 완료: {image_path.name} ({len(lines)} boxes)")

    def reset_defaults(self):
        controls=(self.start_h,self.end_h,self.s_low,self.s_high,self.v_low,self.v_high,self.min_area,self.kernel)
        for control,value in zip(controls,DEFAULTS): control.setValue(value)
        self.last_move_hue = 170; self.move_h.setValue(170)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_S: self.save_sample()
        elif event.key() in (Qt.Key_Q,Qt.Key_Escape): self.close()
        else: super().keyPressEvent(event)

    def closeEvent(self,event):
        self.timer.stop()
        if self.capture is not None: self.capture.release()
        event.accept()


def main():
    args=parse_args(); app=QApplication(sys.argv); app.setStyle("Fusion")
    window=MainWindow(args); window.show(); return app.exec_()


if __name__=="__main__": raise SystemExit(main())
