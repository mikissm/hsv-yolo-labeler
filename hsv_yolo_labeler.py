#!/usr/bin/env python3
"""Portable Qt GUI for HSV-assisted YOLO dataset labeling."""
from __future__ import annotations

import argparse
import ast
import math
import random
import re
import shutil
import uuid
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox,
    QFileDialog, QDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QLineEdit, QMessageBox, QPushButton, QCheckBox, QShortcut, QSlider, QSpinBox,
    QHeaderView, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QProgressDialog, QSizePolicy, QVBoxLayout, QWidget)

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
        self.trash_session=dataset_dir/".trash"/datetime.now().strftime("review_%Y%m%d_%H%M%S_%f")
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
        delete_button = QPushButton("현재 이미지 삭제")
        previous.clicked.connect(lambda: self.move(-1)); following.clicked.connect(lambda: self.move(1))
        apply_button.clicked.connect(self.apply_class)
        delete_button.clicked.connect(self.delete_current_image)
        row.addWidget(previous); row.addWidget(following); row.addStretch()
        row.addWidget(QLabel("클래스 지정")); row.addWidget(self.class_combo); row.addWidget(apply_button)
        row.addWidget(delete_button)
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

    def delete_current_image(self):
        if not self.images: return
        image=self.images[self.index]; label=self.label_path(image)
        answer=QMessageBox.question(self,"현재 이미지 삭제",
            f"{image.name}\n\n이미지와 라벨을 복구 가능한 휴지통으로 이동할까요?",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        split=image.parent.parent.name
        image_target=self.trash_session/split/"images"/image.name
        label_target=self.trash_session/split/"labels"/label.name
        try:
            image_target.parent.mkdir(parents=True,exist_ok=True); label_target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(image),str(image_target))
            if label.exists(): shutil.move(str(label),str(label_target))
        except Exception as error:
            QMessageBox.critical(self,"이동 중 오류",
                f"파일을 완전히 이동하지 못했습니다. 다음 위치에서 복구 상태를 확인하세요.\n"
                f"{self.trash_session}\n\n{error}")
            self.all_images=[path for path in self.all_images if path.exists()]
            self.change_filter(); return
        self.all_images=[path for path in self.all_images if path!=image]
        self.change_filter()


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


class BoxCanvas(QWidget):
    def __init__(self,image,boxes,class_names,selected=0,parent=None):
        super().__init__(parent); self.image=image; self.boxes=boxes; self.class_names=class_names
        self.selected=max(0,min(selected,len(boxes)-1)) if boxes else -1; self.drag_corner=None
        self.setMinimumSize(760,520); self.setMouseTracking(True)

    def transform(self):
        height,width=self.image.shape[:2]; scale=min(self.width()/width,self.height()/height)
        return scale,(self.width()-width*scale)/2,(self.height()-height*scale)/2

    def paintEvent(self,_):
        painter=QPainter(self); painter.fillRect(self.rect(),Qt.black)
        rgb=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB); h,w,c=rgb.shape
        qimage=QImage(rgb.data,w,h,c*w,QImage.Format_RGB888).copy(); scale,ox,oy=self.transform()
        painter.drawImage(QRectF(ox,oy,w*scale,h*scale),qimage)
        for index,box in enumerate(self.boxes):
            _,class_id,x1,y1,x2,y2=box; color=QColor("#00ff66" if index==self.selected else "#ffcc00")
            rect=QRectF(ox+x1*scale,oy+y1*scale,(x2-x1)*scale,(y2-y1)*scale)
            painter.setPen(QPen(color,4 if index==self.selected else 2)); painter.drawRect(rect)
            name=self.class_names[class_id] if 0<=class_id<len(self.class_names) else str(class_id)
            painter.drawText(rect.topLeft()+QPointF(4,-6),f"{class_id}: {name}")
            if index==self.selected:
                painter.setBrush(color)
                for point in (rect.topLeft(),rect.topRight(),rect.bottomLeft(),rect.bottomRight()):
                    painter.drawEllipse(point,7,7)

    def mousePressEvent(self,event):
        scale,ox,oy=self.transform(); best=None
        for index,box in enumerate(self.boxes):
            _,_,x1,y1,x2,y2=box
            corners=((x1,y1),(x2,y1),(x1,y2),(x2,y2))
            for corner,(x,y) in enumerate(corners):
                distance=(event.x()-(ox+x*scale))**2+(event.y()-(oy+y*scale))**2
                if distance<=225 and (best is None or distance<best[0]): best=(distance,index,corner)
        if best: _,self.selected,self.drag_corner=best; self.update()

    def mouseMoveEvent(self,event):
        if self.drag_corner is None or self.selected<0: return
        scale,ox,oy=self.transform(); h,w=self.image.shape[:2]
        x=max(0,min(w,(event.x()-ox)/scale)); y=max(0,min(h,(event.y()-oy)/scale))
        box=self.boxes[self.selected]; x1,y1,x2,y2=box[2:]
        if self.drag_corner in (0,2): x1=min(x,x2-1)
        else: x2=max(x,x1+1)
        if self.drag_corner in (0,1): y1=min(y,y2-1)
        else: y2=max(y,y1+1)
        box[2:]=[x1,y1,x2,y2]; self.update()

    def mouseReleaseEvent(self,event):
        self.drag_corner=None; super().mouseReleaseEvent(event)

    def shrink(self,side):
        if self.selected<0: return
        box=self.boxes[self.selected]; x1,y1,x2,y2=box[2:]
        if side in ("left","all") and x2-x1>1: x1+=1
        if side in ("right","all") and x2-x1>1: x2-=1
        if side in ("top","all") and y2-y1>1: y1+=1
        if side in ("bottom","all") and y2-y1>1: y2-=1
        box[2:]=[x1,y1,x2,y2]; self.update()
class BoxEditorDialog(QDialog):
    def __init__(self,image_path,label_path,class_names,target_line,parent=None):
        super().__init__(parent); self.image_path=image_path; self.label_path=label_path
        self.lines=label_path.read_text(encoding="utf-8").splitlines(); image=cv2.imread(str(image_path))
        self.setWindowTitle(f"바운딩 박스 수정 — {image_path.name}"); self.resize(1100,760)
        h,w=image.shape[:2]; boxes=[]; selected=0
        for line_index,line in enumerate(self.lines):
            parts=line.split()
            if len(parts)!=5: continue
            try: class_id=int(parts[0]); cx,cy,bw,bh=map(float,parts[1:])
            except ValueError: continue
            if line_index+1==target_line: selected=len(boxes)
            boxes.append([line_index,class_id,(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
        layout=QVBoxLayout(self); self.canvas=BoxCanvas(image,boxes,class_names,selected,self)
        layout.addWidget(self.canvas,1); info=QLabel("꼭짓점을 드래그하거나 선택 박스의 벽을 1px 안쪽으로 이동하세요.")
        info.setAlignment(Qt.AlignCenter); layout.addWidget(info)
        row=QHBoxLayout()
        for text,side in (("왼쪽 +1px","left"),("오른쪽 -1px","right"),("위쪽 +1px","top"),
                          ("아래쪽 -1px","bottom"),("전체 1px 축소","all")):
            button=QPushButton(text); button.clicked.connect(lambda _,s=side:self.canvas.shrink(s)); row.addWidget(button)
        row.addStretch(); save=QPushButton("라벨 저장"); cancel=QPushButton("취소")
        save.clicked.connect(self.save); cancel.clicked.connect(self.reject); row.addWidget(cancel); row.addWidget(save)
        layout.addLayout(row)

    def save(self):
        h,w=self.canvas.image.shape[:2]
        for line_index,class_id,x1,y1,x2,y2 in self.canvas.boxes:
            x1=max(0,min(w,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
            self.lines[line_index]=(f"{class_id} {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} "
                                    f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        self.label_path.write_text("\n".join(self.lines)+("\n" if self.lines else ""),encoding="utf-8")
        self.accept()


class AugmentationSettingDialog(QDialog):
    def __init__(self,title,description,low_control,high_control,minimum,maximum,
                 decimals,suffix,image,renderer,parent=None):
        super().__init__(parent); self.low_control=low_control; self.high_control=high_control
        self.image=image; self.renderer=renderer; self.factor=10**decimals
        self.setWindowTitle(f"{title} 증강 설정"); self.resize(1250,700)
        layout=QVBoxLayout(self); intro=QLabel(description); intro.setWordWrap(True)
        intro.setStyleSheet("font-size:15px;padding:8px"); layout.addWidget(intro)
        previews=QHBoxLayout(); self.low_preview=Preview("최소 적용"); self.original_preview=Preview("원본")
        self.high_preview=Preview("최대 적용")
        for heading,preview in (("최소 적용",self.low_preview),("원본",self.original_preview),("최대 적용",self.high_preview)):
            column=QVBoxLayout(); label=QLabel(heading); label.setAlignment(Qt.AlignCenter)
            preview.setMinimumSize(260,340); column.addWidget(label); column.addWidget(preview,1); previews.addLayout(column,1)
        layout.addLayout(previews,1)
        form=QFormLayout(); self.low_slider=QSlider(Qt.Horizontal); self.high_slider=QSlider(Qt.Horizontal)
        self.low_spin=QDoubleSpinBox(); self.high_spin=QDoubleSpinBox()
        for slider,spin,value in ((self.low_slider,self.low_spin,low_control.value()),
                                  (self.high_slider,self.high_spin,high_control.value())):
            slider.setRange(round(minimum*self.factor),round(maximum*self.factor)); slider.setValue(round(value*self.factor))
            spin.setRange(minimum,maximum); spin.setDecimals(decimals); spin.setSuffix(suffix); spin.setValue(value)
            slider.valueChanged.connect(lambda number,s=spin:s.setValue(number/self.factor))
            spin.valueChanged.connect(lambda number,s=slider:s.setValue(round(number*self.factor)))
            slider.valueChanged.connect(self.refresh)
        low_row=QWidget(); low_layout=QHBoxLayout(low_row); low_layout.setContentsMargins(0,0,0,0)
        low_layout.addWidget(self.low_slider,1); low_layout.addWidget(self.low_spin)
        high_row=QWidget(); high_layout=QHBoxLayout(high_row); high_layout.setContentsMargins(0,0,0,0)
        high_layout.addWidget(self.high_slider,1); high_layout.addWidget(self.high_spin)
        form.addRow("최소",low_row); form.addRow("최대",high_row); layout.addLayout(form)
        self.error_label=QLabel(); self.error_label.setAlignment(Qt.AlignCenter); layout.addWidget(self.error_label)
        buttons=QHBoxLayout(); buttons.addStretch(); cancel=QPushButton("취소"); self.apply_button=QPushButton("적용")
        cancel.clicked.connect(self.reject); self.apply_button.clicked.connect(self.apply)
        buttons.addWidget(cancel); buttons.addWidget(self.apply_button); layout.addLayout(buttons)
        self.original_preview.show_bgr(image); self.refresh()

    def refresh(self,*_):
        valid=self.low_spin.value()<=self.high_spin.value()
        self.apply_button.setEnabled(valid)
        self.error_label.setText("" if valid else "최소값은 최대값보다 클 수 없습니다.")
        self.error_label.setStyleSheet("color:#dc2626;font-weight:bold")
        self.low_preview.show_bgr(self.renderer(self.image,self.low_spin.value(),101))
        self.high_preview.show_bgr(self.renderer(self.image,self.high_spin.value(),202))

    def apply(self):
        if self.low_spin.value()>self.high_spin.value():
            QMessageBox.warning(self,"범위 오류","최소값은 최대값보다 클 수 없습니다."); return
        self.low_control.setValue(self.low_spin.value()); self.high_control.setValue(self.high_spin.value())
        self.accept()


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
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(430); self.controls_scroll.setMaximumWidth(430)
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
        self.workflow_tabs.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
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
        split_group=QGroupBox("Train / Valid / Test 클래스 균형 분할")
        split_form=QFormLayout(split_group)
        self.train_ratio=QSpinBox(); self.valid_ratio=QSpinBox(); self.test_ratio=QSpinBox()
        self.train_ratio.setRange(0,100); self.valid_ratio.setRange(0,100); self.test_ratio.setRange(0,100)
        self.train_ratio.setValue(70); self.valid_ratio.setValue(20); self.test_ratio.setValue(10)
        self.split_seed=QSpinBox(); self.split_seed.setRange(0,999999999); self.split_seed.setValue(42)
        ratio_widget=QWidget(); ratio_row=QHBoxLayout(ratio_widget); ratio_row.setContentsMargins(0,0,0,0)
        for name,control in (("Train",self.train_ratio),("Valid",self.valid_ratio),("Test",self.test_ratio)):
            ratio_row.addWidget(QLabel(name)); ratio_row.addWidget(control)
        preview_split=QPushButton("분할 예상 확인"); run_split=QPushButton("클래스 균형 분할 실행")
        preview_split.clicked.connect(self.preview_dataset_split); run_split.clicked.connect(self.execute_dataset_split)
        split_form.addRow("비율 (%)",ratio_widget); split_form.addRow("랜덤 시드",self.split_seed)
        split_buttons=QWidget(); split_row=QHBoxLayout(split_buttons); split_row.setContentsMargins(0,0,0,0)
        split_row.addWidget(preview_split); split_row.addWidget(run_split)
        split_form.addRow("",split_buttons); review_layout.addWidget(split_group)
        self.review_summary = QLabel("데이터 통계를 불러오는 중...")
        self.review_summary.setStyleSheet("font-size:18px;font-weight:bold;padding:8px")
        self.review_table = QTableWidget(0,5)
        self.review_table.setHorizontalHeaderLabels(("색상","클래스","박스 개수","검수","이미지 삭제"))
        self.review_table.verticalHeader().setVisible(False); self.review_table.setAlternatingRowColors(True)
        self.review_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.review_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.review_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeToContents)
        self.review_table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeToContents)
        refresh_info = QPushButton("통계 새로고침")
        refresh_info.clicked.connect(self.refresh_review_table)
        review_sections=QTabWidget(); stats_page=QWidget(); stats_layout=QVBoxLayout(stats_page)
        stats_layout.addWidget(self.review_summary); stats_layout.addWidget(self.review_table,1)
        stats_layout.addWidget(refresh_info)
        errors_page=QWidget(); errors_layout=QVBoxLayout(errors_page)
        error_controls=QHBoxLayout(); self.error_scan_button=QPushButton("전체 데이터 오류 검사")
        self.error_scan_running=False; self.error_scan_button.clicked.connect(self.scan_dataset_errors)
        self.error_summary=QLabel("검사를 실행하면 이미지와 라벨의 오류를 확인합니다.")
        error_controls.addWidget(self.error_scan_button); error_controls.addWidget(self.error_summary,1)
        self.error_table=QTableWidget(0,5)
        self.error_table.setHorizontalHeaderLabels(("Split","파일","행","문제","수정"))
        self.error_table.verticalHeader().setVisible(False); self.error_table.setAlternatingRowColors(True)
        self.error_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.error_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.error_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeToContents)
        self.error_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        self.error_table.horizontalHeader().setSectionResizeMode(3,QHeaderView.Stretch)
        self.error_table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeToContents)
        errors_layout.addLayout(error_controls); errors_layout.addWidget(self.error_table,1)
        review_sections.addTab(stats_page,"클래스 통계"); review_sections.addTab(errors_page,"오류 파일")
        review_layout.addWidget(review_sections,1)
        aug_page=QWidget(); aug_page_layout=QHBoxLayout(aug_page)
        aug_settings_scroll=QScrollArea(); aug_settings_scroll.setWidgetResizable(True)
        aug_settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        aug_settings_scroll.setMinimumWidth(390); aug_settings_scroll.setMaximumWidth(430)
        aug_settings=QWidget(); aug_form=QFormLayout(aug_settings)
        aug_settings_scroll.setWidget(aug_settings); aug_page_layout.addWidget(aug_settings_scroll)
        aug_intro=QLabel("원본 데이터셋은 유지하고 같은 위치에 '<폴더명>_aug' 데이터셋을 생성합니다.\n"
                         "Train만 증강하며 Valid/Test는 원본 그대로 복사합니다.")
        aug_intro.setWordWrap(True); aug_intro.setStyleSheet("font-size:15px;padding:8px")
        self.aug_outputs=QSpinBox(); self.aug_outputs.setRange(1,20); self.aug_outputs.setValue(3)
        self.aug_sat_min,self.aug_sat_max=QSpinBox(),QSpinBox()
        self.aug_bright_min,self.aug_bright_max=QSpinBox(),QSpinBox()
        self.aug_exposure_min,self.aug_exposure_max=QSpinBox(),QSpinBox()
        for control in (self.aug_sat_min,self.aug_sat_max,self.aug_bright_min,self.aug_bright_max,
                        self.aug_exposure_min,self.aug_exposure_max): control.setRange(-100,100)
        self.aug_sat_min.setValue(-25); self.aug_sat_max.setValue(25)
        self.aug_bright_min.setValue(-15); self.aug_bright_max.setValue(15)
        self.aug_exposure_min.setValue(-10); self.aug_exposure_max.setValue(10)
        self.aug_blur_min=QDoubleSpinBox(); self.aug_blur=QDoubleSpinBox()
        for control in (self.aug_blur_min,self.aug_blur): control.setRange(0,20); control.setDecimals(1); control.setSuffix(" px")
        self.aug_blur_min.setValue(0); self.aug_blur.setValue(2.5)
        self.aug_noise_min=QDoubleSpinBox(); self.aug_noise=QDoubleSpinBox()
        for control in (self.aug_noise_min,self.aug_noise): control.setRange(0,10); control.setDecimals(3); control.setSuffix(" %")
        self.aug_noise_min.setValue(0); self.aug_noise.setValue(0.1)
        self.aug_seed=QSpinBox(); self.aug_seed.setRange(0,999999999); self.aug_seed.setValue(42)
        self.aug_skip_background=QCheckBox("라벨 없는 배경 이미지 증강 제외")
        self.aug_skip_background.setChecked(True); self.aug_preview_image=None; self.aug_preview_variant=0
        self.aug_setting_buttons={}
        for kind,title in (("saturation","Saturation"),("brightness","Brightness"),
                           ("exposure","Exposure"),("blur","Blur"),("noise","Noise")):
            button=QPushButton(); button.clicked.connect(lambda _,k=kind:self.open_augmentation_setting(k))
            self.aug_setting_buttons[kind]=button
        aug_buttons=QWidget(); aug_box=QVBoxLayout(aug_buttons); aug_box.setContentsMargins(0,0,0,0)
        aug_preview_row=QHBoxLayout()
        aug_preview=QPushButton("첫 Train 이미지 미리보기"); aug_reroll=QPushButton("다른 랜덤 결과")
        aug_run=QPushButton("새 _aug 데이터셋 생성")
        aug_preview.clicked.connect(self.preview_augmentation); aug_run.clicked.connect(self.run_augmentation)
        aug_reroll.clicked.connect(self.reroll_augmentation_preview)
        aug_preview_row.addWidget(aug_preview); aug_preview_row.addWidget(aug_reroll)
        aug_box.addLayout(aug_preview_row); aug_box.addWidget(aug_run)
        self.aug_estimate=QLabel(); self.aug_outputs.valueChanged.connect(self.update_augmentation_estimate)
        aug_form.addRow(aug_intro); aug_form.addRow("증강본 수 / 원본",self.aug_outputs)
        for kind in ("saturation","brightness","exposure","blur","noise"):
            aug_form.addRow(self.aug_setting_buttons[kind])
        aug_form.addRow("랜덤 시드",self.aug_seed); aug_form.addRow("배경 처리",self.aug_skip_background)
        aug_form.addRow("예상 결과",self.aug_estimate)
        aug_form.addRow("",aug_buttons)
        preview_widget=QWidget(); preview_row=QHBoxLayout(preview_widget)
        original_col=QVBoxLayout(); original_title=QLabel("원본"); original_title.setAlignment(Qt.AlignCenter)
        self.aug_original_preview=Preview("미리보기 버튼을 누르세요")
        result_col=QVBoxLayout(); result_title=QLabel("증강 결과"); result_title.setAlignment(Qt.AlignCenter)
        self.aug_result_preview=Preview("설정 변경이 실시간 반영됩니다")
        for preview in (self.aug_original_preview,self.aug_result_preview):
            preview.setMinimumSize(0,260)
            preview.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        original_col.addWidget(original_title); original_col.addWidget(self.aug_original_preview,1)
        result_col.addWidget(result_title); result_col.addWidget(self.aug_result_preview,1)
        preview_row.addLayout(original_col,1); preview_row.addLayout(result_col,1)
        aug_page_layout.addWidget(preview_widget,1)
        for control in (self.aug_outputs,self.aug_sat_min,self.aug_sat_max,self.aug_bright_min,
                        self.aug_bright_max,self.aug_exposure_min,self.aug_exposure_max,
                        self.aug_blur_min,self.aug_blur,self.aug_noise_min,self.aug_noise,self.aug_seed):
            control.valueChanged.connect(self.refresh_augmentation_preview)
        self.aug_skip_background.stateChanged.connect(self.update_augmentation_estimate)
        self.update_augmentation_button_texts()
        self.workflow_tabs.addTab(camera,"실시간 카메라")
        self.workflow_tabs.addTab(video,"영상")
        self.workflow_tabs.addTab(review_page,"검수")
        self.workflow_tabs.addTab(aug_page,"증강")
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
        layout.addWidget(self.status)
        self.start_h.slider.valueChanged.connect(self.update_ring)
        self.end_h.slider.valueChanged.connect(self.update_ring)
        self.move_h.slider.valueChanged.connect(self.move_hue_range)
        for control in (self.start_h,self.end_h,self.s_low,self.s_high,self.v_low,self.v_high):
            control.slider.valueChanged.connect(self.update_spectrum)
        self.camera_combo.currentTextChanged.connect(lambda text: self.open_camera(int(text)))
        self.update_augmentation_estimate()
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
        full_page = index in (2,3)
        self.preview_panel.setVisible(not full_page)
        for panel in (self.labeling_panel,self.hue_panel,self.sv_panel,self.filter_panel,
                      self.action_panel,self.status):
            panel.setVisible(not full_page)
        self.main_title.setText(("데이터셋 검수" if index==2 else "데이터 증강") if full_page else "HSV → YOLO")
        if full_page:
            self.controls_scroll.setMinimumWidth(0); self.controls_scroll.setMaximumWidth(16777215)
            self.workflow_tabs.setMinimumHeight(520); self.workflow_tabs.setMaximumHeight(16777215)
            self.workflow_tabs.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
            if index==2: self.refresh_review_table()
            else: self.update_augmentation_estimate()
        else:
            self.controls_scroll.setMinimumWidth(430); self.controls_scroll.setMaximumWidth(430)
            self.workflow_tabs.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
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
            self.review_table.setCellWidget(class_id,3,button)
            delete=QPushButton("해당 이미지 삭제")
            delete.clicked.connect(lambda _,cid=class_id:self.delete_images_by_class(cid))
            self.review_table.setCellWidget(class_id,4,delete); self.review_table.setRowHeight(class_id,42)

    def images_containing_class(self,class_id):
        matches=[]
        for image,label,counts in self.dataset_items():
            if class_id<len(counts) and counts[class_id]>0: matches.append((image,label))
        return matches

    def delete_images_by_class(self,class_id):
        if not 0<=class_id<len(self.class_names): return
        matches=self.images_containing_class(class_id)
        if not matches:
            QMessageBox.information(self,"삭제할 이미지 없음",f"'{self.class_names[class_id]}' 클래스가 포함된 이미지가 없습니다."); return
        answer=QMessageBox.question(self,"클래스 이미지 삭제",
            f"'{self.class_names[class_id]}' 클래스가 포함된 이미지 {len(matches):,}장을 휴지통으로 이동할까요?\n\n"
            "여러 클래스가 함께 있는 이미지는 이미지 전체와 모든 라벨이 같이 이동합니다.",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        trash=self.dataset_dir/".trash"/datetime.now().strftime(f"class_{class_id}_%Y%m%d_%H%M%S_%f")
        moved=0
        try:
            for image,label in matches:
                split=image.parent.parent.name
                image_target=trash/split/"images"/image.name; label_target=trash/split/"labels"/label.name
                image_target.parent.mkdir(parents=True,exist_ok=True); label_target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(image),str(image_target))
                if label.exists(): shutil.move(str(label),str(label_target))
                moved+=1
        except Exception as error:
            QMessageBox.critical(self,"이동 중 오류",
                f"{moved}장 이동 후 오류가 발생했습니다. 이동된 파일은 다음 위치에 있습니다.\n{trash}\n\n{error}")
            self.refresh_review_table(); return
        self.refresh_review_table()
        QMessageBox.information(self,"이동 완료",f"{moved:,}장을 휴지통으로 이동했습니다.\n{trash}")

    def scan_dataset_errors(self):
        if self.error_scan_running: return
        self.error_scan_running=True; self.error_scan_button.setEnabled(False)
        self.error_summary.setText("검사 중..."); QApplication.processEvents()
        extensions={".jpg",".jpeg",".png",".bmp",".webp"}; errors=[]
        for split in ("train","valid","test"):
            image_dir=self.dataset_dir/split/"images"; label_dir=self.dataset_dir/split/"labels"
            images={}
            for path in image_dir.glob("*"):
                if path.suffix.lower() in extensions: images.setdefault(path.stem,[]).append(path)
            labels={path.stem:path for path in label_dir.glob("*.txt")}
            for stem,paths in images.items():
                if len(paths)>1:
                    errors.append((split,", ".join(path.name for path in paths),"-","같은 이름의 이미지가 여러 형식으로 존재"))
                image_path=paths[0]
                if cv2.imread(str(image_path)) is None:
                    errors.append((split,image_path.name,"-","이미지가 손상되었거나 읽을 수 없음"))
                if stem not in labels:
                    errors.append((split,image_path.name,"-","대응하는 라벨 파일 없음"))
            for stem,label_path in labels.items():
                if stem not in images:
                    errors.append((split,label_path.name,"-","대응하는 이미지 파일 없음"))
                try: lines=label_path.read_text(encoding="utf-8").splitlines()
                except (OSError,UnicodeError) as error:
                    errors.append((split,label_path.name,"-",f"라벨 파일을 읽을 수 없음: {error}")); continue
                for line_number,line in enumerate(lines,1):
                    if not line.strip(): continue
                    parts=line.split()
                    if len(parts)!=5:
                        errors.append((split,label_path.name,str(line_number),"YOLO 라벨은 5개 값이어야 함")); continue
                    try: class_id=int(parts[0]); cx,cy,width,height=map(float,parts[1:])
                    except ValueError:
                        errors.append((split,label_path.name,str(line_number),"숫자 형식 오류")); continue
                    if not all(math.isfinite(value) for value in (cx,cy,width,height)):
                        errors.append((split,label_path.name,str(line_number),"NaN 또는 무한대 좌표")); continue
                    if not 0<=class_id<len(self.class_names):
                        errors.append((split,label_path.name,str(line_number),f"존재하지 않는 클래스 ID {class_id}"))
                    if width<=0 or height<=0:
                        errors.append((split,label_path.name,str(line_number),"박스 너비 또는 높이가 0 이하")); continue
                    if not all(0<=value<=1 for value in (cx,cy,width,height)):
                        errors.append((split,label_path.name,str(line_number),"정규화 좌표가 0~1 범위를 벗어남"))
                    if cx-width/2 < 0 or cy-height/2 < 0 or cx+width/2 > 1 or cy+height/2 > 1:
                        errors.append((split,label_path.name,str(line_number),"바운딩 박스가 이미지 경계를 벗어남"))
        self.show_dataset_errors(errors)

    def show_dataset_errors(self,errors):
        self.dataset_errors=errors; self.error_table.setRowCount(len(errors))
        for row,values in enumerate(errors):
            for column,value in enumerate(values): self.error_table.setItem(row,column,QTableWidgetItem(str(value)))
            split,file_name,line_number,_=values
            if file_name.lower().endswith(".txt") and line_number.isdigit():
                edit=QPushButton("박스 수정"); edit.clicked.connect(lambda _,r=row:self.open_error_editor(r))
                self.error_table.setCellWidget(row,4,edit)
        if errors:
            self.error_summary.setText(f"오류 {len(errors)}건 발견")
            self.error_summary.setStyleSheet("font-weight:bold;color:#dc2626")
        else:
            self.error_summary.setText("오류 없음 — 이미지와 라벨 쌍 및 좌표가 정상입니다.")
            self.error_summary.setStyleSheet("font-weight:bold;color:#16a34a")
        QTimer.singleShot(500,self.finish_error_scan)

    def finish_error_scan(self):
        self.error_scan_running=False; self.error_scan_button.setEnabled(True)

    def open_error_editor(self,row):
        if row<0 or row>=len(getattr(self,"dataset_errors",[])): return
        split,file_name,line_number,_=self.dataset_errors[row]
        label_path=self.dataset_dir/split/"labels"/file_name
        image_dir=self.dataset_dir/split/"images"
        matches=[path for path in image_dir.glob(f"{Path(file_name).stem}.*")
                 if path.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
        if not label_path.exists() or not matches:
            QMessageBox.warning(self,"수정 불가","대응하는 이미지와 라벨 파일을 찾을 수 없습니다."); return
        try: dialog=BoxEditorDialog(matches[0],label_path,self.class_names,int(line_number),self)
        except Exception as error:
            QMessageBox.critical(self,"편집기 오류",str(error)); return
        if not dialog.canvas.boxes:
            QMessageBox.warning(self,"수정 불가","편집할 수 있는 정상 형식의 박스가 없습니다."); return
        if dialog.exec_()==QDialog.Accepted: self.recheck_edited_label(split,file_name,int(line_number))

    def validate_label_line(self,label_path,line_number):
        try: lines=label_path.read_text(encoding="utf-8").splitlines()
        except (OSError,UnicodeError) as error: return [f"라벨 파일을 읽을 수 없음: {error}"]
        if line_number<1 or line_number>len(lines): return ["라벨 행이 존재하지 않음"]
        parts=lines[line_number-1].split()
        if len(parts)!=5: return ["YOLO 라벨은 5개 값이어야 함"]
        try: class_id=int(parts[0]); cx,cy,width,height=map(float,parts[1:])
        except ValueError: return ["숫자 형식 오류"]
        problems=[]
        if not all(math.isfinite(value) for value in (cx,cy,width,height)):
            return ["NaN 또는 무한대 좌표"]
        if not 0<=class_id<len(self.class_names): problems.append(f"존재하지 않는 클래스 ID {class_id}")
        if width<=0 or height<=0: problems.append("박스 너비 또는 높이가 0 이하")
        else:
            if not all(0<=value<=1 for value in (cx,cy,width,height)):
                problems.append("정규화 좌표가 0~1 범위를 벗어남")
            if cx-width/2<0 or cy-height/2<0 or cx+width/2>1 or cy+height/2>1:
                problems.append("바운딩 박스가 이미지 경계를 벗어남")
        return problems

    def recheck_edited_label(self,split,file_name,line_number):
        remaining=[error for error in getattr(self,"dataset_errors",[])
                   if not (error[0]==split and error[1]==file_name and error[2]==str(line_number))]
        label_path=self.dataset_dir/split/"labels"/file_name
        remaining.extend((split,file_name,str(line_number),problem)
                         for problem in self.validate_label_line(label_path,line_number))
        self.show_dataset_errors(remaining)
        self.error_summary.setText(
            f"수정 행만 재검사 완료 — 남은 오류 {len(remaining)}건 | 전체 확인은 '전체 데이터 오류 검사'")

    def all_train_images(self):
        extensions={".jpg",".jpeg",".png",".bmp",".webp"}
        return sorted(path for path in (self.dataset_dir/"train/images").glob("*")
                      if path.suffix.lower() in extensions and "_aug_" not in path.stem)

    def train_images(self):
        images=self.all_train_images()
        if hasattr(self,"aug_skip_background") and self.aug_skip_background.isChecked():
            images=[path for path in images if not self.is_background_image(path)]
        return images

    def is_background_image(self, image_path):
        label=self.dataset_dir/"train/labels"/f"{image_path.stem}.txt"
        return not label.exists() or not label.read_text(encoding="utf-8").strip()

    def update_augmentation_estimate(self):
        if not hasattr(self,"aug_estimate"): return
        original=len(self.all_train_images()); eligible=len(self.train_images())
        augmented=eligible*self.aug_outputs.value()
        self.aug_estimate.setText(
            f"전체 원본 {original:,}장 + 증강 {augmented:,}장 (대상 {eligible:,}장) = Train {original+augmented:,}장")

    def augmentation_ranges_valid(self):
        pairs=((self.aug_sat_min,self.aug_sat_max,"Saturation"),
               (self.aug_bright_min,self.aug_bright_max,"Brightness"),
               (self.aug_exposure_min,self.aug_exposure_max,"Exposure"),
               (self.aug_blur_min,self.aug_blur,"Blur"),
               (self.aug_noise_min,self.aug_noise,"Noise"))
        for low,high,name in pairs:
            if low.value()>high.value():
                QMessageBox.warning(self,"증강 범위 오류",f"{name}의 최소값이 최대값보다 큽니다."); return False
        return True

    def update_augmentation_button_texts(self):
        if not hasattr(self,"aug_setting_buttons"): return
        values={
            "saturation":f"Saturation  {self.aug_sat_min.value():+d}% ~ {self.aug_sat_max.value():+d}%",
            "brightness":f"Brightness  {self.aug_bright_min.value():+d}% ~ {self.aug_bright_max.value():+d}%",
            "exposure":f"Exposure  {self.aug_exposure_min.value():+d}% ~ {self.aug_exposure_max.value():+d}%",
            "blur":f"Blur  {self.aug_blur_min.value():.1f}px ~ {self.aug_blur.value():.1f}px",
            "noise":f"Noise  {self.aug_noise_min.value():.3f}% ~ {self.aug_noise.value():.3f}%",
        }
        for kind,text in values.items(): self.aug_setting_buttons[kind].setText(text)

    def apply_single_augmentation(self,image,kind,value,seed):
        if kind=="saturation":
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1]=np.clip(hsv[:,:,1]*(1+value/100),0,255)
            return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        if kind=="brightness":
            return np.clip(image.astype(np.float32)+value*2.55,0,255).astype(np.uint8)
        if kind=="exposure":
            return np.clip(image.astype(np.float32)*(1+value/100),0,255).astype(np.uint8)
        if kind=="blur":
            if value<0.15: return image.copy()
            kernel=max(3,int(math.ceil(value*3))*2+1)
            return cv2.GaussianBlur(image,(kernel,kernel),value)
        result=image.copy(); count=round(result.shape[0]*result.shape[1]*value/100)
        if count:
            rng=np.random.default_rng(seed); ys=rng.integers(0,result.shape[0],count)
            xs=rng.integers(0,result.shape[1],count)
            result[ys,xs]=rng.integers(0,256,(count,3),dtype=np.uint8)
        return result

    def open_augmentation_setting(self,kind):
        if self.aug_preview_image is None:
            images=self.train_images()
            if not images:
                QMessageBox.warning(self,"설정 미리보기 불가","증강 가능한 Train 이미지가 없습니다."); return
            self.aug_preview_image=cv2.imread(str(images[0]))
            if self.aug_preview_image is None:
                QMessageBox.warning(self,"설정 미리보기 불가","첫 Train 이미지를 읽을 수 없습니다."); return
            self.aug_original_preview.show_bgr(self.aug_preview_image)
        configs={
            "saturation":("Saturation","색의 선명도입니다. 음수는 무채색에 가깝게, 양수는 색을 진하게 만듭니다.",self.aug_sat_min,self.aug_sat_max,-100,100,0," %"),
            "brightness":("Brightness","픽셀 밝기를 일정하게 올리거나 내려 전체 영상을 밝고 어둡게 만듭니다.",self.aug_bright_min,self.aug_bright_max,-100,100,0," %"),
            "exposure":("Exposure","밝기 값을 비율로 증폭하거나 감소시켜 노출 차이를 만듭니다.",self.aug_exposure_min,self.aug_exposure_max,-100,100,0," %"),
            "blur":("Blur","가우시안 블러 강도입니다. 0은 원본이며 값이 클수록 흐려집니다.",self.aug_blur_min,self.aug_blur,0,20,1," px"),
            "noise":("Noise","무작위 픽셀 잡음 비율입니다. 0은 원본이며 과도한 값은 학습을 방해할 수 있습니다.",self.aug_noise_min,self.aug_noise,0,10,3," %"),
        }
        title,description,low,high,minimum,maximum,decimals,suffix=configs[kind]
        renderer=lambda image,value,seed:self.apply_single_augmentation(image,kind,value,seed)
        dialog=AugmentationSettingDialog(title,description,low,high,minimum,maximum,
                                         decimals,suffix,self.aug_preview_image,renderer,self)
        available=QApplication.primaryScreen().availableGeometry()
        dialog.resize(min(1250,available.width()-60),min(700,available.height()-80))
        if dialog.exec_()==QDialog.Accepted:
            self.update_augmentation_button_texts(); self.refresh_augmentation_preview()

    def augment_image(self, image, rng):
        saturation=rng.uniform(self.aug_sat_min.value(),self.aug_sat_max.value())/100
        brightness=rng.uniform(self.aug_bright_min.value(),self.aug_bright_max.value())/100
        exposure=rng.uniform(self.aug_exposure_min.value(),self.aug_exposure_max.value())/100
        result=np.clip(image.astype(np.float32)*(1+exposure)+brightness*255,0,255).astype(np.uint8)
        hsv=cv2.cvtColor(result,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*(1+saturation),0,255)
        result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        sigma=rng.uniform(self.aug_blur_min.value(),self.aug_blur.value())
        if sigma>=0.15:
            kernel=max(3,int(math.ceil(sigma*3))*2+1)
            result=cv2.GaussianBlur(result,(kernel,kernel),sigma)
        pixel_count=round(result.shape[0]*result.shape[1]*
                          rng.uniform(self.aug_noise_min.value(),self.aug_noise.value())/100)
        if pixel_count>0:
            noise_rng=np.random.default_rng(rng.randrange(2**32))
            ys=noise_rng.integers(0,result.shape[0],pixel_count)
            xs=noise_rng.integers(0,result.shape[1],pixel_count)
            result[ys,xs]=noise_rng.integers(0,256,(pixel_count,3),dtype=np.uint8)
        return result

    def preview_augmentation(self):
        if not self.augmentation_ranges_valid(): return
        images=self.train_images()
        if not images:
            QMessageBox.warning(self,"미리보기 불가","증강 가능한 Train 이미지가 없습니다.\n배경 제외 옵션도 확인하세요."); return
        original=cv2.imread(str(images[0]))
        if original is None:
            QMessageBox.warning(self,"미리보기 불가",f"이미지를 읽을 수 없습니다: {images[0].name}"); return
        self.aug_preview_image=original; self.aug_preview_variant=0
        self.aug_original_preview.show_bgr(original); self.refresh_augmentation_preview()

    def reroll_augmentation_preview(self):
        if self.aug_preview_image is None: self.preview_augmentation(); return
        self.aug_preview_variant+=1; self.refresh_augmentation_preview()

    def refresh_augmentation_preview(self,*_):
        self.update_augmentation_estimate()
        valid=(self.aug_sat_min.value()<=self.aug_sat_max.value() and
               self.aug_bright_min.value()<=self.aug_bright_max.value() and
               self.aug_exposure_min.value()<=self.aug_exposure_max.value() and
               self.aug_blur_min.value()<=self.aug_blur.value() and
               self.aug_noise_min.value()<=self.aug_noise.value())
        if self.aug_preview_image is None or not valid: return
        rng=random.Random(self.aug_seed.value()+self.aug_preview_variant*1000003)
        self.aug_result_preview.show_bgr(self.augment_image(self.aug_preview_image,rng))

    def run_augmentation(self):
        if not self.augmentation_ranges_valid(): return
        images=self.train_images()
        if not images:
            QMessageBox.warning(self,"증강 불가","train/images에 원본 이미지가 없습니다."); return
        source=self.dataset_dir.resolve(); destination=source.parent/f"{source.name}_aug"; suffix=2
        while destination.exists(): destination=source.parent/f"{source.name}_aug_{suffix}"; suffix+=1
        total=len(images)*self.aug_outputs.value()
        answer=QMessageBox.question(self,"새 증강 데이터셋 생성",
            f"{destination.name}\n\nTrain 증강본 {total:,}장을 생성할까요?\n원본 데이터셋은 변경하지 않습니다.",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        stage=source.parent/f".{destination.name}_tmp_{uuid.uuid4().hex}"
        progress=QProgressDialog("원본 데이터셋 복사 중...","취소",0,total,self)
        progress.setWindowTitle("데이터 증강"); progress.setWindowModality(Qt.WindowModal); progress.show()
        try:
            shutil.copytree(source,stage)
            rng=random.Random(self.aug_seed.value()); completed=0
            output_images=stage/"train/images"; output_labels=stage/"train/labels"
            for image_path in images:
                image=cv2.imread(str(image_path))
                if image is None: raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path.name}")
                source_label=source/"train/labels"/f"{image_path.stem}.txt"
                for index in range(1,self.aug_outputs.value()+1):
                    if progress.wasCanceled(): raise InterruptedError
                    stem=f"{image_path.stem}_aug_{index:03d}"
                    augmented=self.augment_image(image,rng)
                    if not cv2.imwrite(str(output_images/f"{stem}{image_path.suffix}"),augmented):
                        raise RuntimeError(f"이미지 저장 실패: {stem}")
                    target_label=output_labels/f"{stem}.txt"
                    if source_label.exists(): shutil.copy2(source_label,target_label)
                    else: target_label.write_text("",encoding="utf-8")
                    completed+=1; progress.setValue(completed); QApplication.processEvents()
            yaml_lines=["# Ultralytics YOLO11 dataset","path: .","train: train/images",
                        "val: valid/images","test: test/images","",f"nc: {len(self.class_names)}","names:"]
            yaml_lines.extend(f"  {i}: '{name.replace(chr(39),chr(39)*2)}'" for i,name in enumerate(self.class_names))
            (stage/"data.yaml").write_text("\n".join(yaml_lines)+"\n",encoding="utf-8")
            stage.rename(destination); progress.setValue(total)
        except InterruptedError:
            if stage.exists(): shutil.rmtree(stage)
            progress.close(); QMessageBox.information(self,"증강 취소","임시 결과를 삭제했습니다."); return
        except Exception as error:
            progress.close(); QMessageBox.critical(self,"증강 실패",f"임시 결과를 보존했습니다.\n{stage}\n\n{error}"); return
        QMessageBox.information(self,"증강 완료",f"새 데이터셋을 생성했습니다.\n{destination}")

    def dataset_items(self):
        extensions={".jpg",".jpeg",".png",".bmp",".webp"}; items=[]
        for split in ("train","valid","test"):
            for image in sorted((self.dataset_dir/split/"images").glob("*")):
                if image.suffix.lower() not in extensions: continue
                label=self.dataset_dir/split/"labels"/f"{image.stem}.txt"
                counts=[0]*len(self.class_names)
                if label.exists():
                    for line in label.read_text(encoding="utf-8").splitlines():
                        try: class_id=int(line.split()[0])
                        except (ValueError,IndexError): continue
                        if 0<=class_id<len(counts): counts[class_id]+=1
                items.append((image,label,counts))
        return items

    def make_split_plan(self):
        ratios=[self.train_ratio.value(),self.valid_ratio.value(),self.test_ratio.value()]
        if sum(ratios)!=100:
            QMessageBox.warning(self,"비율 오류",f"Train + Valid + Test 합계가 100이어야 합니다. 현재 {sum(ratios)}입니다.")
            return None
        items=self.dataset_items(); total=len(items)
        if total==0:
            QMessageBox.warning(self,"데이터 없음","분할할 이미지가 없습니다."); return None
        raw=[total*ratio/100 for ratio in ratios]; target=[int(value) for value in raw]
        for index in sorted(range(3),key=lambda i:raw[i]-target[i],reverse=True)[:total-sum(target)]: target[index]+=1
        class_totals=[sum(item[2][c] for item in items) for c in range(len(self.class_names))]
        class_targets=[[value*ratio/100 for value in class_totals] for ratio in ratios]
        assigned=[[],[],[]]; class_now=[[0]*len(self.class_names) for _ in range(3)]
        rng=random.Random(self.split_seed.value())
        decorated=[]
        for item in items:
            rarity=sum(count/max(1,class_totals[c]) for c,count in enumerate(item[2]))
            decorated.append((rarity,sum(item[2]),rng.random(),item))
        for _,_,_,item in sorted(decorated,reverse=True,key=lambda row:(row[0],row[1],row[2])):
            candidates=[i for i in range(3) if len(assigned[i])<target[i]] or list(range(3))
            def score(i):
                class_need=sum(count*max(0,class_targets[i][c]-class_now[i][c])/
                               max(1,class_targets[i][c]) for c,count in enumerate(item[2]))
                capacity=(target[i]-len(assigned[i]))/max(1,target[i])
                return class_need+0.25*capacity+rng.random()*1e-8
            chosen=max(candidates,key=score); assigned[chosen].append(item)
            for c,count in enumerate(item[2]): class_now[chosen][c]+=count
        return assigned,class_now

    def split_summary(self, plan):
        assigned,class_counts=plan; names=("Train","Valid","Test")
        lines=[f"{names[i]}: 이미지 {len(assigned[i])}장" for i in range(3)]
        lines.append("")
        for class_id,name in enumerate(self.class_names):
            values=" / ".join(str(class_counts[i][class_id]) for i in range(3))
            lines.append(f"{class_id} {name}: {values}  (Train / Valid / Test)")
        return "\n".join(lines)

    def preview_dataset_split(self):
        plan=self.make_split_plan()
        if plan: QMessageBox.information(self,"클래스 균형 분할 예상",self.split_summary(plan))

    def execute_dataset_split(self):
        plan=self.make_split_plan()
        if not plan: return
        answer=QMessageBox.question(self,"데이터셋 분할 실행",
            self.split_summary(plan)+"\n\n이미지와 라벨을 위 구성으로 이동할까요?",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        stage=self.dataset_dir/f".split_staging_{uuid.uuid4().hex}"
        stage.mkdir(parents=True)
        staged=[]
        try:
            for split_index,items in enumerate(plan[0]):
                for item_index,(image,label,_) in enumerate(items):
                    token=f"{split_index}_{item_index}_{image.name}"
                    staged_image=stage/token; shutil.move(str(image),str(staged_image))
                    staged_label=None
                    if label.exists():
                        staged_label=stage/f"{split_index}_{item_index}_{label.name}"
                        shutil.move(str(label),str(staged_label))
                    staged.append((split_index,staged_image,staged_label,image.name))
            split_names=("train","valid","test")
            for split_index,image,label,original_name in staged:
                image_dir=self.dataset_dir/split_names[split_index]/"images"
                label_dir=self.dataset_dir/split_names[split_index]/"labels"
                destination=image_dir/original_name; suffix=1
                while destination.exists():
                    destination=image_dir/f"{Path(original_name).stem}_split{suffix}{Path(original_name).suffix}"; suffix+=1
                shutil.move(str(image),str(destination))
                label_destination=label_dir/f"{destination.stem}.txt"
                if label is not None: shutil.move(str(label),str(label_destination))
                else: label_destination.write_text("",encoding="utf-8")
        except Exception as error:
            QMessageBox.critical(self,"분할 실패",
                f"분할 중 오류가 발생했습니다. 복구를 위해 임시 파일을 유지합니다.\n{stage}\n\n{error}")
            return
        if stage.exists(): shutil.rmtree(stage)
        self.refresh_review_table()
        QMessageBox.information(self,"분할 완료","클래스 분포를 고려해 데이터셋을 분할했습니다.")

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
        self.refresh_review_table()

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
    window=MainWindow(args)
    available=app.primaryScreen().availableGeometry()
    window.resize(min(1480,max(900,available.width()-40)),
                  min(880,max(650,available.height()-60)))
    window.move(available.x()+(available.width()-window.width())//2,
                available.y()+(available.height()-window.height())//2)
    window.show(); return app.exec_()


if __name__=="__main__": raise SystemExit(main())
