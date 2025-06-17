# --- 1. IMPORTAÇÕES ---
import sys
import cv2
import os
import random
import time
import pygame
from datetime import datetime
import csv

from ultralytics import YOLO

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QMessageBox, 
                             QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QSizePolicy, QCheckBox, 
                             QFileDialog, QDialog, QLineEdit, QFormLayout, QDialogButtonBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- CLASSE PARA A JANELA DE DIÁLOGO DE LOGIN RTSP ---
class RTSPLoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conectar a Câmera IP/DVR")
        layout = QFormLayout(self)
        self.user_input = QLineEdit(self)
        self.pass_input = QLineEdit(self); self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.ip_input = QLineEdit(self); self.ip_input.setPlaceholderText("ex: 192.168.0.108")
        self.channel_input = QLineEdit(self); self.channel_input.setText("1")
        layout.addRow("Usuário:", self.user_input); layout.addRow("Senha:", self.pass_input)
        layout.addRow("IP/Endereço:", self.ip_input); layout.addRow("Canal:", self.channel_input)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    def get_rtsp_url(self):
        user, password, ip, channel = self.user_input.text().strip(), self.pass_input.text().strip(), self.ip_input.text().strip(), self.channel_input.text().strip()
        if not ip: return None
        return f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0"

# --- 2. CLASSE DA THREAD DE VÍDEO ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    finished_signal = pyqtSignal()
    connection_failed_signal = pyqtSignal()

    def __init__(self, model_path, video_source=0, sound_enabled=False, selected_sound=""):
        super().__init__()
        self.model_path = model_path
        self.video_source = video_source
        self._run_flag = True
        self.sound_dir = 'sons'
        self.sound_files = [f for f in os.listdir(self.sound_dir) if f.endswith('.mp3')] if os.path.isdir(self.sound_dir) else []
        self.auto_sound_enabled = sound_enabled
        self.selected_sound = selected_sound
        self.pombo_detectado_anteriormente = False
    
    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {self.video_source}")
            self.connection_failed_signal.emit()
            return
        
        model = YOLO(self.model_path)
        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = model(frame, verbose=False)
            pombo_detectado_agora = len(results[0].boxes) > 0
            if pombo_detectado_agora and not self.pombo_detectado_anteriormente:
                timestamp = datetime.now()
                detection_count = len(results[0].boxes)
                camera_id = f"Câmera {self.video_source}" if isinstance(self.video_source, int) else self.video_source
                log_file = 'datalog.csv'
                file_exists = os.path.isfile(log_file)
                try:
                    with open(log_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists: writer.writerow(['Data', 'Hora', 'Contagem', 'Camera'])
                        writer.writerow([timestamp.strftime('%Y-%m-%d'), timestamp.strftime('%H:%M:%S'), detection_count, camera_id])
                except Exception as e: print(f"Erro ao escrever no log: {e}")
                if self.auto_sound_enabled and not pygame.mixer.get_busy() and self.sound_files:
                    sound_to_play_path = None
                    if self.selected_sound == "Aleatório": sound_to_play_path = os.path.join(self.sound_dir, random.choice(self.sound_files))
                    elif self.selected_sound in self.sound_files: sound_to_play_path = os.path.join(self.sound_dir, self.selected_sound)
                    if sound_to_play_path: pygame.mixer.Sound(sound_to_play_path).play()
            self.pombo_detectado_anteriormente = pombo_detectado_agora
            annotated_frame = results[0].plot()
            count_text = f"Pombos Detectados: {len(results[0].boxes)}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
        cap.release()
        self.finished_signal.emit()

    def stop(self):
        self._run_flag = False

# --- 3. CLASSE DA JANELA PRINCIPAL (COM AS MODIFICAÇÕES) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PigeonPeek - Equipe 15")
        self.setGeometry(100, 100, 850, 700)
        self.model_path = "D:/Porto Itaqui/Yolo/Aplicação/treinamento/pombos_yolov12n/weights/best.pt"
        self.video_thread = None
        pygame.mixer.init()
        main_layout = QVBoxLayout()
        
        camera_controls_layout = QHBoxLayout()
        camera_controls_layout.addWidget(QLabel("Fonte de Vídeo:"))
        self.camera_selector = QComboBox()
        camera_controls_layout.addWidget(self.camera_selector)
        self.btn_add_ip_cam = QPushButton("Adicionar Câmera IP")
        self.btn_add_ip_cam.clicked.connect(self.open_rtsp_login_dialog)
        self.btn_add_ip_cam.setEnabled(False) 
        camera_controls_layout.addWidget(self.btn_add_ip_cam)
        self.btn_refresh = QPushButton("Atualizar Webcams")
        self.btn_refresh.clicked.connect(self.scan_for_cameras)
        camera_controls_layout.addWidget(self.btn_refresh)
        self.btn_toggle = QPushButton("Iniciar Detecção")
        self.btn_toggle.clicked.connect(self.toggle_webcam)
        camera_controls_layout.addWidget(self.btn_toggle)
        self.btn_report = QPushButton("Emitir Relatório")
        self.btn_report.clicked.connect(self.generate_report)
        camera_controls_layout.addWidget(self.btn_report)
        
        sound_controls_layout = QHBoxLayout()
        self.auto_sound_checkbox = QCheckBox("Alerta Automático")
        self.auto_sound_checkbox.setChecked(True)
        sound_controls_layout.addWidget(self.auto_sound_checkbox)
        sound_controls_layout.addWidget(QLabel("Áudio:"))
        self.sound_selector = QComboBox()
        sound_controls_layout.addWidget(self.sound_selector)
        self.btn_play_sound = QPushButton("Emitir Som")
        self.btn_play_sound.clicked.connect(self.play_selected_sound)
        sound_controls_layout.addWidget(self.btn_play_sound)
        self.btn_stop_sound = QPushButton("Parar Som")
        self.btn_stop_sound.clicked.connect(self.stop_all_sounds)
        sound_controls_layout.addWidget(self.btn_stop_sound)
        
        self.video_label = QLabel("Selecione uma fonte de vídeo e pressione 'Iniciar Detecção'", self)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(True)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        
        main_layout.addLayout(camera_controls_layout)
        main_layout.addLayout(sound_controls_layout)
        main_layout.addWidget(self.video_label)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.scan_for_cameras()
        self.scan_for_sounds()

    def open_rtsp_login_dialog(self):
        dialog = RTSPLoginDialog(self)
        if dialog.exec():
            url = dialog.get_rtsp_url()
            if url:
                ip_address = dialog.ip_input.text()
                self.camera_selector.addItem(f"IP: {ip_address}", userData=url)
                self.camera_selector.setCurrentText(f"IP: {ip_address}")
                self.btn_toggle.setEnabled(True)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O campo de IP/Endereço é obrigatório.")

    def play_selected_sound(self):
        if pygame.mixer.get_busy():
            print("Comando ignorado: um som já está em reprodução.")
            return
        selected_filename = self.sound_selector.currentText()
        if "não encontrada" in selected_filename: return
        sound_path = ""
        if selected_filename == "Aleatório":
            sound_dir = 'sons'
            if os.path.isdir(sound_dir):
                all_sounds = [f for f in os.listdir(sound_dir) if f.endswith('.mp3')]
                if all_sounds: sound_path = os.path.join(sound_dir, random.choice(all_sounds))
        else:
            sound_path = os.path.join('sons', selected_filename)
        if sound_path and os.path.exists(sound_path):
            pygame.mixer.Sound(sound_path).play()

    def toggle_webcam(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.on_processing_finished()
        else:
            current_index = self.camera_selector.currentIndex()
            if current_index == -1: return
            selected_text, user_data = self.camera_selector.currentText(), self.camera_selector.itemData(current_index)
            video_source = user_data if user_data else (int(selected_text.split()[-1]) if selected_text.startswith("Câmera") else None)
            if video_source is None:
                QMessageBox.warning(self, "Fonte Inválida", "Nenhuma fonte de vídeo válida selecionada.")
                return

            auto_sound_is_enabled, selected_sound_name = self.auto_sound_checkbox.isChecked(), self.sound_selector.currentText()
            
            self.video_thread = VideoThread(
                model_path=self.model_path, 
                video_source=video_source,
                sound_enabled=auto_sound_is_enabled,
                selected_sound=selected_sound_name
            )
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.finished_signal.connect(self.on_processing_finished)
            self.video_thread.connection_failed_signal.connect(self.on_connection_failed)
            
            self.video_thread.start()
            self.btn_toggle.setText("Parar Detecção")
            self.set_controls_enabled(False)
            self.video_label.setText("Tentando conectar à câmera...")

    def on_connection_failed(self):
        QMessageBox.critical(self, "Erro de Conexão", "Não foi possível conectar à fonte de vídeo selecionada.\nVerifique se os dados estão corretos e se a câmera está na rede.")
        self.on_processing_finished()

    def on_processing_finished(self):
        if self.video_thread:
            try:
                self.video_thread.change_pixmap_signal.disconnect()
                self.video_thread.finished_signal.disconnect()
                self.video_thread.connection_failed_signal.disconnect()
            except TypeError:
                pass
        self.video_label.clear()
        self.video_label.setText("Detecção parada. Selecione uma fonte de vídeo para começar.")
        self.btn_toggle.setText("Iniciar Detecção")
        self.set_controls_enabled(True)
        self.stop_all_sounds()

    def generate_report(self):
        log_file = 'datalog.csv';
        if not os.path.exists(log_file): QMessageBox.warning(self, "Relatório", "Nenhum dado registrado ainda."); return
        default_filename = f"relatorio_{datetime.now().strftime('%Y-%m-%d')}.txt"
        save_path, _ = QFileDialog.getSaveFileName(self, "Salvar Relatório Como", default_filename, "Arquivos de Texto (*.txt);;Todos os Arquivos (*)")
        if not save_path: return
        daily_counts, camera_counts, total_detections = {}, {}, 0
        try:
            with open(log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f); next(reader)
                for row in reader:
                    if len(row) < 4: continue
                    data, _, contagem_str, camera = row; contagem = int(contagem_str); total_detections += contagem
                    daily_counts[data] = daily_counts.get(data, 0) + contagem; camera_counts[camera] = camera_counts.get(camera, 0) + contagem
            report_content = [
                "--- Relatório de Detecção de Pombos ---\n", f"Relatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                "="*40 + "\n", f"TOTAL DE POMBOS DETECTADOS: {total_detections}\n", "="*40 + "\n\n", "--- Detecções por Dia ---\n"
            ]
            for data, contagem in sorted(daily_counts.items()): report_content.append(f"- {data}: {contagem} pombos\n")
            report_content.append("\n--- Detecções por Câmera ---\n")
            for camera, contagem in sorted(camera_counts.items()): report_content.append(f"- {camera}: {contagem} pombos\n")
            with open(save_path, 'w', encoding='utf-8') as f: f.write("".join(report_content))
            QMessageBox.information(self, "Sucesso", f"Relatório salvo em:\n{save_path}")
        except Exception as e: QMessageBox.critical(self, "Erro ao Gerar Relatório", f"Ocorreu um erro: {e}")
        
    def scan_for_sounds(self):
        self.sound_selector.clear(); self.sound_selector.addItem("Aleatório"); sound_dir = 'sons'
        if os.path.isdir(sound_dir):
            sound_files = [f for f in os.listdir(sound_dir) if f.endswith('.mp3')]
            if sound_files: self.sound_selector.addItems(sound_files)
        else:
            self.sound_selector.addItem("Pasta 'sons' não encontrada"); self.btn_play_sound.setEnabled(False)
            
    def stop_all_sounds(self):
        pygame.mixer.stop()
        
    def scan_for_cameras(self):
        self.camera_selector.clear(); index = 0; available_cameras = []
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW);
            if cap.isOpened(): available_cameras.append(f"Câmera {index}"); cap.release(); index += 1
            else: break
            if index > 10: break
        self.camera_selector.addItems(available_cameras)
        if self.camera_selector.count() == 0:
            self.camera_selector.addItem("Nenhuma câmera encontrada"); self.btn_toggle.setEnabled(False)
            
    def set_controls_enabled(self, enabled):
        self.camera_selector.setEnabled(enabled)
        # --- CORREÇÃO APLICADA AQUI ---
        # Esta linha que controlava o botão foi removida para que ele
        # permaneça no estado definido no __init__ (inativo).
        # self.btn_add_ip_cam.setEnabled(enabled) 
        self.btn_refresh.setEnabled(enabled)
        self.auto_sound_checkbox.setEnabled(enabled)
        self.sound_selector.setEnabled(enabled)
        self.btn_play_sound.setEnabled(enabled)
        self.btn_report.setEnabled(enabled)
        
    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning(): self.video_thread.stop()
        pygame.mixer.quit()
        event.accept()

# --- 4. PONTO DE ENTRADA DA APLICAÇÃO ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())