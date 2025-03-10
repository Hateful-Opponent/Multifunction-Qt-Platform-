import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import mole
import pet
import threading
from face_detectation_module import FaceDetectionWidget

class MainWindow(QMainWindow):
    game_finished = pyqtSignal(int)  # 定义一个信号，用于传递得分   

    def __init__(self):
        super().__init__()

        self.setMouseTracking(True)
        self.oldPos = self.pos()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setStyleSheet("QMainWindow {background-image: url('./photo/background.jpg');}")

        self.pet_onpened = False

        self.setGeometry(450, 300, 1200, 800)

        # Create a central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a QStackedWidget
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        self.button_layout = QGridLayout()
        layout.addLayout(self.button_layout)

        # Create the main interface
        self.create_main()

        # Create the game interface
        self.create_game_whack_a_mole()
        
        self.pet()

        self.det()

        self.talk()

        # Create buttons to switch between interfaces
        self.Home = QPushButton('主界面')
        self.Home.setFixedSize(100, 60)  # 设置按钮的宽度为100，高度为30
        self.Home.setFont(QFont('Times', 11, QFont.Black))
        self.Home.clicked.connect(self.show_main)
        self.adjust_font_size(self.Home)

        self.Home.hide()

        
        self.det = QPushButton('检测')
        self.det.setFont(QFont('Times', 20, QFont.Black))
        self.det.setFixedSize(200, 80)  
        self.det.clicked.connect(self.show_det)
        self.adjust_font_size(self.det)
  

        self.talk = QPushButton('聊聊天')
        self.talk.setFont(QFont('Times', 20, QFont.Black))
        self.talk.setFixedSize(200, 80)  
        self.talk.clicked.connect(self.show_talk)
        self.adjust_font_size(self.talk)


        self.change_to_mole = QPushButton('打地鼠')
        self.change_to_mole.setFont(QFont('Times', 20, QFont.Black))
        self.change_to_mole.setFixedSize(200, 80)  # 设置按钮的宽度为100，高度为30
        self.change_to_mole.clicked.connect(self.show_game_whack_a_mole)
        self.adjust_font_size(self.change_to_mole)
    

    
        self.exit = QPushButton('关闭')
        self.exit.setFont(QFont('Times', 20, QFont.Black))
        self.exit.setFixedSize(200, 80)
        self.exit.clicked.connect(self.close)


        self.intelligent_pet = QPushButton('宠物')
        self.intelligent_pet.setFont(QFont('Times', 20, QFont.Black))
        self.intelligent_pet.setFixedSize(200, 80)
        self.intelligent_pet.clicked.connect(self.show_pet)

        # Add buttons to button layout using QGridLayout
        self.button_layout.addWidget(self.Home, 0, 0, 1, 2)
        self.button_layout.addWidget(self.change_to_mole, 1, 0)
        self.button_layout.addWidget(self.det, 1, 1)
        self.button_layout.addWidget(self.intelligent_pet, 2, 0)
        self.button_layout.addWidget(self.exit, 2, 2)
        self.button_layout.addWidget(self.talk, 2, 1)

        self.game_finished.connect(self.update_score)

    def create_centered_layout(self, widget):
        layout = QHBoxLayout()
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(widget)
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        return layout
    
    def talk(self):
        # Create the main interface layout and widgets
        pets = QWidget()
        layout = QVBoxLayout(pets)
        label = QLabel('对话治疗模式正在开发中')
        label.setFont(QFont('Times', 30, QFont.Black))
        label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(label)
        self.stacked_widget.addWidget(pets)

    def pet(self):
        # Create the main interface layout and widgets
        pets = QWidget()
        layout = QVBoxLayout(pets)
        label = QLabel('Q版萌宠')
        label.setFont(QFont('Times', 30, QFont.Black))
        label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(label)

        # Add the main interface to the stacked widget
        self.stacked_widget.addWidget(pets)
        # Add the main interface to the stacked widget
        self.stacked_widget.addWidget(pets)

        self.start_button = QPushButton('召唤宠物')
        self.start_button.clicked.connect(pet.start_pet_thread)
        self.start_button.setFont(QFont('Times', 20, QFont.Black))
        self.start_button.setFixedSize(200, 80)
        layout.addWidget(self.start_button)


        self.end_button = QPushButton('召回宠物')
        self.end_button.clicked.connect(pet.destroy_pet_thread)
        self.end_button.setFont(QFont('Times', 20, QFont.Black))
        self.end_button.setFixedSize(200, 80)
        layout.addWidget(self.end_button)

    def det(self):
        # Create the main interface layout and widgets
        pets = QWidget()
        layout = QVBoxLayout(pets)
        label = QLabel('检测')
        label.setFont(QFont('Times', 30, QFont.Black))  # 设置标签文本居中
        label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(label)

        self.face_detection_widget = FaceDetectionWidget()
        layout.addWidget(self.face_detection_widget)
        # Add the main interface to the stacked widget
        self.stacked_widget.addWidget(pets)

    def create_main(self):
        # Create the main interface layout and widgets
        pets = QWidget()
        layout = QVBoxLayout(pets)
        label = QLabel('主界面')
        label.setFont(QFont('Times', 30, QFont.Black))
        label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(label)

        # Add the main interface to the stacked widget
        self.stacked_widget.addWidget(pets)

    def create_game_whack_a_mole(self):
        # Create the second interface layout and widgets
        game_whack_a_mole = QWidget()
        layout = QVBoxLayout(game_whack_a_mole)
        label = QLabel('打地鼠')
        label.setFont(QFont('Times', 30, QFont.Black))
        label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(label)
        
        self.score_label = QLabel('您的得分是:0')
        self.score_label.setFont(QFont('Times', 15, QFont.Black))
        self.score_label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(self.score_label)

        self.time_label = QLabel('请输入本次游戏时间（秒）：')
        layout.addWidget(self.time_label)

        self.time_spinbox = QSpinBox()
        self.time_spinbox.setRange(1, 3600)  # 设置时间范围，例如1秒到1小时
        self.time_spinbox.setValue(60)  # 默认60秒
        layout.addWidget(self.time_spinbox)

        self.difficulty_label = QLabel('请选择游戏难度：')
        layout.addWidget(self.difficulty_label)

        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["简单", "中等", "困难"])
        layout.addWidget(self.difficulty_combo)

        self.start_button = QPushButton('开始游戏')
        self.start_button.clicked.connect(self.start_game)
        layout.addWidget(self.start_button)

        game_whack_a_mole.setLayout(layout)
         # Add the second interface to the stacked widget
        self.stacked_widget.addWidget(game_whack_a_mole)

    def show_main(self):
        self.stacked_widget.setCurrentIndex(0)
        self.Home.hide()
        self.change_to_mole.show()
        self.exit.show()
        self.det.show()
        self.talk.show()
        self.intelligent_pet.show()

    def show_game_whack_a_mole(self):
        self.stacked_widget.setCurrentIndex(1)
                
        self.Home.show()
        self.change_to_mole.hide()
        self.exit.hide()
        self.det.hide()
        self.talk.hide()
        self.intelligent_pet.hide()

    def show_pet(self):
        self.stacked_widget.setCurrentIndex(2)
                
    
        self.Home.show()
        self.change_to_mole.hide()
        self.exit.hide()
        self.det.hide()
        self.intelligent_pet.hide()
        self.talk.hide()
    
    def show_det(self):
        self.stacked_widget.setCurrentIndex(3)
        self.det.hide()
      
        self.Home.show()
        self.change_to_mole.hide()
        self.exit.hide()
        self.intelligent_pet.hide()
        self.talk.hide()
    
    def show_talk(self):
        self.stacked_widget.setCurrentIndex(4)
        self.det.hide()
      
        self.Home.show()
        self.change_to_mole.hide()
        self.exit.hide()
        self.intelligent_pet.hide()
        self.talk.hide()

    def start_game(self):
        game_time = self.time_spinbox.value()
        difficulty = self.difficulty_combo.currentText()

        difficulty_map = {
            "简单": 1.5,
            "中等": 0.9,
            "困难": 0.3,
        }

        if difficulty in difficulty_map:
            mole_interval = difficulty_map[difficulty]
            self.run_game(game_time, mole_interval)
        else:
            QMessageBox.warning(self, '输入错误', '需要按照规定输入游戏难度！')

    def run_game(self, game_time, mole_interval):
        # 创建并运行游戏实例
        game = threading.Thread(target=self.continue_run, args=(game_time, mole_interval))
        game.start()

    def continue_run(self, game_time, mole_interval):
        game = mole.WhackAMoleGame(game_time, mole_interval)
        game.run()
        self.show_game_whack_a_mole()
        self.game_finished.emit(game.score)
        
    def update_score(self, score):
        # 更新得分标签
        self.score_label.setText(f'您的得分是: {score}')

    def adjust_font_size(self, button):
        font = button.font()
        font_metrics = QFontMetrics(font)
        text_size = font_metrics.size(0, button.text())
        button_size = button.size()

        while text_size.width() > button_size.width() or text_size.height() > button_size.height():
            font.setPointSize(font.pointSize() - 1)
            font_metrics = QFontMetrics(font)
            text_size = font_metrics.size(0, button.text())

        button.setFont(font)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())