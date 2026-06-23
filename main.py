import sys
import torch  

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app.main_window import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setApplicationName("PotHole Detector")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
