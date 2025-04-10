import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from app.gui.measurement_settings_window import MeasurementSettingsWindow

def main():
    app = QApplication(sys.argv)
    window = MeasurementSettingsWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 