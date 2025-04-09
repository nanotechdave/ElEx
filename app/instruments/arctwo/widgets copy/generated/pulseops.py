# Form implementation generated from reading ui file 'C:\Users\mcfab\AppData\Local\Temp\pip-req-build-8ggypqms\uis\pulseops.ui'
#
# Created by: PyQt6 UI code generator 6.4.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_PulseOpsWidget(object):
    def setupUi(self, PulseOpsWidget):
        PulseOpsWidget.setObjectName("PulseOpsWidget")
        PulseOpsWidget.resize(253, 107)
        self.gridLayout = QtWidgets.QGridLayout(PulseOpsWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.positivePulseSpinBox = QtWidgets.QDoubleSpinBox(PulseOpsWidget)
        self.positivePulseSpinBox.setDecimals(1)
        self.positivePulseSpinBox.setMaximum(10.0)
        self.positivePulseSpinBox.setSingleStep(0.1)
        self.positivePulseSpinBox.setProperty("value", 1.0)
        self.positivePulseSpinBox.setObjectName("positivePulseSpinBox")
        self.gridLayout.addWidget(self.positivePulseSpinBox, 0, 0, 1, 1)
        self.negativePulseSpinBox = QtWidgets.QDoubleSpinBox(PulseOpsWidget)
        self.negativePulseSpinBox.setDecimals(1)
        self.negativePulseSpinBox.setMaximum(10.0)
        self.negativePulseSpinBox.setSingleStep(0.1)
        self.negativePulseSpinBox.setProperty("value", 1.0)
        self.negativePulseSpinBox.setObjectName("negativePulseSpinBox")
        self.gridLayout.addWidget(self.negativePulseSpinBox, 0, 2, 1, 1)
        self.positiveDurationWidget = DurationWidget(PulseOpsWidget)
        self.positiveDurationWidget.setObjectName("positiveDurationWidget")
        self.gridLayout.addWidget(self.positiveDurationWidget, 1, 0, 1, 1)
        self.negativeDurationWidget = DurationWidget(PulseOpsWidget)
        self.negativeDurationWidget.setObjectName("negativeDurationWidget")
        self.gridLayout.addWidget(self.negativeDurationWidget, 1, 2, 1, 1)
        self.posPulseButton = QtWidgets.QPushButton(PulseOpsWidget)
        self.posPulseButton.setObjectName("posPulseButton")
        self.gridLayout.addWidget(self.posPulseButton, 2, 0, 1, 1)
        self.negPulseButton = QtWidgets.QPushButton(PulseOpsWidget)
        self.negPulseButton.setObjectName("negPulseButton")
        self.gridLayout.addWidget(self.negPulseButton, 2, 2, 1, 1)
        self.posPulseReadButton = QtWidgets.QPushButton(PulseOpsWidget)
        self.posPulseReadButton.setObjectName("posPulseReadButton")
        self.gridLayout.addWidget(self.posPulseReadButton, 3, 0, 1, 1)
        self.negPulseReadButton = QtWidgets.QPushButton(PulseOpsWidget)
        self.negPulseReadButton.setObjectName("negPulseReadButton")
        self.gridLayout.addWidget(self.negPulseReadButton, 3, 2, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(0, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.lockPulseCheckBox = QtWidgets.QCheckBox(PulseOpsWidget)
        self.lockPulseCheckBox.setObjectName("lockPulseCheckBox")
        self.horizontalLayout_2.addWidget(self.lockPulseCheckBox)
        spacerItem1 = QtWidgets.QSpacerItem(0, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)

        self.retranslateUi(PulseOpsWidget)
        QtCore.QMetaObject.connectSlotsByName(PulseOpsWidget)

    def retranslateUi(self, PulseOpsWidget):
        _translate = QtCore.QCoreApplication.translate
        PulseOpsWidget.setWindowTitle(_translate("PulseOpsWidget", "Form"))
        self.positivePulseSpinBox.setSuffix(_translate("PulseOpsWidget", " V"))
        self.negativePulseSpinBox.setSuffix(_translate("PulseOpsWidget", " V"))
        self.posPulseButton.setText(_translate("PulseOpsWidget", "Pulse +"))
        self.negPulseButton.setText(_translate("PulseOpsWidget", "Pulse –"))
        self.posPulseReadButton.setText(_translate("PulseOpsWidget", "Pulse Read +"))
        self.negPulseReadButton.setText(_translate("PulseOpsWidget", "Pulse Read –"))
        self.lockPulseCheckBox.setText(_translate("PulseOpsWidget", "Lock"))
from ..duration_widget import DurationWidget
