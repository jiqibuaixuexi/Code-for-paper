# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AS_predict_app.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(805, 834)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.pbPredict = QPushButton(self.centralwidget)
        self.pbPredict.setObjectName(u"pbPredict")
        self.pbPredict.setGeometry(QRect(310, 710, 75, 24))
        self.cbModelArchitecture = QComboBox(self.centralwidget)
        self.cbModelArchitecture.setObjectName(u"cbModelArchitecture")
        self.cbModelArchitecture.setGeometry(QRect(70, 710, 141, 22))
        self.cbModelArchitecture.setEditable(True)
        self.lbShowImg = QLabel(self.centralwidget)
        self.lbShowImg.setObjectName(u"lbShowImg")
        self.lbShowImg.setGeometry(QRect(120, 50, 611, 541))
        self.lbShowImg.setAlignment(Qt.AlignCenter)
        self.pbSelectImg = QPushButton(self.centralwidget)
        self.pbSelectImg.setObjectName(u"pbSelectImg")
        self.pbSelectImg.setGeometry(QRect(70, 610, 91, 24))
        self.lbModelArchitecture = QLabel(self.centralwidget)
        self.lbModelArchitecture.setObjectName(u"lbModelArchitecture")
        self.lbModelArchitecture.setGeometry(QRect(70, 690, 141, 16))
        self.cbModelWeight = QComboBox(self.centralwidget)
        self.cbModelWeight.setObjectName(u"cbModelWeight")
        self.cbModelWeight.setGeometry(QRect(70, 760, 141, 22))
        self.lbModelWeight = QLabel(self.centralwidget)
        self.lbModelWeight.setObjectName(u"lbModelWeight")
        self.lbModelWeight.setGeometry(QRect(70, 740, 141, 16))
        self.lbShowPredict = QLabel(self.centralwidget)
        self.lbShowPredict.setObjectName(u"lbShowPredict")
        self.lbShowPredict.setGeometry(QRect(130, 0, 551, 51))
        font = QFont()
        font.setFamilies([u"Times New Roman"])
        font.setPointSize(16)
        self.lbShowPredict.setFont(font)
        self.lbShowPredict.setTextFormat(Qt.AutoText)
        self.lbShowPredict.setAlignment(Qt.AlignCenter)
        self.pbGradCAM = QPushButton(self.centralwidget)
        self.pbGradCAM.setObjectName(u"pbGradCAM")
        self.pbGradCAM.setGeometry(QRect(530, 710, 121, 24))
        MainWindow.setCentralWidget(self.centralwidget)
        self.lbShowImg.raise_()
        self.pbPredict.raise_()
        self.cbModelArchitecture.raise_()
        self.pbSelectImg.raise_()
        self.lbModelArchitecture.raise_()
        self.cbModelWeight.raise_()
        self.lbModelWeight.raise_()
        self.lbShowPredict.raise_()
        self.pbGradCAM.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 805, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"AS Predictor", None))
        self.pbPredict.setText(QCoreApplication.translate("MainWindow", u"predict", None))
        self.cbModelArchitecture.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6a21\u578b\u67b6\u6784", None))
        self.lbShowImg.setText(QCoreApplication.translate("MainWindow", u"Please upload a picture, preferably in the format of .jpg .jpeg *.png.", None))
        self.pbSelectImg.setText(QCoreApplication.translate("MainWindow", u"Open Image", None))
        self.lbModelArchitecture.setText(QCoreApplication.translate("MainWindow", u"model", None))
        self.lbModelWeight.setText(QCoreApplication.translate("MainWindow", u"checkpoint", None))
        self.lbShowPredict.setText(QCoreApplication.translate("MainWindow", u"Please click the predict button", None))
        self.pbGradCAM.setText(QCoreApplication.translate("MainWindow", u"Show GradCAM", None))
    # retranslateUi

