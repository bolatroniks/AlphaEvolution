# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1460, 742)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(340, 0, 1011, 711))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.graphicsView_plot1 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_plot1.setGeometry(QtCore.QRect(-10, 0, 491, 361))
        self.graphicsView_plot1.setObjectName(_fromUtf8("graphicsView_plot1"))
        self.graphicsView_plot2 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_plot2.setGeometry(QtCore.QRect(490, 0, 521, 361))
        self.graphicsView_plot2.setObjectName(_fromUtf8("graphicsView_plot2"))
        self.graphicsView_plot3 = QtGui.QGraphicsView(self.frame_2)
        self.graphicsView_plot3.setGeometry(QtCore.QRect(0, 360, 1011, 261))
        self.graphicsView_plot3.setObjectName(_fromUtf8("graphicsView_plot3"))
        self.plainTextEdit_output = QtGui.QPlainTextEdit(self.frame_2)
        self.plainTextEdit_output.setGeometry(QtCore.QRect(0, 370, 1011, 241))
        self.plainTextEdit_output.setObjectName(_fromUtf8("plainTextEdit_output"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 331, 701))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.frame_4 = QtGui.QFrame(self.tab)
        self.frame_4.setGeometry(QtCore.QRect(-1, -1, 361, 631))
        self.frame_4.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_4.setObjectName(_fromUtf8("frame_4"))
        self.frame = QtGui.QFrame(self.frame_4)
        self.frame.setGeometry(QtCore.QRect(0, 0, 361, 251))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.ComboInstrument = QtGui.QComboBox(self.frame)
        self.ComboInstrument.setGeometry(QtCore.QRect(130, 10, 111, 27))
        self.ComboInstrument.setObjectName(_fromUtf8("ComboInstrument"))
        self.btnInstrumentLoader = QtGui.QPushButton(self.frame)
        self.btnInstrumentLoader.setGeometry(QtCore.QRect(190, 220, 99, 27))
        self.btnInstrumentLoader.setObjectName(_fromUtf8("btnInstrumentLoader"))
        self.label = QtGui.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(30, 10, 91, 31))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(30, 50, 68, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.dateTimeEdit_from = QtGui.QDateTimeEdit(self.frame)
        self.dateTimeEdit_from.setGeometry(QtCore.QRect(90, 40, 194, 27))
        self.dateTimeEdit_from.setObjectName(_fromUtf8("dateTimeEdit_from"))
        self.label_3 = QtGui.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(30, 80, 68, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.dateTimeEdit_to = QtGui.QDateTimeEdit(self.frame)
        self.dateTimeEdit_to.setGeometry(QtCore.QRect(90, 70, 194, 27))
        self.dateTimeEdit_to.setDate(QtCore.QDate(2015, 12, 31))
        self.dateTimeEdit_to.setTime(QtCore.QTime(23, 59, 0))
        self.dateTimeEdit_to.setObjectName(_fromUtf8("dateTimeEdit_to"))
        self.label_5 = QtGui.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(30, 110, 131, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_6 = QtGui.QLabel(self.frame)
        self.label_6.setGeometry(QtCore.QRect(30, 130, 131, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.combo_fastTimeframe = QtGui.QComboBox(self.frame)
        self.combo_fastTimeframe.setGeometry(QtCore.QRect(190, 110, 91, 25))
        self.combo_fastTimeframe.setEditable(True)
        self.combo_fastTimeframe.setObjectName(_fromUtf8("combo_fastTimeframe"))
        self.combo_slowTimeframe = QtGui.QComboBox(self.frame)
        self.combo_slowTimeframe.setGeometry(QtCore.QRect(190, 130, 91, 25))
        self.combo_slowTimeframe.setEditable(True)
        self.combo_slowTimeframe.setObjectName(_fromUtf8("combo_slowTimeframe"))
        self.label_7 = QtGui.QLabel(self.frame)
        self.label_7.setGeometry(QtCore.QRect(30, 160, 131, 17))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.spinBox_halflife_window = QtGui.QSpinBox(self.frame)
        self.spinBox_halflife_window.setGeometry(QtCore.QRect(100, 160, 61, 21))
        self.spinBox_halflife_window.setMinimum(0)
        self.spinBox_halflife_window.setMaximum(99999)
        self.spinBox_halflife_window.setSingleStep(25)
        self.spinBox_halflife_window.setProperty("value", 250)
        self.spinBox_halflife_window.setObjectName(_fromUtf8("spinBox_halflife_window"))
        self.label_8 = QtGui.QLabel(self.frame)
        self.label_8.setGeometry(QtCore.QRect(30, 190, 131, 17))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.spinBox_daily_delay = QtGui.QSpinBox(self.frame)
        self.spinBox_daily_delay.setGeometry(QtCore.QRect(110, 190, 61, 21))
        self.spinBox_daily_delay.setMinimum(-100)
        self.spinBox_daily_delay.setMaximum(100)
        self.spinBox_daily_delay.setSingleStep(1)
        self.spinBox_daily_delay.setProperty("value", 5)
        self.spinBox_daily_delay.setObjectName(_fromUtf8("spinBox_daily_delay"))
        self.btnInstrumentClear = QtGui.QPushButton(self.frame)
        self.btnInstrumentClear.setGeometry(QtCore.QRect(10, 220, 99, 27))
        self.btnInstrumentClear.setObjectName(_fromUtf8("btnInstrumentClear"))
        self.frame_7 = QtGui.QFrame(self.frame_4)
        self.frame_7.setGeometry(QtCore.QRect(0, 250, 331, 121))
        self.frame_7.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_7.setObjectName(_fromUtf8("frame_7"))
        self.pushButton_relabel = QtGui.QPushButton(self.frame_7)
        self.pushButton_relabel.setGeometry(QtCore.QRect(80, 80, 99, 27))
        self.pushButton_relabel.setObjectName(_fromUtf8("pushButton_relabel"))
        self.label_39 = QtGui.QLabel(self.frame_7)
        self.label_39.setGeometry(QtCore.QRect(30, 30, 68, 17))
        self.label_39.setObjectName(_fromUtf8("label_39"))
        self.label_40 = QtGui.QLabel(self.frame_7)
        self.label_40.setGeometry(QtCore.QRect(30, 10, 111, 17))
        self.label_40.setObjectName(_fromUtf8("label_40"))
        self.doubleSpinBox_minStop = QtGui.QDoubleSpinBox(self.frame_7)
        self.doubleSpinBox_minStop.setGeometry(QtCore.QRect(170, 30, 81, 21))
        self.doubleSpinBox_minStop.setDecimals(4)
        self.doubleSpinBox_minStop.setMaximum(1.0)
        self.doubleSpinBox_minStop.setSingleStep(0.001)
        self.doubleSpinBox_minStop.setProperty("value", 0.005)
        self.doubleSpinBox_minStop.setObjectName(_fromUtf8("doubleSpinBox_minStop"))
        self.label_41 = QtGui.QLabel(self.frame_7)
        self.label_41.setGeometry(QtCore.QRect(30, 50, 121, 17))
        self.label_41.setObjectName(_fromUtf8("label_41"))
        self.doubleSpinBox_volDenominator = QtGui.QDoubleSpinBox(self.frame_7)
        self.doubleSpinBox_volDenominator.setGeometry(QtCore.QRect(170, 50, 81, 21))
        self.doubleSpinBox_volDenominator.setDecimals(1)
        self.doubleSpinBox_volDenominator.setMinimum(-20.0)
        self.doubleSpinBox_volDenominator.setMaximum(20.0)
        self.doubleSpinBox_volDenominator.setProperty("value", 5.0)
        self.doubleSpinBox_volDenominator.setObjectName(_fromUtf8("doubleSpinBox_volDenominator"))
        self.doubleSpinBox_targetMultiple = QtGui.QDoubleSpinBox(self.frame_7)
        self.doubleSpinBox_targetMultiple.setGeometry(QtCore.QRect(170, 10, 81, 21))
        self.doubleSpinBox_targetMultiple.setMaximum(10.0)
        self.doubleSpinBox_targetMultiple.setSingleStep(0.25)
        self.doubleSpinBox_targetMultiple.setProperty("value", 1.0)
        self.doubleSpinBox_targetMultiple.setObjectName(_fromUtf8("doubleSpinBox_targetMultiple"))
        self.frame_8 = QtGui.QFrame(self.frame_4)
        self.frame_8.setGeometry(QtCore.QRect(-1, 369, 331, 171))
        self.frame_8.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_8.setObjectName(_fromUtf8("frame_8"))
        self.comboBox_indicatorToVisualize = QtGui.QComboBox(self.frame_8)
        self.comboBox_indicatorToVisualize.setGeometry(QtCore.QRect(130, 20, 171, 27))
        self.comboBox_indicatorToVisualize.setObjectName(_fromUtf8("comboBox_indicatorToVisualize"))
        self.label_42 = QtGui.QLabel(self.frame_8)
        self.label_42.setGeometry(QtCore.QRect(30, 20, 91, 31))
        self.label_42.setObjectName(_fromUtf8("label_42"))
        self.pushButton_visualize = QtGui.QPushButton(self.frame_8)
        self.pushButton_visualize.setGeometry(QtCore.QRect(100, 130, 99, 27))
        self.pushButton_visualize.setObjectName(_fromUtf8("pushButton_visualize"))
        self.label_43 = QtGui.QLabel(self.frame_8)
        self.label_43.setGeometry(QtCore.QRect(30, 50, 91, 31))
        self.label_43.setObjectName(_fromUtf8("label_43"))
        self.comboBox_versusIndicatorToVisualize = QtGui.QComboBox(self.frame_8)
        self.comboBox_versusIndicatorToVisualize.setGeometry(QtCore.QRect(130, 50, 171, 27))
        self.comboBox_versusIndicatorToVisualize.setObjectName(_fromUtf8("comboBox_versusIndicatorToVisualize"))
        self.comboBox_targetToVisualize = QtGui.QComboBox(self.frame_8)
        self.comboBox_targetToVisualize.setGeometry(QtCore.QRect(130, 80, 71, 27))
        self.comboBox_targetToVisualize.setObjectName(_fromUtf8("comboBox_targetToVisualize"))
        self.label_44 = QtGui.QLabel(self.frame_8)
        self.label_44.setGeometry(QtCore.QRect(30, 80, 91, 31))
        self.label_44.setObjectName(_fromUtf8("label_44"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.frame_6 = QtGui.QFrame(self.tab_2)
        self.frame_6.setGeometry(QtCore.QRect(0, 0, 331, 631))
        self.frame_6.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_6.setObjectName(_fromUtf8("frame_6"))
        self.btnStrategyRunner = QtGui.QPushButton(self.frame_6)
        self.btnStrategyRunner.setGeometry(QtCore.QRect(210, 510, 99, 27))
        self.btnStrategyRunner.setObjectName(_fromUtf8("btnStrategyRunner"))
        self.label_11 = QtGui.QLabel(self.frame_6)
        self.label_11.setGeometry(QtCore.QRect(20, 20, 68, 17))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.label_12 = QtGui.QLabel(self.frame_6)
        self.label_12.setGeometry(QtCore.QRect(20, 80, 68, 17))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.label_13 = QtGui.QLabel(self.frame_6)
        self.label_13.setGeometry(QtCore.QRect(20, 100, 68, 17))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.label_14 = QtGui.QLabel(self.frame_6)
        self.label_14.setGeometry(QtCore.QRect(20, 120, 71, 17))
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.label_15 = QtGui.QLabel(self.frame_6)
        self.label_15.setGeometry(QtCore.QRect(20, 140, 68, 17))
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.label_16 = QtGui.QLabel(self.frame_6)
        self.label_16.setGeometry(QtCore.QRect(20, 160, 81, 17))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.label_17 = QtGui.QLabel(self.frame_6)
        self.label_17.setGeometry(QtCore.QRect(20, 180, 91, 17))
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.textEdit_function = QtGui.QTextEdit(self.frame_6)
        self.textEdit_function.setGeometry(QtCore.QRect(100, 10, 211, 31))
        self.textEdit_function.setObjectName(_fromUtf8("textEdit_function"))
        self.doubleSpinBox_rhoMin = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_rhoMin.setGeometry(QtCore.QRect(160, 80, 62, 21))
        self.doubleSpinBox_rhoMin.setMaximum(1.0)
        self.doubleSpinBox_rhoMin.setSingleStep(0.05)
        self.doubleSpinBox_rhoMin.setProperty("value", 0.3)
        self.doubleSpinBox_rhoMin.setObjectName(_fromUtf8("doubleSpinBox_rhoMin"))
        self.doubleSpinBox_rhoMax = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_rhoMax.setGeometry(QtCore.QRect(160, 100, 62, 21))
        self.doubleSpinBox_rhoMax.setMaximum(1.0)
        self.doubleSpinBox_rhoMax.setSingleStep(0.05)
        self.doubleSpinBox_rhoMax.setProperty("value", 0.7)
        self.doubleSpinBox_rhoMax.setObjectName(_fromUtf8("doubleSpinBox_rhoMax"))
        self.doubleSpinBox_residMin = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_residMin.setGeometry(QtCore.QRect(160, 120, 62, 21))
        self.doubleSpinBox_residMin.setDecimals(1)
        self.doubleSpinBox_residMin.setMinimum(-99.0)
        self.doubleSpinBox_residMin.setMaximum(99.0)
        self.doubleSpinBox_residMin.setProperty("value", 5.0)
        self.doubleSpinBox_residMin.setObjectName(_fromUtf8("doubleSpinBox_residMin"))
        self.doubleSpinBox_residMax = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_residMax.setGeometry(QtCore.QRect(160, 140, 62, 21))
        self.doubleSpinBox_residMax.setDecimals(1)
        self.doubleSpinBox_residMax.setMinimum(-99.0)
        self.doubleSpinBox_residMax.setMaximum(99.0)
        self.doubleSpinBox_residMax.setProperty("value", 10.0)
        self.doubleSpinBox_residMax.setObjectName(_fromUtf8("doubleSpinBox_residMax"))
        self.doubleSpinBox_rsiFastMin = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_rsiFastMin.setGeometry(QtCore.QRect(160, 160, 62, 21))
        self.doubleSpinBox_rsiFastMin.setDecimals(1)
        self.doubleSpinBox_rsiFastMin.setMaximum(100.0)
        self.doubleSpinBox_rsiFastMin.setSingleStep(2.5)
        self.doubleSpinBox_rsiFastMin.setProperty("value", 30.0)
        self.doubleSpinBox_rsiFastMin.setObjectName(_fromUtf8("doubleSpinBox_rsiFastMin"))
        self.doubleSpinBox_rsiFastMax = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_rsiFastMax.setGeometry(QtCore.QRect(160, 180, 62, 21))
        self.doubleSpinBox_rsiFastMax.setDecimals(1)
        self.doubleSpinBox_rsiFastMax.setSingleStep(2.5)
        self.doubleSpinBox_rsiFastMax.setProperty("value", 70.0)
        self.doubleSpinBox_rsiFastMax.setObjectName(_fromUtf8("doubleSpinBox_rsiFastMax"))
        self.doubleSpinBox_rsiSlowMin = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_rsiSlowMin.setGeometry(QtCore.QRect(160, 200, 62, 21))
        self.doubleSpinBox_rsiSlowMin.setDecimals(1)
        self.doubleSpinBox_rsiSlowMin.setMaximum(100.0)
        self.doubleSpinBox_rsiSlowMin.setSingleStep(2.5)
        self.doubleSpinBox_rsiSlowMin.setProperty("value", 30.0)
        self.doubleSpinBox_rsiSlowMin.setObjectName(_fromUtf8("doubleSpinBox_rsiSlowMin"))
        self.label_18 = QtGui.QLabel(self.frame_6)
        self.label_18.setGeometry(QtCore.QRect(20, 200, 101, 17))
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.doubleSpinBox_rsiSlowMax = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_rsiSlowMax.setGeometry(QtCore.QRect(160, 220, 62, 21))
        self.doubleSpinBox_rsiSlowMax.setDecimals(1)
        self.doubleSpinBox_rsiSlowMax.setSingleStep(2.5)
        self.doubleSpinBox_rsiSlowMax.setProperty("value", 70.0)
        self.doubleSpinBox_rsiSlowMax.setObjectName(_fromUtf8("doubleSpinBox_rsiSlowMax"))
        self.label_19 = QtGui.QLabel(self.frame_6)
        self.label_19.setGeometry(QtCore.QRect(20, 220, 91, 17))
        self.label_19.setObjectName(_fromUtf8("label_19"))
        self.label_20 = QtGui.QLabel(self.frame_6)
        self.label_20.setGeometry(QtCore.QRect(20, 240, 131, 17))
        self.label_20.setObjectName(_fromUtf8("label_20"))
        self.label_21 = QtGui.QLabel(self.frame_6)
        self.label_21.setGeometry(QtCore.QRect(20, 260, 131, 17))
        self.label_21.setObjectName(_fromUtf8("label_21"))
        self.label_22 = QtGui.QLabel(self.frame_6)
        self.label_22.setGeometry(QtCore.QRect(20, 300, 151, 17))
        self.label_22.setObjectName(_fromUtf8("label_22"))
        self.label_23 = QtGui.QLabel(self.frame_6)
        self.label_23.setGeometry(QtCore.QRect(20, 280, 141, 17))
        self.label_23.setObjectName(_fromUtf8("label_23"))
        self.spinBox_netTrendlinesMin = QtGui.QSpinBox(self.frame_6)
        self.spinBox_netTrendlinesMin.setGeometry(QtCore.QRect(160, 240, 61, 21))
        self.spinBox_netTrendlinesMin.setMinimum(-999)
        self.spinBox_netTrendlinesMin.setMaximum(999)
        self.spinBox_netTrendlinesMin.setProperty("value", 3)
        self.spinBox_netTrendlinesMin.setObjectName(_fromUtf8("spinBox_netTrendlinesMin"))
        self.spinBox_netTrendlinesMax = QtGui.QSpinBox(self.frame_6)
        self.spinBox_netTrendlinesMax.setGeometry(QtCore.QRect(160, 260, 61, 21))
        self.spinBox_netTrendlinesMax.setMinimum(-99)
        self.spinBox_netTrendlinesMax.setProperty("value", 10)
        self.spinBox_netTrendlinesMax.setObjectName(_fromUtf8("spinBox_netTrendlinesMax"))
        self.spinBox_trendlinesDeltaMin = QtGui.QSpinBox(self.frame_6)
        self.spinBox_trendlinesDeltaMin.setGeometry(QtCore.QRect(160, 280, 61, 21))
        self.spinBox_trendlinesDeltaMin.setMinimum(-99)
        self.spinBox_trendlinesDeltaMin.setProperty("value", 3)
        self.spinBox_trendlinesDeltaMin.setObjectName(_fromUtf8("spinBox_trendlinesDeltaMin"))
        self.spinBox_trendlinesDeltaMax = QtGui.QSpinBox(self.frame_6)
        self.spinBox_trendlinesDeltaMax.setGeometry(QtCore.QRect(160, 300, 61, 21))
        self.spinBox_trendlinesDeltaMax.setMinimum(-99)
        self.spinBox_trendlinesDeltaMax.setProperty("value", 8)
        self.spinBox_trendlinesDeltaMax.setObjectName(_fromUtf8("spinBox_trendlinesDeltaMax"))
        self.label_24 = QtGui.QLabel(self.frame_6)
        self.label_24.setGeometry(QtCore.QRect(20, 50, 68, 17))
        self.label_24.setObjectName(_fromUtf8("label_24"))
        self.comboBox_criterium = QtGui.QComboBox(self.frame_6)
        self.comboBox_criterium.setGeometry(QtCore.QRect(100, 40, 85, 27))
        self.comboBox_criterium.setObjectName(_fromUtf8("comboBox_criterium"))
        self.label_46 = QtGui.QLabel(self.frame_6)
        self.label_46.setGeometry(QtCore.QRect(20, 460, 181, 17))
        self.label_46.setObjectName(_fromUtf8("label_46"))
        self.spinBox_serialGap = QtGui.QSpinBox(self.frame_6)
        self.spinBox_serialGap.setGeometry(QtCore.QRect(200, 460, 61, 21))
        self.spinBox_serialGap.setMinimum(0)
        self.spinBox_serialGap.setMaximum(99999)
        self.spinBox_serialGap.setSingleStep(25)
        self.spinBox_serialGap.setProperty("value", 0)
        self.spinBox_serialGap.setObjectName(_fromUtf8("spinBox_serialGap"))
        self.pushButton_saveStrategy = QtGui.QPushButton(self.frame_6)
        self.pushButton_saveStrategy.setGeometry(QtCore.QRect(10, 510, 99, 27))
        self.pushButton_saveStrategy.setObjectName(_fromUtf8("pushButton_saveStrategy"))
        self.pushButton_loadStrategy = QtGui.QPushButton(self.frame_6)
        self.pushButton_loadStrategy.setGeometry(QtCore.QRect(110, 510, 99, 27))
        self.pushButton_loadStrategy.setObjectName(_fromUtf8("pushButton_loadStrategy"))
        self.comboBox_strategyFilename = QtGui.QComboBox(self.frame_6)
        self.comboBox_strategyFilename.setGeometry(QtCore.QRect(20, 480, 241, 27))
        self.comboBox_strategyFilename.setEditable(True)
        self.comboBox_strategyFilename.setObjectName(_fromUtf8("comboBox_strategyFilename"))
        self.label_25 = QtGui.QLabel(self.frame_6)
        self.label_25.setGeometry(QtCore.QRect(20, 340, 151, 17))
        self.label_25.setObjectName(_fromUtf8("label_25"))
        self.spinBox_halflifeMin = QtGui.QSpinBox(self.frame_6)
        self.spinBox_halflifeMin.setGeometry(QtCore.QRect(150, 320, 71, 21))
        self.spinBox_halflifeMin.setMinimum(-10000)
        self.spinBox_halflifeMin.setMaximum(10000)
        self.spinBox_halflifeMin.setSingleStep(25)
        self.spinBox_halflifeMin.setProperty("value", 150)
        self.spinBox_halflifeMin.setObjectName(_fromUtf8("spinBox_halflifeMin"))
        self.label_26 = QtGui.QLabel(self.frame_6)
        self.label_26.setGeometry(QtCore.QRect(20, 320, 141, 17))
        self.label_26.setObjectName(_fromUtf8("label_26"))
        self.spinBox_halflifeMax = QtGui.QSpinBox(self.frame_6)
        self.spinBox_halflifeMax.setGeometry(QtCore.QRect(150, 340, 71, 21))
        self.spinBox_halflifeMax.setMinimum(-10000)
        self.spinBox_halflifeMax.setMaximum(10000)
        self.spinBox_halflifeMax.setSingleStep(25)
        self.spinBox_halflifeMax.setProperty("value", 10000)
        self.spinBox_halflifeMax.setObjectName(_fromUtf8("spinBox_halflifeMax"))
        self.label_27 = QtGui.QLabel(self.frame_6)
        self.label_27.setGeometry(QtCore.QRect(20, 360, 141, 17))
        self.label_27.setObjectName(_fromUtf8("label_27"))
        self.label_28 = QtGui.QLabel(self.frame_6)
        self.label_28.setGeometry(QtCore.QRect(20, 380, 141, 17))
        self.label_28.setObjectName(_fromUtf8("label_28"))
        self.doubleSpinBox_closeOverMAmax = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_closeOverMAmax.setGeometry(QtCore.QRect(160, 380, 62, 21))
        self.doubleSpinBox_closeOverMAmax.setMinimum(-10.0)
        self.doubleSpinBox_closeOverMAmax.setMaximum(10.0)
        self.doubleSpinBox_closeOverMAmax.setSingleStep(0.5)
        self.doubleSpinBox_closeOverMAmax.setProperty("value", 1.0)
        self.doubleSpinBox_closeOverMAmax.setObjectName(_fromUtf8("doubleSpinBox_closeOverMAmax"))
        self.doubleSpinBox_closeOverMAmin = QtGui.QDoubleSpinBox(self.frame_6)
        self.doubleSpinBox_closeOverMAmin.setGeometry(QtCore.QRect(160, 360, 62, 21))
        self.doubleSpinBox_closeOverMAmin.setMinimum(-10.0)
        self.doubleSpinBox_closeOverMAmin.setMaximum(10.0)
        self.doubleSpinBox_closeOverMAmin.setSingleStep(0.5)
        self.doubleSpinBox_closeOverMAmin.setProperty("value", 1.0)
        self.doubleSpinBox_closeOverMAmin.setObjectName(_fromUtf8("doubleSpinBox_closeOverMAmin"))
        self.label_30 = QtGui.QLabel(self.frame_6)
        self.label_30.setGeometry(QtCore.QRect(20, 410, 191, 31))
        self.label_30.setObjectName(_fromUtf8("label_30"))
        self.checkBox_avoidOBFast = QtGui.QCheckBox(self.frame_6)
        self.checkBox_avoidOBFast.setGeometry(QtCore.QRect(150, 400, 161, 22))
        self.checkBox_avoidOBFast.setObjectName(_fromUtf8("checkBox_avoidOBFast"))
        self.checkBox_avoidOBSlow = QtGui.QCheckBox(self.frame_6)
        self.checkBox_avoidOBSlow.setGeometry(QtCore.QRect(30, 400, 131, 22))
        self.checkBox_avoidOBSlow.setObjectName(_fromUtf8("checkBox_avoidOBSlow"))
        self.spinBox_OBSlowWindow = QtGui.QSpinBox(self.frame_6)
        self.spinBox_OBSlowWindow.setGeometry(QtCore.QRect(210, 420, 61, 21))
        self.spinBox_OBSlowWindow.setMinimum(1)
        self.spinBox_OBSlowWindow.setMaximum(252)
        self.spinBox_OBSlowWindow.setProperty("value", 14)
        self.spinBox_OBSlowWindow.setObjectName(_fromUtf8("spinBox_OBSlowWindow"))
        self.label_31 = QtGui.QLabel(self.frame_6)
        self.label_31.setGeometry(QtCore.QRect(20, 430, 191, 31))
        self.label_31.setObjectName(_fromUtf8("label_31"))
        self.spinBox_OBFastWindow = QtGui.QSpinBox(self.frame_6)
        self.spinBox_OBFastWindow.setGeometry(QtCore.QRect(210, 440, 61, 21))
        self.spinBox_OBFastWindow.setMinimum(1)
        self.spinBox_OBFastWindow.setMaximum(252)
        self.spinBox_OBFastWindow.setProperty("value", 14)
        self.spinBox_OBFastWindow.setObjectName(_fromUtf8("spinBox_OBFastWindow"))
        self.checkBox_plotPnLSum = QtGui.QCheckBox(self.frame_6)
        self.checkBox_plotPnLSum.setGeometry(QtCore.QRect(20, 550, 131, 22))
        self.checkBox_plotPnLSum.setObjectName(_fromUtf8("checkBox_plotPnLSum"))
        self.ComboPCAChoice = QtGui.QComboBox(self.frame_6)
        self.ComboPCAChoice.setGeometry(QtCore.QRect(220, 80, 61, 27))
        self.ComboPCAChoice.setMaxVisibleItems(3)
        self.ComboPCAChoice.setObjectName(_fromUtf8("ComboPCAChoice"))
        self.ComboPCAChoice.addItem(_fromUtf8(""))
        self.ComboPCAChoice.addItem(_fromUtf8(""))
        self.ComboPCAChoice.addItem(_fromUtf8(""))
        self.frame_9 = QtGui.QFrame(self.tab_2)
        self.frame_9.setGeometry(QtCore.QRect(-1, 629, 331, 41))
        self.frame_9.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_9.setObjectName(_fromUtf8("frame_9"))
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.tab_5 = QtGui.QWidget()
        self.tab_5.setObjectName(_fromUtf8("tab_5"))
        self.label_45 = QtGui.QLabel(self.tab_5)
        self.label_45.setGeometry(QtCore.QRect(17, 16, 301, 551))
        self.label_45.setObjectName(_fromUtf8("label_45"))
        self.tabWidget.addTab(self.tab_5, _fromUtf8(""))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1460, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.ComboPCAChoice.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.btnInstrumentLoader.setText(_translate("MainWindow", "Load", None))
        self.label.setText(_translate("MainWindow", "Instrument", None))
        self.label_2.setText(_translate("MainWindow", "From", None))
        self.label_3.setText(_translate("MainWindow", "To", None))
        self.label_5.setText(_translate("MainWindow", "Trading Timeframe", None))
        self.label_6.setText(_translate("MainWindow", "Slow Timeframe", None))
        self.label_7.setText(_translate("MainWindow", "Half Life", None))
        self.label_8.setText(_translate("MainWindow", "Daily delay", None))
        self.btnInstrumentClear.setText(_translate("MainWindow", "Clear", None))
        self.pushButton_relabel.setText(_translate("MainWindow", "Relabel", None))
        self.label_39.setText(_translate("MainWindow", "Min stop", None))
        self.label_40.setText(_translate("MainWindow", "Target Multiple", None))
        self.label_41.setText(_translate("MainWindow", "Vol denominator", None))
        self.label_42.setText(_translate("MainWindow", "Indicator", None))
        self.pushButton_visualize.setText(_translate("MainWindow", "Visualize", None))
        self.label_43.setText(_translate("MainWindow", "Versus", None))
        self.label_44.setText(_translate("MainWindow", "Target", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Single Pair", None))
        self.btnStrategyRunner.setText(_translate("MainWindow", "Run", None))
        self.label_11.setText(_translate("MainWindow", "Function", None))
        self.label_12.setText(_translate("MainWindow", "Rho min", None))
        self.label_13.setText(_translate("MainWindow", "Rho max", None))
        self.label_14.setText(_translate("MainWindow", "Resid min", None))
        self.label_15.setText(_translate("MainWindow", "Resid max", None))
        self.label_16.setText(_translate("MainWindow", "RSI fast min", None))
        self.label_17.setText(_translate("MainWindow", "RSI fast max", None))
        self.textEdit_function.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">compute_predictions_pca</p></body></html>", None))
        self.label_18.setText(_translate("MainWindow", "RSI slow min", None))
        self.label_19.setText(_translate("MainWindow", "RSI slow max", None))
        self.label_20.setText(_translate("MainWindow", "Net trendlines min", None))
        self.label_21.setText(_translate("MainWindow", "Net trendlines max", None))
        self.label_22.setText(_translate("MainWindow", "Delta trendlines max", None))
        self.label_23.setText(_translate("MainWindow", "Delta trendlines min", None))
        self.label_24.setText(_translate("MainWindow", "Criterium", None))
        self.label_46.setText(_translate("MainWindow", "Remove serial predictions", None))
        self.pushButton_saveStrategy.setText(_translate("MainWindow", "Save", None))
        self.pushButton_loadStrategy.setText(_translate("MainWindow", "Load", None))
        self.label_25.setText(_translate("MainWindow", "Halflife max", None))
        self.label_26.setText(_translate("MainWindow", "Halflife min", None))
        self.label_27.setText(_translate("MainWindow", "Close_over_MA_min", None))
        self.label_28.setText(_translate("MainWindow", "Close_over_MA_max", None))
        self.label_30.setText(_translate("MainWindow", "OB Slow window", None))
        self.checkBox_avoidOBFast.setText(_translate("MainWindow", "Avoid OB Fast", None))
        self.checkBox_avoidOBSlow.setText(_translate("MainWindow", "Avoid OB slow", None))
        self.label_31.setText(_translate("MainWindow", "OB Fast window", None))
        self.checkBox_plotPnLSum.setText(_translate("MainWindow", "Plot PnL Sum", None))
        self.ComboPCAChoice.setItemText(0, _translate("MainWindow", "1", None))
        self.ComboPCAChoice.setItemText(1, _translate("MainWindow", "2", None))
        self.ComboPCAChoice.setItemText(2, _translate("MainWindow", "3", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Strategy", None))
        self.label_45.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">ToDo:</span></p><p><span style=\" font-size:8pt;\">Features:</span></p><p><span style=\" font-size:8pt;\">- add mean reversion features (hurst exponent/lambda)</span></p><p><span style=\" font-size:8pt;\">- add reversal criteria</span></p><p><span style=\" font-size:8pt;\">- bollinger bands</span></p><p><span style=\" font-size:8pt;\">- parabolic</span></p><p><span style=\" font-size:8pt;\">Plots:</span></p><p><span style=\" font-size:8pt;\">- add statistics</span></p><p><span style=\" font-size:8pt;\">- add tabs to plots</span></p><p><span style=\" font-size:8pt;\">- make plots resizable</span></p><p><span style=\" font-size:8pt;\">- add possibility to plot PnL, Labels, signals, on top of features</span></p><p><span style=\" font-size:8pt;\">- plot features vs labels (scatter, correlation or smtg)</span></p><p><span style=\" font-size:8pt;\">ML:</span></p><p><span style=\" font-size:8pt;\">- auto-encode candles + volume</span></p><p><span style=\" font-size:8pt;\">Strategy:</span></p><p><span style=\" font-size:8pt;\">- add trading time restrictions</span></p><p><span style=\" font-size:8pt;\">- re-create bars based on volume before computing features (what is the volume data provided by Oanda?)</span></p><p><span style=\" font-size:8pt;\">- save config</span></p><p><span style=\" font-size:8pt;\">- load config</span></p></body></html>", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "ToDo", None))

