from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsPixmapItem, QVBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QByteArray
import sys
import tensorflow as tf
import csv
import numpy as np
import datetime
from pymongo import MongoClient
import gridfs
import codecs


class MushroomDb(object):
    DATABASE = None

    def __init__(self):
        self.fs = gridfs.GridFS(MushroomDb.DATABASE, 'mushroom_db')

    @staticmethod
    def initialize():
        client = MongoClient("localhost", 27017)
        MushroomDb.DATABASE = client['mushroom_db']
        gridfs.GridFS(MushroomDb.DATABASE, 'mushroom_db')
        print("Connection Successful")

    @staticmethod
    def insert(data):
        MushroomDb.DATABASE["test"].insert_one(data)

    @staticmethod
    def insert_fs(data, filename):
        gridfs.GridFS(MushroomDb.DATABASE).put(data, filename=filename)

    @staticmethod
    def find():
        return MushroomDb.DATABASE["test"].find()

    @staticmethod
    def find_fs():
        return gridfs.GridFS(MushroomDb.DATABASE).find()

    @staticmethod
    def find_one(query):
        return MushroomDb.DATABASE["test"].find_one(query)

    @staticmethod
    def find_one_fs(query):
        return gridfs.GridFS(MushroomDb.DATABASE).find_one(query)

    @staticmethod
    def drop_coll():
        MushroomDb.DATABASE["fs.chunks"].drop()
        MushroomDb.DATABASE["fs.files"].drop()
        MushroomDb.DATABASE["test"].drop()


class UiMainWindow(QMainWindow):

    def h5_load(self):
        self.model = tf.keras.models.load_model("ModelResNet50.h5")
        with open("classnames.txt") as load_file:
            reader = csv.reader(load_file, delimiter=",")
            self.class_names = [tuple(row) for row in reader]

    def h5_predict(self):
        self.image = tf.keras.preprocessing.image.load_img(self.imgname[0], target_size=(224, 224, 3))
        self.image_array = tf.keras.utils.img_to_array(self.image)
        self.image_array = tf.expand_dims(self.image_array, 0)  # Create a batch
        self.predictions = self.model.predict(self.image_array)
        self.score = tf.nn.softmax(self.predictions[0])
        self.class_index = self.class_names[0][tf.argmax(self.score)]
        self.percentage_score = 100 * np.max(self.score)
        self.textAnaliza.setText("This image most likely belongs to {} with a {:.2f} percent confidence."
                                 .format(self.class_index, self.percentage_score))

    def load(self):
        self.scene2 = QtWidgets.QGraphicsScene()
        self.imgname = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files (*)")
        self.image_qt = QPixmap(self.imgname[0])
        self.pic2_resized = self.image_qt.scaled(244, 244, Qt.AspectRatioMode.KeepAspectRatio)
        self.pic2 = QGraphicsPixmapItem(self.pic2_resized)
        self.scene2.addItem(self.pic2)
        self.graphicsViewWczytaj.setScene(self.scene2)

    def db_insert(self):
        time = datetime.datetime.utcnow()
        data = {"name": self.class_index, "score": self.percentage_score, "date": time}
        MushroomDb.insert(data)

        with open(self.imgname[0], "rb") as file:
            MushroomDb.insert_fs(file, self.percentage_score)

    def dropCol(self):
        MushroomDb.drop_coll()
        self.loaddata()

    def open_page1(self):
        self.stackedWidget.setCurrentIndex(0)

    def open_page2(self):
        self.stackedWidget.setCurrentIndex(1)

    def open_page3(self):
        self.loaddata()
        self.stackedWidget.setCurrentIndex(2)

    def loaddata(self):
        self.mushroom_db_data = MushroomDb.find()
        self.mushroom_details = list(MushroomDb.find())
        val_list = list()
        mushroom_list = list()
        for i in range(0, len(self.mushroom_details)):
            val_list = [self.mushroom_details[i]["name"], self.mushroom_details[i]["score"],
                        self.mushroom_details[i]["date"]]
            mushroom_list.append(val_list)

        self.table_widget.setRowCount(len(self.mushroom_details))
        self.table_widget.setColumnCount(4)
        self.table_widget.setColumnWidth(0, 200)
        self.table_widget.setColumnWidth(1, 200)
        self.table_widget.setColumnWidth(2, 100)
        self.table_widget.setColumnWidth(3, 100)
        for row in range(len(self.mushroom_details)):
            self.table_widget.setRowHeight(row, 100)
        self.table_widget.setHorizontalHeaderLabels(("Nazwa", "Prawdopodobieństwo", "Data analizy", "Zdjęcie"))

        row_index = 0
        for i in mushroom_list:
            col_index = 0
            for j in mushroom_list[col_index]:
                self.table_widget.setItem(row_index, 4, QTableWidgetItem())
                self.table_widget.setItem(row_index, col_index,
                                          QTableWidgetItem(str(mushroom_list[row_index][col_index])))

                file_data = MushroomDb.find_one_fs({"filename": self.mushroom_details[row_index]['score']})
                if file_data:
                    base64_data = codecs.encode(file_data.read(), 'base64')
                    byte_array = QByteArray.fromBase64(base64_data)
                    pixmapbyte = QPixmap()
                    pixmapbyte.loadFromData(byte_array)
                    pixmapbyte_scaled = pixmapbyte.scaled(100, 100)
                    item = QTableWidgetItem()
                    item.setData(1, pixmapbyte_scaled)
                    self.table_widget.setItem(row_index, 3, item)
                col_index += 1
            row_index += 1

    def setupUi(self, MainWindow):

        # Main Window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Stacked Widget Page 1
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(40, 30, 701, 501))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")

        # Graphic View, Main Window
        self.graphicsViewGlowna = QtWidgets.QGraphicsView(self.page)
        self.graphicsViewGlowna.setFrameStyle(0)
        self.graphicsViewGlowna.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsViewGlowna.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsViewGlowna.setObjectName("graphicsViewGlowna")
        self.graphicsViewGlowna.setGeometry(QtCore.QRect(210, 20, 244, 244))
        self.scene = QtWidgets.QGraphicsScene()
        self.pixmap = QPixmap('C:\8my semestr\Wieloplatformowe\Projekt\mainimage.png')
        self.pixmap_resized = self.pixmap.scaled(244, 244)
        self.item = QGraphicsPixmapItem(self.pixmap_resized)
        self.scene.addItem(self.item)
        self.graphicsViewGlowna.setScene(self.scene)

        # Analiza, Push Button, Main Window
        self.pushButtonAnaliza = QtWidgets.QPushButton(self.page)
        self.pushButtonAnaliza.setGeometry(QtCore.QRect(290, 280, 93, 28))
        self.pushButtonAnaliza.setObjectName("pushButtonAnaliza")
        self.pushButtonAnaliza.clicked.connect(self.open_page2)

        # Historia, Push Button, Main Window
        self.pushButtonHistoria = QtWidgets.QPushButton(self.page)
        self.pushButtonHistoria.setGeometry(QtCore.QRect(290, 330, 93, 28))
        self.pushButtonHistoria.setObjectName("pushButtonHistoria")
        self.pushButtonHistoria.clicked.connect(self.open_page3)

        # Stacked Widget Page 2
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")

        # Graphic View, Wczytaj
        self.graphicsViewWczytaj = QtWidgets.QGraphicsView(self.page_2)
        self.graphicsViewWczytaj.setFrameStyle(0)
        self.graphicsViewWczytaj.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsViewWczytaj.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsViewWczytaj.setGeometry(QtCore.QRect(210, 20, 244, 244))
        self.graphicsViewWczytaj.setObjectName("graphicsViewWczytaj")

        # Wczytaj, Push Button, Analiza Page
        self.pushButtonWczytaj = QtWidgets.QPushButton(self.page_2)
        self.pushButtonWczytaj.setGeometry(QtCore.QRect(290, 330, 93, 28))
        self.pushButtonWczytaj.setObjectName("pushButtonWczytaj")
        self.pushButtonWczytaj.clicked.connect(self.load)

        # Analizuj, Push Button, Analiza Page
        self.pushButtonAnalizuj = QtWidgets.QPushButton(self.page_2)
        self.pushButtonAnalizuj.setGeometry(QtCore.QRect(290, 370, 93, 28))
        self.pushButtonAnalizuj.setObjectName("pushButtonAnalizuj")
        self.pushButtonAnalizuj.clicked.connect(self.h5_predict)

        # Zapisz, Push Button, Analiza Page
        self.pushButtonZapisz = QtWidgets.QPushButton(self.page_2)
        self.pushButtonZapisz.setGeometry(QtCore.QRect(290, 410, 93, 28))
        self.pushButtonZapisz.setObjectName("pushButtonZapisz")
        self.pushButtonZapisz.clicked.connect(self.db_insert)

        # Text, textAnaliza, Analiza Page
        self.textAnaliza = QtWidgets.QTextBrowser(self.page_2)
        self.textAnaliza.setGeometry(QtCore.QRect(210, 270, 244, 41))
        self.textAnaliza.setObjectName("textAnaliza")

        # Wroc, Push Button, Analiza Page
        self.pushButtonWroc_2 = QtWidgets.QPushButton(self.page_2)
        self.pushButtonWroc_2.setGeometry(QtCore.QRect(20, 20, 93, 28))
        self.pushButtonWroc_2.setObjectName("pushButtonWroc_2")
        self.pushButtonWroc_2.clicked.connect(self.open_page1)

        # Stacked Widget, Page 3
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")

        # Table Widget, Page 3
        layout = QVBoxLayout(self.page_3)
        self.setLayout(layout)
        self.table_widget = QTableWidget(self.page_3)
        layout.addWidget(self.table_widget)

        # Wroc, Push Button, Analiza Page
        self.pushButtonWroc = QtWidgets.QPushButton(self.page_3)
        self.pushButtonWroc.setGeometry(QtCore.QRect(0, 0, 93, 28))
        self.pushButtonWroc.setObjectName("pushButtonWroc")
        self.pushButtonWroc.clicked.connect(self.open_page1)
        layout.addWidget(self.pushButtonWroc)

        # Drop collections
        self.pushButtonDrop = QtWidgets.QPushButton(self.page_3)
        layout.addWidget(self.pushButtonDrop)
        # self.pushButtonDrop.setGeometry(QtCore.QRect(290, 330, 93, 28))
        self.pushButtonDrop.setObjectName("pushButtonDrop")
        self.pushButtonDrop.clicked.connect(self.dropCol)


        # Rest
        self.stackedWidget.addWidget(self.page_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FungEye"))
        self.pushButtonAnaliza.setText(_translate("MainWindow", "Analiza"))
        self.pushButtonHistoria.setText(_translate("MainWindow", "Historia"))
        self.pushButtonWczytaj.setText(_translate("MainWindow", "Wczytaj"))
        self.pushButtonZapisz.setText(_translate("MainWindow", "Zapisz"))
        self.pushButtonWroc_2.setText(_translate("MainWindow", "Wróć"))
        self.pushButtonDrop.setText(_translate("MainWindow", "Wyczyść historię"))
        self.pushButtonWroc.setText(_translate("MainWindow", "Wróć"))
        self.pushButtonAnalizuj.setText(_translate("MainWindow", "Analizuj"))





def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    MushroomDb.initialize()
    ui.setupUi(MainWindow)
    ui.h5_load()
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
