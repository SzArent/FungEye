import unittest
from PyQt5 import QtCore


class TestMenuBar:
    def test_menubar(self):
        object_name = self.widget.menubar.objectName()
        self.assertEqual(object_name, "menubar")
        self.assertEqual(self.widget.menuBar(), self.widget.menubar)
        self.assertEqual(self.widget.menubar.geometry(), QtCore.QRect(0, 0, 800, 26))


if __name__ == '__main__':
    unittest.main()
