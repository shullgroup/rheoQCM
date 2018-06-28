
'''
matplotlibwidget.py
'''
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    # NavigationToolbar2QT as NavigationToolbar)
    NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib import pyplot as plt

rcParams['font.size'] = 9

class NavigationToolbar(NavigationToolbar2QT):
    # print(dir(NavigationToolbar2QT))
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

    contentsMargins = 0
class MatplotlibWidget(QWidget):
    

    def __init__(self, parent=None, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', width=4, height=3, dpi=100, *args, **kwargs):
        super(MatplotlibWidget, self).__init__(parent)
        # self.figure = Figure(tight_layout=True, dpi=dpi)
        # self.canvas = FigureCanvas(self.figure)
        # FigureCanvas.__init__(self, self.figure)
        # self.axes = self.figure.add_subplot(111)

        fig = Figure(tight_layout=True, dpi=dpi)
        self.axes = fig.add_subplot(111)

        # FigureCanvas.__init__(self, fig)
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        self.toolbar = NavigationToolbar(self.canvas, self)
        # layout
        # self.layoutmlp = QVBoxLayout()
        # self.layoutmlp.addWidget(self.toolbar)
        # self.layoutmlp.addWidget(self.canvas)
        # self.setLayout(self.layoutmlp)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

        # axes
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if xscale is not None:
            self.axes.set_xscale(xscale)
        if yscale is not None:
            self.axes.set_yscale(yscale)
        if xlim is not None:
            self.axes.set_xlim(*xlim)
        if ylim is not None:
            self.axes.set_ylim(*ylim)

        plt.tight_layout()

        # super(MatplotlibWidget, self).__init__(self.figure)
        # self.setParent(parent)
        # super(MatplotlibWidget, self).setSizePolicy(
        #     QSizePolicy.Expanding, QSizePolicy.Expanding
        #     # QSizePolicy.Preferred, QSizePolicy.Preferred
        #     )
        # super(MatplotlibWidget, self).updateGeometry()

    # def sizeHint(self):
    #     return QSize(*self.get_width_height())

    # def minimumSizeHint(self):
    #     return QSize(10, 10)

    def initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # self.axes.cla()
        self.axes.plot([0, 1, 2, 3], [0, 1, 2, 3], 'r')
        self.canvas.draw()
