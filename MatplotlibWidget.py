
'''
matplotlibwidget.py
'''
import matplotlib
matplotlib.use('QT5Agg')
# matplotlib.rcParams['toolbar'] = 'toolmanager'
matplotlib.rcParams['font.size'] = 9

# import matplotlib.rcParams
# rcParams['font.size'] = 9

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    # NavigationToolbar2QT as NavigationToolbar)
    NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from GUISettings import max_mpl_toolbar_height

# rcParams['toolbar'] = 'toolmanager'

class NavigationToolbar(NavigationToolbar2QT):
    # set buttons to show in toolbar
    # toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom')]
    pass

class MatplotlibWidget(QWidget):
    

    def __init__(self, parent=None, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', showtoolbar=True, dpi=100, *args, **kwargs):
        super(MatplotlibWidget, self).__init__(parent)
        # self.figure = Figure(tight_layout=True, dpi=dpi)
        # self.canvas = FigureCanvas(self.figure)
        # FigureCanvas.__init__(self, self.figure)
        # self.ax = self.figure.add_subplot(111)
        
        self.fig = Figure(tight_layout=True, dpi=dpi, facecolor='none')
        self.ax = self.fig.add_subplot(111, facecolor='none')

        # FigureCanvas.__init__(self, fig)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        # self.canvas.mpl_connect("resize_event", self.resize)

        # layout
        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0) # set layout margins
        self.vbox.addWidget(self.canvas)
        self.setLayout(self.vbox)

        # add toolbar and buttons given by showtoolbar
        if showtoolbar:
            if isinstance(showtoolbar, tuple):
                NavigationToolbar.toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in showtoolbar]
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(max_mpl_toolbar_height)
            self.toolbar.setStyleSheet("QToolBar { border: 0px;}")
            self.toolbar.isMovable()
            self.vbox.addWidget(self.toolbar)

        # axes
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if xscale is not None:
            self.ax.set_xscale(xscale)
        if yscale is not None:
            self.ax.set_yscale(yscale)
        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)

        # print(self.ax.format_coord)
        # print(self.ax.format_cursor_data)
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



    def resize(self, event):
        # on resize reposition the navigation toolbar to (0,0) of the axes.
        # require connect
        # self.canvas.mpl_connect("resize_event", self.resize)
        x,y = self.fig.axes[0].transAxes.transform((0,0))
        figw, figh = self.fig.get_size_inches()
        ynew = figh*self.fig.dpi-y - self.toolbar.frameGeometry().height()
        self.toolbar.move(x,ynew)        

    def initial_figure(self):
        self.ax.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # self.ax.cla()
        self.ax.plot([0, 1, 2, 3], [0, 1, 2, 3], 'r')
        self.canvas.draw()

    def add2ndyaxis(self, ylabel):
        color = ['tab:red', 'tab:blue']
        
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel(ylabel, color=color[1]) # set ylabel of axes2
        self.ax2.tick_params(axis='y', labelcolor=color[1], color=color[1])
        self.ax2.yaxis.label.set_color(color[1])
        self.ax2.spines['right'].set_color(color[1])

        # change axes color
        self.ax.tick_params(axis='y', labelcolor=color[0], color=color[0])
        self.ax.yaxis.label.set_color(color[0])
        self.ax.spines['left'].set_color(color[0])

        self.ax2.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)