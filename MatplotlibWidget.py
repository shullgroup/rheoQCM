
'''
matplotlibwidget.py
'''

'''
fig.delaxes(ax)
ax.set_visible(False)
ax.change_geometry(2,2,i+1)
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
    NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import numpy as np
from UISettings import settings_init

# color map for plot
color = ['tab:blue', 'tab:red']

# rcParams['toolbar'] = 'toolmanager'

# class NavigationToolbar(NavigationToolbar2QT):
    # set buttons to show in toolbar
    # toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom')]
    # pass


class MatplotlibWidget(QWidget):
    

    def __init__(self, parent=None, axtype='', title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', showtoolbar=True, dpi=100, *args, **kwargs):
        super(MatplotlibWidget, self).__init__(parent)
        
        self.fig = Figure(tight_layout=True, dpi=dpi, facecolor='none')

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
                class NavigationToolbar(NavigationToolbar2QT):
                    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in showtoolbar]
            else:
                class NavigationToolbar(NavigationToolbar2QT):
                    pass                    

            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(settings_init['max_mpl_toolbar_height'])
            self.toolbar.setStyleSheet("QToolBar { border: 0px;}")
            # if isinstance(showtoolbar, tuple):
            #     print(self.toolbar.toolitems)
            #     NavigationToolbar.toolitems = (t for t in NavigationToolbar2QT.toolitems if t[0] in showtoolbar)
            #     print(self.toolbar.toolitems)

            # self.toolbar.hide() # hide toolbar (or setHidden(bool))
            self.toolbar.isMovable()
            self.vbox.addWidget(self.toolbar)
        
        # initialize axes (ax) and plots (l)
        self.ax = [] # list of axes 
        self.l = {} # all the plot stored in dict
        
        self.initial_axes(axtype=axtype, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale='linear')
        # self.canvas.draw()

    def initax_xy(self, *args, **kwargs):
        # axes
        ax1 = self.fig.add_subplot(111, facecolor='none')
        ax1.autoscale()
        # print(ax.format_coord)
        # print(ax.format_cursor_data)
        # plt.tight_layout()
        # plt.tight_layout(pad=None, w_pad=None, h_pad=None,  rect=None)
        
        # append to list
        self.ax.append(ax1)

    def initax_xyy(self):
        '''
        input: ax1 
        output: [ax1, ax2]
        '''
        self.initax_xy()
        
        ax2 = self.ax[0].twinx()
        ax2.tick_params(axis='y', labelcolor=color[1], color=color[1])
        ax2.yaxis.label.set_color(color[1])
        ax2.spines['right'].set_color(color[1])
        ax2.autoscale()
        ax2.spines['left'].set_visible(False)

        # change axes color
        self.ax[0].tick_params(axis='y', labelcolor=color[0], color=color[0])
        self.ax[0].yaxis.label.set_color(color[0])
        self.ax[0].spines['left'].set_color(color[0])
        self.ax[0].spines['right'].set_visible(False)

        # append ax2 to self.ax
        self.ax.append(ax2)

    def init_sp(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the sp[n]
        initialize ax[0]: .lG, .lGfit, .lp, .lpfit plot
        initialize ax[1]: .lB, .lBfit, plot
        '''
        self.initax_xyy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], title=title, xlabel=r'$f$ (Hz)',ylabel=r'$G_P$ (mS)')
        self.set_ax_items(self.ax[1], xlabel=r'$f$ (Hz)',ylabel=r'$B_P$ (mS)')

        self.ax[0].margins(x=0)
        self.ax[1].margins(x=0)
        self.ax[0].margins(y=.05)
        self.ax[1].margins(y=.05)

        self.ax[0].autoscale()
        self.ax[1].autoscale()

        self.l['lG'] = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[0]
        ) # G
        self.l['lB'] = self.ax[1].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[1]
        ) # B
        self.l['lGpre'] = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color='gray'
        ) # previous G
        self.l['lBpre'] = self.ax[1].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color='gray'
        ) # previous B
        self.l['lGfit'] = self.ax[0].plot(
            [], [], 
            color='k'
        ) # G fit
        self.l['lBfit'] = self.ax[1].plot(
            [], [], 
            color='k'
        ) # B fit
        self.l['lp'] = self.ax[0].scatter(
            [], [],
            marker='x',
            color='k'
        ) # polar plot
        self.l['lpfit'] = self.ax[0].plot(
            [], [],
            color='k'
        ) # polar plot fit

    def init_sp_fit(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the spectra fit
        initialize .lG, .lB, .lGfit, .lBfit .lf, .lg plot
        '''
        self.initax_xyy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel=r'$f$ (Hz)',ylabel=r'$G_P$ (mS)')
        self.set_ax_items(self.ax[1], xlabel=r'$f$ (Hz)',ylabel=r'$B_P$ (mS)')

        self.ax[0].margins(x=0)
        self.ax[1].margins(x=0)
        self.ax[0].margins(y=.05)
        self.ax[1].margins(y=.05)

        self.ax[0].autoscale()
        self.ax[1].autoscale()

        self.l['lG'] = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[0]
        ) # G
        self.l['lB'] = self.ax[1].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[1]
        ) # B
        self.l['lGfit'] = self.ax[0].plot(
            [], [], 
            color='k'
        ) # G fit
        self.l['lBfit'] = self.ax[1].plot(
            [], [], 
            color='k'
        ) # B fit
        self.l['lf'] = self.ax[1].scatter(
            [], [],
            marker='x',
            color='k'
        ) # f: G peak
        self.l['lg'] = self.ax[1].plot(
            [], [],
            color='k'
        ) # g: gamma (fwhm)

    def init_sp_polar(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the spectra polar
        initialize plot: l['l'], l['lfit']
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel=r'$G_P$ (mS)',ylabel=r'$B_P$ (mS)')

        self.ax[0].autoscale()

        self.l['l'] = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[0]
        ) # G vs. B
        self.l['lfit'] = self.ax[0].plot(
            [], [], 
            color='k'
        ) # fit

    def init_data(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the mpl_plt1 & mpl_plt2
        initialize plot: 
            .l<nharm> 
            .lm<nharm>
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel='Time (s)',ylabel=ylabel)

        self.ax[0].autoscale()

        for i in range(1, int(settings_init['max_harmonic']+2), 2):
            self.l['l' + str(i)] = self.ax[0].plot(
                [], [], 
                marker='o', 
                markerfacecolor='none', 
            ) # l
            self.l['lm' + str(i)] = self.ax[0].plot(
                    [], [], 
                    marker='x', 
                    color=self.l['l' + str(i)][0].get_color() # set the same color as .l
                ) # lm

    def init_contour(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the mechanics_contour1 & mechanics_contour2
        initialize plot: 
            l['C'] (contour) 
            .l['cbar'] (colorbar)
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel=r'$d/\lambda$',ylabel=r'$\Phi$ ($\degree$)')

        self.ax[0].autoscale()

        # initiate X, Y, Z data
        levels = settings_init['contour']['levels']
        num = settings_init['contour']['num']
        phi_lim = settings_init['contour']['phi_lim']
        dlam_lim = settings_init['contour']['dlam_lim']
        x = np.linspace(phi_lim[0], phi_lim[1], num=num)
        y = np.linspace(dlam_lim[0], dlam_lim[1], num=num)
        X, Y = np.meshgrid(x, y)
        Z = np.ones(X.shape)
        self.l['C'] = self.ax[0].contourf(
            X, Y, Z, levels, # X, Y, Z, N
        ) # l
        self.l['cbar'] = plt.colorbar(self.l['C'], ax=self.ax[0]) # lm

    # def sizeHint(self):
    #     return QSize(*self.get_width_height())

    # def minimumSizeHint(self):
    #     return QSize(10, 10)

    def set_ax_items(self, ax, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)


    def resize(self, event):
        # on resize reposition the navigation toolbar to (0,0) of the axes.
        # require connect
        # self.canvas.mpl_connect("resize_event", self.resize)
        x,y = self.fig.axes[0].transAxes.transform((0,0))
        figw, figh = self.fig.get_size_inches()
        ynew = figh*self.fig.dpi-y - self.toolbar.frameGeometry().height()
        self.toolbar.move(x,ynew)        

    def initial_axes(self, axtype='', title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        intialize axes by axtype:
        'xy', 'xyy', 'sp', 'sp_fit', 'sp_polar', 'data', 'contour'
        '''
        if axtype == 'xy':
            self.initax_xy()
        elif axtype == 'xyy':
            self.initax_xyy()
        elif axtype == 'sp':
            self.init_sp(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif axtype == 'sp_fit':
            self.init_sp_fit(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif axtype == 'sp_polar':
            self.init_sp_polar(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif axtype == 'data':
            self.init_data(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif axtype == 'contour':
            self.init_contour(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        else:
            pass

    def update_data(self, ls=[], xdata=[], ydata=[]):
        ''' 
        update data of given ld (list of string)
        '''
        for l, x, y in zip(ls, xdata, ydata):
            self.l[l][0].set_xdata(x)
            self.l[l][0].set_ydata(y)
        self.canvas.draw()

    def new_data(self, xdata=[], ydata=[], title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        ''' 
        plot data of in new plots 
        #?? need to define xdata, ydata structure
        [[x1], [x2], ...] ?
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)

        self.ax[0].autoscale()

        self.l = {}
        for i, x, y in enumerate(zip(xdata, ydata)):
            self.l[i] = self.ax[0].plot(
                x, y, 
                marker='o', 
                markerfacecolor='none', 
            ) # l[i]
