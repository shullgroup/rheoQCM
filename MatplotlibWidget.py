
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
matplotlib.rcParams['toolbar'] = 'toolmanager'
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
from UISettings import settings_init

# color map for plot
color = ['tab:blue', 'tab:red']

# rcParams['toolbar'] = 'toolmanager'

# class NavigationToolbar(NavigationToolbar2QT):
    # set buttons to show in toolbar
    # toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom')]
    # pass
    # # def __init__(self, canvas_, parent_):
    #     self.toolitems = (
    #         ('Home', 'Reset original view', 'home', 'home'),
    #         ('Back', 'Back to      previous view', 'back', 'back'),
    #         ('Forward', 'Forward to next view', 'forward', 'forward'),
    #         (None, None, None, None),
    #         ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
    #         ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    #         (None, None, None, None),
    #         ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
    #         ('Save', 'Save the figure', 'filesave', 'save_figure'),
    #         )   
    #     NavigationToolbar2QT.__init__(self,canvas_,parent_)

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
        
        # initialize axes
        self.initial_axes(title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)


    def initax_xy(self, *args, **kwargs):
        # axes
        ax1 = self.fig.add_subplot(111, facecolor='none')
        ax1.autoscale()
        # print(ax.format_coord)
        # print(ax.format_cursor_data)
        # plt.tight_layout()
        # plt.tight_layout(pad=None, w_pad=None, h_pad=None,  rect=None)
        
        # append to list
        ax = []
        ax.append(ax1)

        self.ax = ax

    def initiax_xyy(self):
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
        self.set_ax_items(self.ax[0], xlabel=r'$f$ (Hz)',ylabel=r'$G_P$ (mS)')
        self.set_ax_items(self.ax[1], xlabel=r'$f$ (Hz)',ylabel=r'$B_P$ (mS)')

        self.ax[0].margins(x=0)
        self.ax[1].margins(x=0)
        self.ax[0].margins(y=.05)
        self.ax[1].margins(y=.05)

        self.ax[0].autoscale()
        self.ax[1].autoscale()

        self.lG = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[0]
        ) # G
        self.lB = self.ax[1].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[1]
        ) # B
        self.lGfit = self.ax[0].plot(
            [], [], 
            color='k'
        ) # G fit
        self.lBfit = self.ax[1].plot(
            [], [], 
            color='k'
        ) # B fit
        self.lp = self.ax[0].scatter(
            [], [],
            marker='x',
            color='k'
        ) # polar plot
        self.lpfit = self.ax[0].plot(
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

        self.lG = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[0]
        ) # G
        self.lB = self.ax[1].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[1]
        ) # B
        self.lGfit = self.ax[0].plot(
            [], [], 
            color='k'
        ) # G fit
        self.lBfit = self.ax[1].plot(
            [], [], 
            color='k'
        ) # B fit
        self.lf = self.ax[1].scatter(
            [], [],
            marker='x',
            color='k'
        ) # f: G peak
        self.lg = self.ax[1].plot(
            [], [],
            color='k'
        ) # g: gamma (fwhm)

    def init_sp_polar(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the spectra polar
        initialize .l .lfit plot
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel=r'$G_P$ (mS)',ylabel=r'$B_P$ (mS)')

        self.ax[0].autoscale()

        self.l = self.ax[0].plot(
            [], [], 
            marker='o', 
            markerfacecolor='none', 
            color=color[0]
        ) # G vs. B
        self.lfit = self.ax[0].plot(
            [], [], 
            color='k'
        ) # fit

    def init_data(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the mpl_plt1 & mpl_plt2
        initialize plot: 
            .l [0: (Settings_init['max_harmonic']-1)/2] 
            .lm[0: (Settings_init['max_harmonic']-1)/2]
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel='Time (s)',ylabel=ylabel)

        self.ax[0].autoscale()

        for i in range(int(settings_init['max_harmonic']+1)/2):
            self.l[i] = self.ax[0].plot(
                [], [], 
                marker='o', 
                markerfacecolor='none', 
            ) # l
            self.lm[i] = self.ax[0].plot(
                [], [], 
                marker='x', 
                color=self.l[i][0].get_color() # set the same color as .l
            ) # lm

    def init_contour(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the mechanics_contour1 & mechanics_contour2
        initialize plot: 
            .C (contour) 
            .cbar (colorbar)
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self.ax[0], xlabel=r'$d/\lambda$',ylabel=r'$\Phi$ ($\degree$)')

        self.ax[0].autoscale()

        self.C = self.ax[0].contourf(
            [], [], [], [], # X, Y, Z, N
        ) # l
        self.cbar = plt.colorbar(self.C) # lm


    # def sizeHint(self):
    #     return QSize(*self.get_width_height())

    # def minimumSizeHint(self):
    #     return QSize(10, 10)

    def set_ax_items(self, ax, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        ax.set_title(title)
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

    def initial_axes(self, axtype='', *args, **kwargs):
        '''
        intialize axes by axtype:
        'xy', 'xyy', 'sp', 'sp_fit', 'sp_polar', 'data', 'contour'
        '''
        if axtype == 'xy':
            self.initax_xy()
        elif axtype == 'xyy':
            self.initax_xyy()
        elif axtype == 'sp':
            self.init_sp(title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)
        elif axtype == 'sp_fit':
            self.init_sp_fit(title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)
        elif axtype == 'sp_polar':
            self.init_sp_polar(title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)
        elif axtype == 'data':
            self.init_data(title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)
        elif axtype == 'contour':
            self.init_contour(title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)
        else:
            pass

    def update_data(self, ls=[], xdata=[], ydata=[]):
        ''' 
        update data of given ls 
        '''
        if not isinstance(ls, list):
            ls.set_xdata(xdata)
            ls.set_ydata(ydata)
        else:
            for l, x, y in zip(ls, xdata, ydata):
                l.set_xdata(x)
                l.set_ydata(y)
        self.canvas.draw()

    def new_data(self, xdata=[], ydata=[], title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        ''' 
        plot data of in new plots 
        #?? need to define xdata, ydata structure
        [[x1], [x2], ...] ?
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax_items(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)

        self.ax[0].autoscale()

        self.l = []
        for x, y in zip(xdata, ydata):
            self.l.append(
                self.ax[0].plot(
                    [], [], 
                    marker='o', 
                    markerfacecolor='none', 
                ) # l
            )
