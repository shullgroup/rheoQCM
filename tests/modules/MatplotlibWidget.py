
'''
matplotlibwidget.py
'''

'''
fig.delaxes(ax)
ax.set_visible(False)
ax.change_geometry(2,2,i+1)
'''

# import matplotlib
# matplotlib.use('QT5Agg')
# matplotlib.rcParams['toolbar'] = 'toolmanager'
# matplotlib.rcParams['font.size'] = 10

# import matplotlib.rcParams
# rcParams['font.size'] = 9

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QPixmap

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.projections import register_projection
from matplotlib.widgets import RectangleSelector, SpanSelector
import matplotlib.ticker as ticker
from matplotlib.container import ErrorbarContainer # for checking if l[ln] is ErrorbarContainer

import types

import numpy as np
import pandas as pd

from UISettings import settings_init
from modules import UIModules

# color map for plot
color = ['tab:blue', 'tab:red', 'tab:orange', 'tab:gray']

# class NavigationToolbar(NavigationToolbar2QT):
    # set buttons to show in toolbar
    # toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom')]
    # pass

class AxesLockY(Axes): 
    '''
    cutomized axes with constrained pan to x only
    '''
    def __init__(self, partent=None):
        super(AxesLockY, self).__init__(partent)
    name = 'AxeslockY'
    def drag_pan(self, button, key, x, y):
        Axes.drag_pan(self, button, key, 'x', y) # pretend key=='x

register_projection(AxesLockY)


class SelectorSwitch(ToolToggleBase):
    '''turn on and of '''
    # In case we want to add a toggle button to the toolbar for active/deactive change span
    # default_keymap = 'S'
    description = 'Turn on and off the data selector'

    def __init__(self, *args, **kwargs):
        self.span_selector = kwargs.pop('selector')
        ToolToggleBase.__init__(self, *args, **kwargs)

    def enable(self, *args):
        self.span_selector.set_active(True)

    def disable(self, *args):
        self.span_selector.set_active(False)


class MatplotlibWidget(QWidget):

    def __init__(self, parent=None, axtype='', title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', showtoolbar=True, dpi=100, *args, **kwargs):
        super(MatplotlibWidget, self).__init__(parent)

        # initialize axes (ax) and plots (l)
        self.axtype = axtype
        self.ax = [] # list of axes 
        self.l = {} # all the plot stored in dict
        self.l['temp'] = [] # for temp lines in list
        self.txt = {} # all text in fig
        self.leg = '' # initiate legend 

        # set padding size
        if axtype == 'sp': 
            self.fig = Figure(tight_layout={'pad': 0.05}, dpi=dpi)
            # self.fig = Figure(tight_layout={'pad': 0.05}, dpi=dpi, facecolor='none')
        elif axtype == 'legend':
            self.fig = Figure(dpi=dpi, facecolor='none')

        else:
            self.fig = Figure(tight_layout={'pad': 0.2}, dpi=dpi)
            # self.fig = Figure(tight_layout={'pad': 0.2}, dpi=dpi, facecolor='none')
        ### set figure background transparsent
        # self.setStyleSheet("background: transparent;")

        # FigureCanvas.__init__(self, fig)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        # connect with resize function
        # self.canvas.mpl_connect("resize_event", self.resize)

        # layout
        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0) # set layout margins
        self.vbox.addWidget(self.canvas)
        self.setLayout(self.vbox)

        # add toolbar and buttons given by showtoolbar
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
        #     print(self.toolbar.toolitems) #testprint
        #     NavigationToolbar.toolitems = (t for t in NavigationToolbar2QT.toolitems if t[0] in showtoolbar)
        #     print(self.toolbar.toolitems) #testprint

        # self.toolbar.hide() # hide toolbar (or setHidden(bool))
        self.toolbar.isMovable()
        if showtoolbar:
            self.vbox.addWidget(self.toolbar)
            if self.axtype == 'sp_fit':
                self.toolbar.press_zoom = types.MethodType(press_zoomX, self.toolbar)
        else:
            # pass
            self.toolbar.hide() # hide toolbar. remove this will make every figure with shwotoolbar = False show tiny short toolbar 
        
       
        self.initial_axes(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale='linear')

        # set figure border
        # self.resize('draw_event')
        # if axtype == 'sp':
        #     plt.tight_layout(pad=1.08)
        # else:
        #     self.fig.subplots_adjust(left=0.12, bottom=0.13, right=.97, top=.98, wspace=0, hspace=0)
        #     # self.fig.tight_layout()
        #     # self.fig.tight_layout(pad=0.5, h_pad=0, w_pad=0, rect=(0, 0, 1, 1))

        self.canvas_draw()
    
             
        # self.fig.set_constrained_layout_pads(w_pad=0., h_pad=0., hspace=0., wspace=0.) # for python >= 3.6
        # self.fig.tight_layout()


    def initax_xy(self, *args, **kwargs):
        # axes
        ax1 = self.fig.add_subplot(111, facecolor='none')
        # ax1 = self.fig.add_subplot(111)

        # if self.axtype == 'sp_fit':
        #     # setattr(ax1, 'drag_pan', AxesLockY.drag_pan)
        #     ax1 = self.fig.add_subplot(111, facecolor='none', projection='AxesLockY')
        # else:
        #     ax1 = self.fig.add_subplot(111, facecolor='none')

        # ax1.autoscale()
        # print(ax.format_coord) #testprint
        # print(ax.format_cursor_data) #testprint
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
        self.ax[0].set_zorder(self.ax[0].get_zorder()+1)

        ax2.tick_params(axis='y', labelcolor=color[1], color=color[1])
        ax2.yaxis.label.set_color(color[1])
        ax2.spines['right'].set_color(color[1])
        # ax2.autoscale()
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

        self.ax[0].margins(x=0)
        self.ax[1].margins(x=0)
        self.ax[0].margins(y=.05)
        self.ax[1].margins(y=.05)

        # self.ax[0].autoscale()
        # self.ax[1].autoscale()

        self.l['lG'] = self.ax[0].plot(
            [], [], 
            marker='.', 
            linestyle='none',
            markerfacecolor='none', 
            color=color[0]
        ) # G
        self.l['lB'] = self.ax[1].plot(
            [], [], 
            marker='.', 
            linestyle='none',
            markerfacecolor='none', 
            color=color[1]
        ) # B
        # self.l['lGpre'] = self.ax[0].plot(
        #     [], [], 
        #     marker='.', 
        #     linestyle='none',
        #     markerfacecolor='none', 
        #     color='gray'
        # ) # previous G
        # self.l['lBpre'] = self.ax[1].plot(
        #     [], [], 
        #     marker='.', 
        #     linestyle='none',
        #     markerfacecolor='none', 
        #     color='gray'
        # ) # previous B
        # self.l['lPpre'] = self.ax[1].plot(
        #     [], [], 
        #     marker='.', 
        #     linestyle='none',
        #     markerfacecolor='none', 
        #     color='gray'
        # ) # previous polar
        self.l['lGfit'] = self.ax[0].plot(
            [], [], 
            color='k'
        ) # G fit
        self.l['lBfit'] = self.ax[1].plot(
            [], [], 
            color='k'
        ) # B fit
        self.l['strk'] = self.ax[0].plot(
            [], [],
            marker='+',
            linestyle='none',
            color='r'
        ) # center of tracking peak
        self.l['srec'] = self.ax[0].plot(
            [], [],
            marker='x',
            linestyle='none',
            color='g'
        ) # center of recording peak

        self.l['ltollb'] = self.ax[0].plot(
            [], [],
            linestyle='--',
            color='k'
        ) # tolerance interval lines
        self.l['ltolub'] = self.ax[0].plot(
            [], [],
            linestyle='--',
            color='k'
        ) # tolerance interval lines


        self.l['lP'] = self.ax[0].plot(
            [], [],
            marker='.', 
            linestyle='none',
            markerfacecolor='none', 
            color=color[0]
        ) # polar plot
        self.l['lPfit'] = self.ax[0].plot(
            [], [],
            color='k'
        ) # polar plot fit
        self.l['strk'] = self.ax[0].plot(
            [], [],
            marker='+',
            linestyle='none',
            color='r'
        ) # center of tracking peak
        self.l['srec'] = self.ax[0].plot(
            [], [],
            marker='x',
            linestyle='none',
            color='g'
        ) # center of recording peak

        self.l['lsp'] = self.ax[0].plot(
            [], [],
            color=color[2]
        ) # peak freq span


        # self.ax[0].xaxis.set_major_locator(plt.AutoLocator())
        # self.ax[0].xaxis.set_major_locator(plt.LinearLocator())
        # self.ax[0].xaxis.set_major_locator(plt.MaxNLocator(3))

        # add text
        self.txt['sp_harm'] = self.fig.text(0.01, 0.98, '', va='top',ha='left') # option: weight='bold'
        self.txt['chi'] = self.fig.text(0.01, 0.01, '', va='bottom',ha='left')

        # set label of ax[1]
        self.set_ax(self.ax[0], title=title, xlabel=r'$f$ (Hz)',ylabel=r'$G_P$ (mS)')
        self.set_ax(self.ax[1], xlabel=r'$f$ (Hz)',ylabel=r'$B_P$ (mS)')

        self.ax[0].xaxis.set_major_locator(ticker.LinearLocator(3))



    def update_sp_text_harm(self, harm):
        if isinstance(harm, int):
            harm = str(harm)
        self.txt['sp_harm'].set_text(harm)


    def update_sp_text_chi(self, chi=None):
        '''
        chi: number
        '''
        if not chi:
            self.txt['chi'].set_text('')
        else:
            self.txt['chi'].set_text(r'$\chi^2$ = {:.4f}'.format(chi))


    def init_sp_fit(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the spectra fit
        initialize .lG, .lB, .lGfit, .lBfit .lf, .lg plot
        '''
        self.initax_xyy()

        self.l['lG'] = self.ax[0].plot(
            [], [], 
            marker='.', 
            linestyle='none',
            markerfacecolor='none', 
            color=color[0]
        ) # G
        self.l['lB'] = self.ax[1].plot(
            [], [], 
            marker='.', 
            linestyle='none',
            markerfacecolor='none', 
            color=color[1]
        ) # B
        # self.l['lGpre'] = self.ax[0].plot(
        #     [], [], 
        #     marker='.', 
        #     linestyle='none',
        #     markerfacecolor='none', 
        #     color='gray'
        # ) # previous G
        # self.l['lBpre'] = self.ax[1].plot(
        #     [], [], 
        #     marker='.', 
        #     linestyle='none',
        #     markerfacecolor='none', 
        #     color='gray'
        # ) # previous B
        self.l['lGfit'] = self.ax[0].plot(
            [], [], 
            color='k'
        ) # G fit
        self.l['lBfit'] = self.ax[1].plot(
            [], [], 
            color='k'
        ) # B fit
        self.l['strk'] = self.ax[0].plot(
            [], [],
            marker='+',
            linestyle='none',
            color='r'
        ) # center of tracking peak
        self.l['srec'] = self.ax[0].plot(
            [], [],
            marker='x',
            linestyle='none',
            color='g'
        ) # center of recording peak

        self.l['lsp'] = self.ax[0].plot(
            [], [],
            color=color[2]
        ) # peak freq span

        # set label of ax[1]
        self.set_ax(self.ax[0], xlabel=r'$f$ (Hz)',ylabel=r'$G_P$ (mS)')
        self.set_ax(self.ax[1], xlabel=r'$f$ (Hz)',ylabel=r'$B_P$ (mS)')

        self.ax[0].xaxis.set_major_locator(ticker.LinearLocator(3))

        # self.ax[0].xaxis.set_major_locator(plt.AutoLocator())
        # self.ax[0].xaxis.set_major_locator(plt.LinearLocator())
        # self.ax[0].xaxis.set_major_locator(plt.MaxNLocator(3))

        self.ax[0].margins(x=0)
        self.ax[1].margins(x=0)
        self.ax[0].margins(y=.05)
        self.ax[1].margins(y=.05)
        # self.ax[1].sharex = self.ax[0]

        # self.ax[0].autoscale()
        # self.ax[1].autoscale()

        # add span selector
        self.span_selector_zoomin = SpanSelector(
            self.ax[0], 
            self.sp_spanselect_zoomin_callback,
            direction='horizontal', 
            useblit=True,
            button=[1],  # left click
            minspan=5,
            span_stays=False,
            rectprops=dict(facecolor='red', alpha=0.2)
        )        

        self.span_selector_zoomout = SpanSelector(
            self.ax[0], 
            self.sp_spanselect_zoomout_callback,
            direction='horizontal', 
            useblit=True,
            button=[3],  # right
            minspan=5,
            span_stays=False,
            rectprops=dict(facecolor='blue', alpha=0.2)
        )        


    def sp_spanselect_zoomin_callback(self, xclick, xrelease): 
        '''
        callback of span_selector
        '''

        self.ax[0].set_xlim(xclick, xrelease)


    def sp_spanselect_zoomout_callback(self, xclick, xrelease): 
        '''
        callback of span_selector
        '''
        curr_f1, curr_f2 = self.ax[0].get_xlim()
        curr_fc, curr_fs = UIModules.converter_startstop_to_centerspan(curr_f1, curr_f2)
        # selected range
        sel_fc, sel_fs = UIModules.converter_startstop_to_centerspan(xclick, xrelease)
        # calculate the new span
        ratio = curr_fs / sel_fs
        new_fs = curr_fs * ratio
        new_fc = curr_fc * (1 + ratio) - sel_fc * ratio
        # center/span to f1/f2
        new_f1, new_f2 = UIModules.converter_centerspan_to_startstop(new_fc, new_fs)
        # print('curr_fs', curr_fs) #testprint
        # print('sel_fs', sel_fs) #testprint
        # print('new_fs', new_fs) #testprint
        # print('curr', curr_f1, curr_f2) #testprint
        # print('new', new_f1, new_f2) #testprint
        # set new xlim
        self.ax[0].set_xlim(new_f1, new_f2)
       

    def init_sp_polar(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the spectra polar
        initialize plot: l['l'], l['lfit']
        '''
        self.initax_xy()

        self.l['l'] = self.ax[0].plot(
            [], [], 
            marker='.', 
            linestyle='none',
            markerfacecolor='none', 
            color=color[0]
        ) # G vs. B
        self.l['lfit'] = self.ax[0].plot(
            [], [], 
            color='k'
        ) # fit

        self.l['lsp'] = self.ax[0].plot(
            [], [], 
            color=color[2]
        ) # fit in span range

        # set label of ax[1]
        self.set_ax(self.ax[0], xlabel=r'$G_P$ (mS)',ylabel=r'$B_P$ (mS)')

        # self.ax[0].autoscale()
        self.ax[0].set_aspect('equal')


    def init_data(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the mpl_plt1 & mpl_plt2
        initialize plot: 
            .l<nharm> 
            .lm<nharm>
        '''
        self.sel_mode = 'none' # 'none', 'selector', 'picker'

        self.initax_xy()

        for i in range(1, int(settings_init['max_harmonic']+2), 2):
            self.l['l' + str(i)] = self.ax[0].plot(
                [], [], 
                marker='o', 
                markerfacecolor='none', 
                picker=5, # 5 points tolerance
                label='l'+str(i),
                alpha=0.75, # TODO markerfacecolor becomes dark on Linux when alpha used
            ) # l
        
        for i in range(1, int(settings_init['max_harmonic']+2), 2):
            self.l['lm' + str(i)] = self.ax[0].plot(
                [], [], 
                marker='o', 
                color=self.l['l' + str(i)][0].get_color(), # set the same color as .l
                linestyle='none',
                picker=5, # 5 points tolerance
                label='lm'+str(i),
                alpha=0.75,
            ) # maked points of line
            self.l['ls' + str(i)] = self.ax[0].plot(
                [], [], 
                marker='o', 
                markeredgecolor=color[1], 
                markerfacecolor=color[1],
                alpha= 0.5,
                linestyle='none',
                label='ls'+str(i),
            ) # points in rectangle_selector

        self.l['lp'] = self.ax[0].plot(
            [], [], 
            marker='+', 
            markersize = 12,
            markeredgecolor=color[1],
            markeredgewidth=1, 
            alpha= 1,
            linestyle='none',
            label='',
        ) # points of picker
        
        self.cid = None # for storing the cid of pick_event

        # set label of ax[1]
        self.set_ax(self.ax[0], xlabel='Time (s)',ylabel=ylabel)

        # self.ax[0].autoscale()

        # add rectangle_selector
        self.rect_selector = RectangleSelector(
            self.ax[0], 
            self.data_rectselector_callback,
            drawtype='box',
            button=[1], # left
            useblit=True,
            minspanx=5,
            minspany=5,
            # lineprops=None,
            rectprops=dict(edgecolor = 'black', facecolor='none', alpha=0.2, fill=False),
            spancoords='pixels', # default 'data'
            maxdist=10,
            marker_props=None,
            interactive=False, # change rect after drawn
            state_modifier_keys=None,
        )  
        # set if inavtive
        self.rect_selector.set_active(False)

        # create a toggle button fro selector
        self.pushButton_selectorswitch = QPushButton()
        self.pushButton_selectorswitch.setText('')
        self.pushButton_selectorswitch.setCheckable(True)
        self.pushButton_selectorswitch.setFlat(True)
        # icon
        icon_sel = QIcon()
        icon_sel.addPixmap(QPixmap(':/button/rc/selector.svg'), QIcon.Normal, QIcon.Off)
        self.pushButton_selectorswitch.setIcon(icon_sel)
        
        self.pushButton_selectorswitch.clicked.connect(self.data_rectsleector_picker_switch)

        # add it to toolbar
        self.toolbar.addWidget(self.pushButton_selectorswitch)
        # # add selector switch button to toolbar
        # self.fig.canvas.manager.toolmanager.add_tool('Data Selector', SelectorSwitch, selector=self.rect_selector)

        # # add button to toolbar
        # self.canvas.manager.toolbar.add_tool('DataSelector', 'zoom_pan', 1)


    def data_rectsleector_picker_switch(self, checked):
        if checked:
            print(True) #testprint
            # active rectangle selector
            self.rect_selector.set_active(True)
            # connect pick event
            self.cid = self.canvas.mpl_connect('pick_event', self.data_onpick_callback)
           
            if self.toolbar._active == "PAN":
                self.toolbar.pan()
            elif self.toolbar._active == "ZOOM":
                self.toolbar.zoom()
        else:

            # deactive rectangle selector
            self.rect_selector.set_active(False)
            # reset .l['ls<n>']
            self.clr_lines(l_list=['ls'+ str(i) for i in range(1, int(settings_init['max_harmonic']+2), 2)])
            # deactive pick event            
            self.canvas.mpl_disconnect(self.cid)
            # clear data .l['lp']
            self.clr_lines(l_list=['lp'])

            # reset
            self.sel_mode = 'none'


    def data_rectselector_callback(self, eclick, erelease):
        '''
        when rectangle selector is active
        '''
        # clear pick data .l['lp']
        self.clr_lines(l_list=['lp'])

        # print(dir(eclick)) #testprint
        print(eclick) #testprint
        print(erelease) #testprint
        x1, x2 = sorted([eclick.xdata, erelease.xdata]) # x1 < x2
        y1, y2 = sorted([eclick.ydata, erelease.ydata]) # y1 < y2
        
        # # dict for storing the selected indices
        # sel_idx_dict = {}
        # list for updating selected data
        sel_list = []
        # find the points in rect
        for l_str in ['l', 'lm']: # only one will be not empty
            for harm in range(1, settings_init['max_harmonic']+2, 2):
                harm = str(harm)
                # print(harm) #testprint
                # get data from current plotted lines
                # clear .l['ls<n>']
                self.clr_lines(l_list=['ls'+harm])
                
                # print(l_str) #testprint
                # print(self.l[l_str + harm][0].get_data()) #testprint
                harm_x, harm_y = self.l[l_str + harm][0].get_data()
                
                if isinstance(harm_x, pd.Series): # if data is series (not empty)
                    sel_bool = harm_x.between(x1, x2) & harm_y.between(y1, y2)

                    # save data for plotting selected data
                    sel_list.append({'ln': 'ls'+harm, 'x': harm_x[sel_bool], 'y': harm_y[sel_bool]})

                    # # save indices for later process
                    # sel_idx = harm_x[sel_bool].index
                    # print(sel_idx) #testprint
                    # # update selected indices
                    # sel_idx_dict[harm] = sel_idx
            if (l_str == 'l') and sel_list: # UI mode showall
                break
                #TODO It can also set the display mode from UI (showall/showmarked) and do the loop by the mode

        if sel_list: # there is data selected
            self.sel_mode = 'selector'
        else:
            self.sel_mode = 'none'

        # print(sel_list) #testprint
        # plot the selected data
        self.update_data(*sel_list)


    def data_onpick_callback(self, event):
        '''
        callback function of mpl_data pick_event
        '''
        # clear selector data .l['ls<n>']
        self.clr_lines(l_list=['ls'+ str(i) for i in range(1, int(settings_init['max_harmonic']+2), 2)])

        # print(dir(event)) #testprint
        thisline = event.artist
        x_p = thisline.get_xdata()
        y_p = thisline.get_ydata()
        ind = event.ind[0]
        print(thisline) #testprint
        # print(dir(thisline)) #testprint
        print(thisline.get_label()) #testprint
        print(x_p.name) #testprint
        print(y_p.name)  #testprint
        print(ind)    #testprintk
        # print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind))) #testprint

        # plot
        print('x_p', x_p) #testprint
        print(x_p.iloc[ind], y_p.iloc[ind]) #testprint
        self.l['lp'][0].set_data(x_p.iloc[ind], y_p.iloc[ind])
        self.l['lp'][0].set_label(thisline.get_label() + '_' + str(ind)) # transfer the label of picked line and ind to 'lp'
        self.canvas_draw()

        # set
        self.sel_mode = 'picker'


    def init_contour(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize the mechanics_contour1 & mechanics_contour2
        initialize plot: 
            l['C'] (contour) 
            .l['cbar'] (colorbar)
        '''
        self.initax_xy()

        # initiate X, Y, Z data
        levels = settings_init['contour']['levels']
        num = settings_init['contour']['num']
        phi_lim = settings_init['contour']['phi_lim']
        dlam_lim = settings_init['contour']['dlam_lim']
        x = np.linspace(phi_lim[0], phi_lim[1], num=num)
        y = np.linspace(dlam_lim[0], dlam_lim[1], num=num)
        X, Y = np.meshgrid(y, x)
        Z = np.ones(X.shape)
        self.l['C'] = self.ax[0].contourf(
            X, Y, Z, levels, # X, Y, Z, N
        ) # l
        self.l['colorbar'] = plt.colorbar(self.l['C'], ax=self.ax[0]) # lm
        self.canvas_draw()

        # set label of ax[1]
        self.set_ax(self.ax[0], xlabel=r'$d/\lambda$',ylabel=r'$\Phi$ ($\degree$)')

        # self.ax[0].autoscale()


    def init_legendfig(self, *args, **kwargs):
        ''' 
        plot a figure with only legend
        '''
        self.initax_xy()

        for i in range(1, settings_init['max_harmonic']+2, 2):
            l = self.ax[0].plot([], [], label=i) # l[i]
        self.leg = self.fig.legend(
            # handles=l,
            # labels=range(1, settings_init['max_harmonic']+2, 2),
            loc='upper center', 
            bbox_to_anchor=(0.5, 1),
            borderaxespad=0.,
            borderpad=0.,
            ncol=int((settings_init['max_harmonic']+1)/2), 
            frameon=False, 
            facecolor='none',
            labelspacing=0.0, 
            columnspacing=0.5
        )
        self.canvas_draw()
        
        # set label of ax[1]
        self.set_ax(self.ax[0], title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)

        self.ax[0].set_axis_off() # turn off the axis

        # print(dir(self.leg)) #testprint
        # p = self.leg.get_window_extent() #Bbox of legend
        # # set window height
        # dpi = self.fig.get_dpi()
        # # print(dir(self.fig)) #testprint
        # fsize = self.fig.get_figheight()
 

    def init_prop(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        initialize property plot
        initialize plot: 
            .l<ln>
            ln = l, lm, p, pm 
        '''
        self.initax_xy()

        for i in range(1, int(settings_init['max_harmonic']+2), 2):
            self.l['l' + str(i)] = self.ax[0].plot(
                [], [], 
                # marker='o', 
                markerfacecolor='none', 
                picker=5, # 5 points tolerance
                label='l'+str(i),
                alpha=0.75, # TODO markerfacecolor becomes dark on Linux when alpha used
            ) # l
        
        # for i in range(1, int(settings_init['max_harmonic']+2), 2):
        #     self.l['lm' + str(i)] = self.ax[0].plot(
        #         [], [], 
        #         # marker='o', 
        #         color=self.l['l' + str(i)][0].get_color(), # set the same color as .l
        #         picker=5, # 5 points tolerance
        #         label='lm'+str(i),
        #         alpha=0.75,
        #     ) # maked points of line

        for i in range(1, int(settings_init['max_harmonic']+2), 2):
            self.l['p' + str(i)] = self.ax[0].errorbar(
                [], [], 
                yerr=np.nan,
                xerr=np.nan,
                marker='o', 
                markerfacecolor='none', 
                linestyle='none',
                color=self.l['l' + str(i)][0].get_color(), # set the same color as .l
                # picker=5, # 5 points tolerance
                label=str(i),
                alpha=0.75, # TODO markerfacecolor becomes dark on Linux when alpha used
                capsize=settings_init['mpl_capsize'],
            ) # prop

        for i in range(1, int(settings_init['max_harmonic']+2), 2):
            self.l['pm' + str(i)] = self.ax[0].errorbar(
                [], [], 
                yerr=np.nan,
                xerr=np.nan,
                marker='o', 
                linestyle='none',
                color=self.l['l' + str(i)][0].get_color(), # set the same color as .l
                # picker=5, # 5 points tolerance
                label=str(i),
                alpha=0.75, # TODO markerfacecolor becomes dark on Linux when alpha used
                capsize=settings_init['mpl_capsize'],
            ) # prop marked
        
        # set label of ax[1]
        self.set_ax(self.ax[0], title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)

    # def sizeHint(self):
    #     return QSize(*self.get_width_height())

    # def minimumSizeHint(self):
    #     return QSize(10, 10)

    def set_ax(self, ax, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        self.set_ax_items(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        self.set_ax_font(ax)


    def set_ax_items(self, ax, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)


    def set_ax_font(self, ax, *args, **kwargs):
        if self.axtype == 'sp':
            fontsize = settings_init['mpl_sp_fontsize']
            legfontsize = settings_init['mpl_sp_legfontsize']
            txtfontsize = settings_init['mpl_sp_txtfontsize']
        else:
            fontsize = settings_init['mpl_fontsize']
            legfontsize = settings_init['mpl_legfontsize']
            txtfontsize = settings_init['mpl_txtfontsize']

        if self.axtype == 'contour':
            for ticklabel in self.l['colorbar'].ax.yaxis.get_ticklabels():
                ticklabel.set_size(fontsize)
            self.l['colorbar'].ax.yaxis.offsetText.set_size(fontsize)
            

        if self.axtype == 'legend':
            self.leg
            plt.setp(self.leg.get_texts(), fontsize=legfontsize) 
            
        ax.title.set_fontsize(fontsize+1)
        ax.xaxis.label.set_size(fontsize+1)
        ax.yaxis.label.set_size(fontsize+1)
        ax.tick_params(labelsize=fontsize)
        ax.xaxis.offsetText.set_size(fontsize)
        ax.yaxis.offsetText.set_size(fontsize)
        # ax.xaxis.set_major_locator(ticker.LinearLocator(3))

        # set text fontsize
        for key in self.txt.keys():
            if key == 'sp_harm':
                self.txt[key].set_fontsize(settings_init['mpl_sp_harmfontsize'])
            else: # normal text
                self.txt[key].set_fontsize(txtfontsize)
        


    def resize(self, event):
        # on resize reposition the navigation toolbar to (0,0) of the axes.
        # require connect
        # self.canvas.mpl_connect("resize_event", self.resize)

        # borders = [60, 40, 10, 5] # left, bottom, right, top in px
        # figw, figh = self.fig.get_size_inches()
        # dpi = self.fig.dpi
        # borders = [
        #     borders[0] / int(figw * dpi), # left
        #     borders[1] / int(figh * dpi), # bottom
        #     (int(figw * dpi) - borders[2]) / int(figw * dpi), # right
        #     (int(figh * dpi) - borders[3]) / int(figh * dpi), # top
        # ]
        # print(figw, figh) #testprint
        # print(borders) #testprint
        # self.fig.subplots_adjust(left=borders[0], bottom=borders[1], right=borders[2], top=borders[3], wspace=0, hspace=0)
        
        self.fig.tight_layout(pad=1.08)
        # x,y = self.ax[0].transAxes.transform((0,0))
        # print(x, y) #testprint
        # figw, figh = self.fig.get_size_inches()
        # ynew = figh*self.fig.dpi-y - self.toolbar.frameGeometry().height()
        # self.toolbar.move(x,ynew)        

    def initial_axes(self, title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        '''
        intialize axes by axtype:
        'xy', 'xyy', 'sp', 'sp_fit', 'sp_polar', 'data', 'contour'
        '''
        if self.axtype == 'xy':
            self.initax_xy()
        elif self.axtype == 'xyy':
            self.initax_xyy()
        elif self.axtype == 'sp':
            self.init_sp(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif self.axtype == 'sp_fit':
            self.init_sp_fit(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif self.axtype == 'sp_polar':
            self.init_sp_polar(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif self.axtype == 'data':
            self.init_data(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif self.axtype == 'contour':
            self.init_contour(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        elif self.axtype == 'legend':
            self.init_legendfig()
        elif self.axtype == 'prop':
            self.init_prop(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        else:
            pass

    def update_data(self, *args):
        ''' 
        update data of given in args (dict)
        arg = {'ln':, 'x':, 'y':, 'xerr':, 'yerr':, 'label':,}
            ln: string of line name
            x : x data
            y : y data
            yerr: y error
        '''
        axs = set() # initialize a empty set
        
        for arg in args:
            keys = arg.keys()

            if ('xerr' in keys) or ('yerr' in keys): # errorbar with caps and barlines
                if isinstance(self.l[arg['ln']], ErrorbarContainer): # type match 
                    # print(arg) #testprint
                    # since we initialize the errorbar plots with xerr and yerr, we don't check if they exist here. If you want, use self.l[ln].has_yerr
                    ln = arg['ln'] 
                    x = arg['x'] 
                    y = arg['y'] 
                    xerr = arg['xerr'] 
                    yerr = arg['yerr']

                    line, caplines, barlinecols = self.l[ln]
                    line.set_data(x, y)
                    # Find the ending points of the errorbars 
                    error_positions = (x-xerr,y), (x+xerr,y), (x,y-yerr), (x,y+yerr) 
                    # Update the caplines 
                    for i, pos in enumerate(error_positions): 
                        # print('i', i) #testprint
                        # print(caplines) #testprint
                        # print('caplines_len', len(caplines)) #testprint
                        caplines[i].set_data(pos) 
                    # Update the error bars 
                    barlinecols[0].set_segments(zip(zip(x-xerr,y), zip(x+xerr,y))) 
                    barlinecols[1].set_segments(zip(zip(x,y-yerr), zip(x,y+yerr))) 
                    
                    # barlinecols[0].set_segments(
                    #     np.array([[x - xerr, y], 
                    #     [x + xerr, y]]).transpose((2, 0, 1))
                    # ) 
                    axs.add(line.axes)
            else: # not errorbar
                ln = arg['ln'] 
                x = arg['x'] 
                y = arg['y']
                # print(len(x), len(y)) #testprint
                # self.l[ln][0].set_xdata(x)
                # self.l[ln][0].set_ydata(y)
                self.l[ln][0].set_data(x, y)
                axs.add(self.l[ln][0].axes)
            
            if 'label' in keys: # with label
                self.l[ln][0].set_label(arg['label'])

            # ax = self.l[ln][0].axes
            # axbackground = self.canvas.copy_from_bbox(ax.bbox)
            # print(ax) #testprint
            # self.canvas.restore_region(axbackground)
            # ax.draw_artist(self.l[ln][0])
            # self.canvas.blit(ax.bbox)

        for ax in axs:
            self.reset_ax_lim(ax)

        self.canvas_draw()


    def get_data(self, ls=[]):
        '''
        get data of given ls (lis of string)
        return a list of data with (x, y)
        '''
        data = []
        for l in ls:
            # xdata = self.l[l][0].get_xdata()
            # ydata = self.l[l][0].get_ydata()
            xdata, ydata = self.l[l][0].get_data()
            data.append((xdata, ydata))
        return data


    def del_templines(self, ax=None):
        ''' 
        del all temp lines .l['temp'][:] 
        '''
        if ax is None:
            ax = self.ax[0]

        # print(ax.lines) #testprint
        # print('len temp', len(self.l['temp'])) #testprint
        # print('temp', self.l['temp']) #testprint

        for l_temp in self.l['temp']:
            # print('l_temp', l_temp) #testprint
            ax.lines.remove(l_temp[0]) # remove from ax
        self.l['temp'] = [] # inintiate

        self.reset_ax_lim(ax)
        self.canvas_draw()


    def clr_all_lines(self):
        self.clr_lines()


    def clr_lines(self, l_list=None):
        ''' 
        clear all lines in .l (but not .l['temp'][:]) of key in l_list
        '''
        # print(self.l) #testprint
        for key in self.l:
            # print(key) #testprint
            if key not in ['temp', 'C', 'colorbar']:
                if  l_list is None: # clear all
                    # self.l[key][0].set_xdata([])
                    # self.l[key][0].set_ydata([])
                    self.l[key][0].set_data([], [])
                else:
                    if key in l_list:
                        # self.l[key][0].set_xdata([])
                        # self.l[key][0].set_ydata([])
                        self.l[key][0].set_data([], [])
            elif key == 'temp':
                for ax in self.ax:
                    self.del_templines(ax=ax)

        self.reset_ax_lim(ax)
        self.canvas_draw()


    def change_style(self, line_list, **kwargs):
        '''
        set style of artists in class
        artists: 'linestyle', 'markersize' etc. the same keywords as in matplotlib
        '''
        print(line_list) #testprint
        print(self.l) #testprint
        print(self.l.keys()) #testprint
        for key, val in kwargs.items():
            for l in line_list:
                eval("self.l['{0}'][0].set_{1}('{2}')".format(l, key, val))


    def new_plt(self, xdata=[], ydata=[], title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs):
        ''' 
        plot data of in new plots 
        #TODO need to define xdata, ydata structure
        [[x1], [x2], ...] ?
        '''
        self.initax_xy()
        # set label of ax[1]
        self.set_ax(self.ax[0], title='', xlabel='', ylabel='', xlim=None, ylim=None, xscale='linear', yscale='linear', *args, **kwargs)

        # self.l = {}
        for i, x, y in enumerate(zip(xdata, ydata)):
            self.l[i] = self.ax[0].plot(
                x, y, 
                marker='o', 
                markerfacecolor='none', 
            ) # l[i]

        self.ax[0].autoscale()


    def add_temp_lines(self, ax=None, xlist=[], ylist=[], label_list=[]):
        '''
        add line in self.l['temp'][i]
        all the lines share the same xdata
        '''
        print('add_temp_lines')
        if len(label_list) == 0:
            label_list = [''] * len(xlist) # make up a label_list with all ''
        for (x, y, label) in zip(xlist, ylist, label_list):
            # print('len x: ', len(x)) #testprint
            # print('len y: ', len(y)) #testprint
            # print(x) #testprint
            # print(y) #testprint
            
            if ax is None:
                ax = self.ax[0]

            self.l['temp'].append(ax.plot(
                x, y,
                linestyle='--',
                color=color[-1],
                )
            )

            if label:
                self.l['temp'][-1][0].set_label(label)
        
        self.canvas_draw()


    def canvas_draw(self):
        '''
        redraw canvas after data changed
        '''
        self.canvas.draw()
        # self.canvas.draw_idle()
        # self.canvas.draw_event()
        # self.canvas.draw_cursor()
        self.canvas.flush_events() # flush the GUI events 


    def reset_ax_lim(self, ax):
        '''
        reset the lim of ax
        this change the display and where home button goes back to
        '''
        ax.relim(visible_only=True)
        ax.autoscale_view(True,True,True)
        # ax.autoscale(True, 'both', False) # the same as autoscale_view

def press_zoomX(obj, event):
    event.key = 'x'
    print('event',event) #testprint
    NavigationToolbar2QT.press_zoom(obj, event)
    print('zoomed on x') #testprint