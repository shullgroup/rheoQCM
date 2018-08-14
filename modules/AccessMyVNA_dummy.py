
import numpy as np
import pandas as pd

class AccessMyVNA():
    '''
    the module used to comunicate with MyVNA
    '''
    def __init__(self):
        super(AccessMyVNA, self).__init__()
        self.f1 = None
        self.f2 = None

    # use __enter__ __exit__ for with or use try finally
    def __enter__(self):
        self.Init()
        print('in')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Close()
        print('out')
        

    def Init(self):
        # close vna window
        return 0

    def Close(self):
        return 0
        
    def ShowWindow(self, nValue=1):
        '''
        nValue 0: show; 1:minimize
        '''
        ret = 0
        return ret

    def GetScanSteps(self, nSteps=200):
        ret = 0
        return ret, nSteps

    def SetScanSteps(self, nSteps=400):
        ret = 0
        return ret, nSteps

    def GetScanAverage(self, nAverage=1):
        ret = 0
        return ret, nAverage

    def SetScanAverage(self, nAverage=1):
        ret = 0
        return ret, nAverage
        

    def GetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9):
        '''
        Get frequency nWhat = GET_SCAN_FREQ_DATA 0
        '''
        if nWhat == 0:
            ndResult = [4.95e+06, 5.05e+06, 4.95e+06, 5.05e+06, 5.00e+06, 1.20e+05, 1.00e+04, 1.00e+06, 1.00e+00]
        elif nWhat == 5:
            ndResult = [1, 2]
        rt = 0
        return rt, ndResult

    def SetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9, nData=[]):
        '''
        Set frequency nWhat = GET_SCAN_FREQ_DATA 0
        '''
        if nWhat == 0:
            nData = [4.95e+06, 5.05e+06, 4.95e+06, 5.05e+06, 5.00e+06, 1.20e+05, 1.00e+04, 1.00e+06, 1.00e+00]
        elif nWhat == 5:
            nData = nData
        rt = 0
        return rt, nData

    def Getinstrmode(self, nMode=0):
        ret = 0
        return ret, nMode

    def Setinstrmode(self, nMode=0):
        '''
        nMode: 0, Reflection
        '''
        ret = 0
        return ret, nMode

    def Getdisplaymode(self, nMode=0):
        ret = 0
        return ret, nMode

    def Setdisplaymode(self, nMode=0):
        ret = 0
        return ret, nMode

    def SingleScan(self):
        '''
        starts a single scan 
        '''
        ret = 0
        return ret

    def EqCctRefine(self):
        ret = 0
        return ret

    def SetFequencies(self, f1=4.95e6, f2=5.05e6, nFlags=1):
        '''
        nFlags: 1 = start / stop
        '''
        self.f1 = f1
        self.f2 = f2
        ret = 0
        return ret, f1, f2
    

    def GetScanData(self, nStart=0, nEnd=299, nWhata=-1, nWhatb=15):
        
        try:
            df = pd.read_csv('data.scv')
        except:
            df = pd.read_csv('./modules/data.csv', header=1)
        df.columns = ['f', 'G', 'B']
        if self.f1 and self.f2:
            df = df.loc[lambda df: (df.f>=self.f1) & (df.f<=self.f2), :]
        # print(df.head(2))
        data = df.values
            
        f = data[:, 0]
        G = data[:, 1]
        B = data[:, 2]


        def assigndata(n):
            if n == -1:
                return f
            elif n == 15:
                return G
            elif n == 16:
                return B
            elif n == -2:
                return np.array([])
        
        data_a = assigndata(nWhata)
        data_b = assigndata(nWhatb)
        ret = 0
        return ret, data_a, data_b

    def Autoscale(self):
        ret = 0
        return ret

    ################ combined functions #################
    def single_scan(self):
        # self.Init()
        self.SingleScan()
        self.Autoscale()
        # wait for some time
        ret, nSteps = self.GetScanSteps()
        ret, f, _ = self.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=-2)
        # time.sleep(1)
        ret, G, B = self.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=15, nWhatb=16)
        # self.Close()
        return ret, f, G, B
    
    def change_settings(self, reflectChn=1, nMode=0, nSteps=400, nAverage=1):
        # ret =           self.Init()
        ret1, nMode =    self.Setinstrmode(nMode)
        ret2, nData =    self.setADCChannel(reflectChn)
        ret3, nSteps =   self.SetScanSteps(nSteps)
        ret4, nAverage = self.SetScanAverage(nAverage)
        # ret =           self.Close()
        return ret1 + ret2 + ret3 + ret4

    def set_steps_freq(self, nSteps=300, f1=4.95e6, f2=5.00e6):
        # set scan parameters
        ret1, nSteps =   self.SetScanSteps(nSteps)
        ret2, f1, f2 = self.SetFequencies(f1, f2, nFlags=1)
        return ret1 + ret2

    def setADCChannel(self, reflectChn):
        # switch ADV channel for test
        # nData = [transChn, reflectChn]
        if reflectChn == 1:
            nData = np.array([2, 1])
        elif reflectChn == 2:
            nData = np.array([1, 2])

        ret, nData = self.SetDoubleArray(nWhat=5, nIndex=0, nArraySize=2, nData=nData)
        return ret, nData


if __name__ == '__main__':
    with AccessMyVNA() as accvna:
        ret, a, b = accvna.GetScanData()
        print(ret)

 