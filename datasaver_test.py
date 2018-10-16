from modules.DataSaver import DataSaver

if __name__ == '__main__':
    data_saver = DataSaver()
    data_saver.init_file(path=r'.\test\h5_test.h5', mode='append')
