from modules.DataSaver import DataSaver
from UISettings import settings_init, settings_default

data_saver = DataSaver()

data_saver.load_file(path=r'.\data\test2.h5')

print(data_saver.get_t_s('samp'))


exit(0)
data_saver.reshape_data_df('samp', mark=True, dropnanrow=True)

data_saver.data_exporter(r'.\data\test2.xlsx')
data_saver.data_exporter(r'.\data\test2.csv')
data_saver.data_exporter(r'.\data\test2.json')
