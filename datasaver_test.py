from modules.DataSaver import DataSaver
from UISettings import settings_init, settings_default

data_saver = DataSaver()

data_saver.load_file(path=r'.\data\test2.h5')

# print(data_saver.get_t_s('samp'))
# print(data_saver.get_t_s_marked_rows('samp', dropnanrows=True))
# print(data_saver.get_t_ref())


# TODO set t0 to exp_ref


# data_saver.copy_to_ref('samp')
data_saver.set_ref_set('samp', 'samp', idx_list=[0])
data_saver.set_ref_set('ref', 'ref', idx_list=[0])
print('----')
# print(data_saver.samp_ref)
# print(data_saver.exp_ref)
# exit(0)
print(data_saver.get_list_column_to_columns_marked_rows('samp', 'fs', mark=False, dropnanrow=False, deltaval=True))
# print(data_saver.get_list_column_to_columns('samp', 'fs', mark=False, deltaval=False))
# print(data_saver.get_list_column_to_columns('samp', 'fs', mark=False, deltaval=True))

# exit(0)
# data_saver.reshape_data_df('samp', mark=True, dropnanrow=True)

data_saver.data_exporter(r'.\data\test2.xlsx')
data_saver.data_exporter(r'.\data\test2.csv')
data_saver.data_exporter(r'.\data\test2.json')
