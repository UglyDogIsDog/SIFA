import os
import numpy as np


def make_datalist(data_fd, data_list):
    filename_all = os.listdir(data_fd)[:1000]
    filename_all = [data_fd+'/'+filename +
                    '\n' for filename in filename_all if filename.endswith('.tfrecords')]

    np.random.shuffle(filename_all)

    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)


if __name__ == '__main__':
    for run_type in ['training', 'validation']:
        for data_type in ['ct', 'mr']:
            data_fd = './data/' + run_type + '_' + data_type
            data_list = './data/datalist/' + run_type + '_' + data_type + '.txt'
            make_datalist(data_fd, data_list)
