import os
import sys
sys.path.append('/userhome/project/Auto_NAS_V2')
from datetime import datetime
import time
import shutil
import glob
import shutil
import pprint
import numpy as np
from network_generator import get_gene_with_skip_connection_constraints, \
    get_gene_by_prob, get_MB_network
import pdb

# walk_dir = sys.argv[1]
#
# print('walk_dir = ' + walk_dir)
#
# print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))


def get_folder_list(walk_dir):
    folder_list = []
    walk_dir = os.path.abspath(walk_dir)
    method_list = os.listdir(walk_dir)
    for method in method_list:
        model_list = os.listdir(os.path.join(walk_dir, method))
        if len(method_list) > 0:
            for model in model_list:
                dataset_list = os.listdir(os.path.join(walk_dir, method, model))
                for dataset in dataset_list:
                    train_name_list = os.listdir(os.path.join(walk_dir, method, model, dataset))
                    for train_name in train_name_list:
                        this_train_folder = os.path.join(walk_dir, method, model, dataset, train_name)
                        folder_list.append(this_train_folder)
    return folder_list


def folder_clean(walk_dir, remove_flag = False):
    folder_list = get_folder_list(walk_dir)
    for this_train_folder in folder_list:
        log_file = os.path.join(this_train_folder, 'logger.log')
        with open(log_file) as f:
            log_lines = f.readlines()
        time_str = log_lines[-1][0:17]
        time_str = time_str.replace(' ', '_')
        time_str = '2020/'+time_str
        FMT = '%Y/%m/%d_%H:%M:%S_%p'
        time_fmt = datetime.strptime(time_str, FMT)
        now_time = datetime.now()
        time_interval = now_time - time_fmt
        if time_interval.seconds > 60 * 60 * 2:
            if not os.path.isfile(os.path.join(this_train_folder, 'best.pth.tar')):
                print(this_train_folder)
                if remove_flag:
                    shutil.rmtree(this_train_folder)


def read_result(walk_dir):
    folder_list = get_folder_list(walk_dir)
    for this_train_folder in folder_list:
        log_file = os.path.join(this_train_folder, 'logger.log')
        with open(log_file) as f:
            log_lines = f.readlines()
        if 'Final' in log_lines[-1]:
            if 'dali' in this_train_folder.split('/')[-1]:
                print(this_train_folder)
                print(log_lines[-1])


def get_result(search_str):
    project_experiment_path = '/userhome/project/Auto_NAS_V2/'
    folder_list = glob.glob(project_experiment_path + 'experiments/dynamic_SNG_V3/ofa/cifar10/' + search_str)
    folder_list.sort()
    for i in folder_list:
        print(i)
        file = open(os.path.join(i, 'logger.log'))
        file_lines = file.readlines()
        for j in file_lines[-7:-1]:
            if '600' in j[25:]:
                print(j[25:])


def get_network():
    project_experiment_path = '/userhome/project/Auto_NAS_V2/'
    folder_list = glob.glob(project_experiment_path + 'experiments/dynamic_SNG_V3/ofa/cifar10/*')
    folder_list.sort()
    dst_folder = '/userhome/project/Auto_NAS_V2/experiments/dynamic_SNG_V3/ofa/cifar10/structure'
    for i in folder_list:
        if 'width_multi' in i:
            _path = os.path.join(i, 'network_info')
            name = i.split('/')[-1][0:-25]
            name = 'ofa_cifar10_' + name
            _dst_folder = os.path.join(dst_folder, name)
            k = 1
            _ = _dst_folder
            while os.path.isdir(_):
                _ = _dst_folder + '_' + str(k)
                k += 1
            _dst_folder = _
            if not os.path.isdir(_dst_folder):
                os.mkdir(_dst_folder)
            for j in range(6):
                this_structure = str((j+1)*100) + '.json'
                _network_path = os.path.join(_path, this_structure)
                dst_network_path = os.path.join(_dst_folder, this_structure)
                shutil.copy(_network_path, dst_network_path)


def get_training_time():
    from datetime import datetime
    save_dir = '/userhome/project/Auto_NAS_V2/experiments/old_0226/DDPNAS_V2/darts/cifar10'
    sub_directorys = glob.glob(save_dir + "/*/")
    sub_directorys.sort()
    print(sub_directorys)
    training_time = []
    for sub_dir in sub_directorys:
        file_name = os.path.join(sub_dir, 'logger.log')
        fileHandle = open(file_name, "r")
        lineList = fileHandle.readlines()
        start_time = lineList[0][:-4]
        end_time = lineList[-1][:-8]
        # pdb.set_trace()
        time_delta = datetime.strptime(end_time, "%m/%d %I:%M:%S %p") - \
                     datetime.strptime(start_time, "%m/%d %I:%M:%S %p")
        training_time.append(time_delta.seconds / 3600. + time_delta.days * 24)
    print(training_time)
    return training_time


def get_training_result():
    save_dir = '/userhome/project/pt.darts/experiment/*/*/*.log'
    sub_directorys = glob.glob(save_dir)
    sub_directorys.sort()
    result = []
    for sub_dir in sub_directorys:
        if 'BN_pruning_step' in sub_dir:
            file_name = os.path.join(sub_dir)
            fileHandle = open(file_name, "r")
            lineList = fileHandle.readlines()
            accuracy = lineList[-2]
            result.append(float(accuracy[-10:-4]))
    print(result)
    print(len(result))
    for i in range(9):
        print(np.mean(np.array(result[i*4:(i+1)*4])))
        print(np.var(np.array(result[i * 4:(i + 1) * 4])))


def get_network_by_constraints():
    save_dir = '/userhome/project/Auto_NAS_V2/experiments/DDPNAS_V3/darts/cifar10/*pruning_step_3_gamma_0.9*/' \
               'network_info/probability.npy'
    sub_directorys = glob.glob(save_dir)
    sub_directorys.sort()
    gen_dict = {}
    flag = 0
    for i in sub_directorys:
        flag += 1
        _this_prob = np.load(i)
        # _this_prob = np.delete(_this_prob, 7, 1)
        # pdb.set_trace()
        for j in range(5):
            constraint = j + 2
            gene = get_gene_with_skip_connection_constraints(_this_prob, skip_constraint=constraint)
            key = str(flag) + '_constraint_' + str(constraint)
            gen_dict[key] = str(gene)
    pprint.pprint(gen_dict)


def get_network_from_different_constraints():
    method_list = ['dynamic_SNG_V3', 'DDPNAS_V3']
    for method in method_list:
        dir_name = '/userhome/project/Auto_NAS_V2/experiments/{}/ofa/imagenet/*/network_info'.format(method)
        dir_list = glob.glob(dir_name)
        dir_list.sort()
        # pprint.pprint(dir_list)
        for i in dir_list:
            i_list = i.split('/')
            for j in [100, 200, 300, 400, 500, 600]:
                save_name = '{0}_{1}_{2}_{3}_{4}'.format(i_list[5], i_list[6], i_list[7], i_list[8][0:15], str(j))
                get_MB_network(i, flops_constraint=j, name=save_name)
                print(save_name)


if __name__ == '__main__':
    get_network_from_different_constraints()
    # get_network_by_constraints()
    # config_path = '/userhome/project/Auto_NAS_V2/experiments/DDPNAS_V3/ofa/imagenet56/width_multi_1.2_epochs' \
    #               '_1000_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3_gamma_0.8_Sat_Feb_29_08:02:38_2020/' \
    #               'network_info'
    # for i in [100, 200, 300, 400, 500, 600]:
    #     get_MB_network(config_path, flops_constraint=i)
    # prob = np.load(os.path.join(config_path, 'probability.npy'))
    # get_gene_by_prob(config_path, prob)
    # get_training_result()
    # get_network_by_constraints()
    # get_network()
    # times = get_training_time()
    # for i in range(int(len(times)/4)):
    #     times_array = np.array(times[0+i*4:(i+1)*4])
    #     print(np.mean(times_array))
    #     print(np.var(times_array))
    # folder_clean(walk_dir, True)
    # read_result(walk_dir)
    # result detect
    # search_space = 'ofa'
    # width_multi = 1.2
    # epoch = 200
    # warm_up_epochs = 0
    # lr = 0.01
    # pruning_step = 3
    # search_str = '{0}__dataset_cifar10_width_multi_{1}_epochs_{2}_data_split_10' \
    #              '_warm_up_epochs_{3}_lr_{4}_pruning_step_{5}*'.format(
    #               str(search_space), str(width_multi), str(epoch), str(warm_up_epochs),
    #               str(lr), str(pruning_step))
    # search_str = '*'
    # get_result(search_str)
    pass

