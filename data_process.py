import numpy as np
import glob
import os
import pickle
from scipy import spatial

# Please change this to your location
# data_root = '/data/xincoder/ApolloScape/'
data_root = 'ApolloScape/'

history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 120  # maximum number of observed objects is 70
neighbor_distance = 10  # meter

# Baidu ApolloScape data format:
# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
total_feature_dimension = 10 + 1  # we add mark "1" to the end of each row to indicate that this row exists


# after zero centralize data max(x)=127.1, max(y)=106.1, thus choose 130

def get_frame_instance_dict(pra_file_path):
    '''
    Read raw data from files and return a dictionary:
        {frame_id:
            {object_id:
                # 10 features
                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
            }
        }
    '''
    with open(pra_file_path, 'r') as reader:
        # print(train_file_path)
        # string.strip([obj])在 string 上执行 lstrip()和 rstrip()
        # string.split(str="", num=string.count(str))以 str 为分隔符切片 string，如果 num 有指定值，则仅分隔 num+ 个子字符串
        content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)

        now_dict = {}
        # 对于数组的每一行
        for row in content:
            # instance = {row[1]:row[2:]}
            # 对于新的frame,创建新的字典
            n_dict = now_dict.get(row[0], {})
            # object_id:10 features
            n_dict[row[1]] = row  # [2:]
            # n_dict.append(instance)
            # now_dict[]
            # frame_id:object_id
            now_dict[row[0]] = n_dict
    # print("dir: ",now_dict)
    return now_dict


# process_data(now_dict, start_ind, end_ind, observed_last)
# 输入:按照人工设计的数据字典,起始索引,终止索引,当前观测索引
# 统一维度,特征值中心归零,区分当前可观测障碍物和不可观测障碍物,加入掩码,设置历史帧长度,构建邻接矩阵
def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
    # 当前能够观测到的所有目标的id列表
    visible_object_id_list = list(
        pra_now_dict[pra_observed_last].keys())  # object_id appears at the last observed frame

    num_visible_object = len(visible_object_id_list)  # number of current observed objects

    # compute the mean values of x and y for zero-centralization.
    # 当前观测矩阵n object*11
    visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))
    # print('visible_object_value:\t',visible_object_value.shape)
    # 提取出第4列和第五列元素(x,y)
    xy = visible_object_value[:, 3:5].astype(float)
    # print(xy.shape)
    mean_xy = np.zeros_like(visible_object_value[0], dtype=float)

    m_xy = np.mean(xy, axis=0)
    mean_xy[3:5] = m_xy
    # compute distance between any pair of two objects,返回ndarry
    dist_xy = spatial.distance.cdist(xy, xy)
    # print(dist_xy.shape)
    # if their distance is less than $neighbor_distance, we regard them are neighbors.
    neighbor_matrix = np.zeros((max_num_object, max_num_object))
    # print(neighbor_matrix.shape)
    neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy < neighbor_distance).astype(int)

    now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind) for val in pra_now_dict[x].keys()])

    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
    num_non_visible_object = len(non_visible_object_id_list)
    # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
    object_feature_list = []
    # non_visible_object_feature_list = []
    # 从起始帧到当前帧
    for frame_ind in range(pra_start_ind, pra_end_ind):
        # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1]
        # -mean_xy is used to zero_centralize data
        now_frame_feature_dict = {obj_id
                                  : (list(pra_now_dict[frame_ind][obj_id] - mean_xy) + [1]
                                     if obj_id in visible_object_id_list
                                     else list(pra_now_dict[frame_ind][obj_id] - mean_xy) + [0])
                                  for obj_id in pra_now_dict[frame_ind]}
        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
        # 	dict.get(key, default=None)返回指定键的值，如果值不在字典中返回default值
        # (object,feature)
        now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension))
                                      for vis_id in visible_object_id_list + non_visible_object_id_list])
        # (frame,object,feature)(12,120,11)
        object_feature_list.append(now_frame_feature)

    # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
    object_feature_list = np.array(object_feature_list)

    # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
    object_frame_feature = np.zeros((max_num_object, pra_end_ind - pra_start_ind, total_feature_dimension))
    object_frame_feature[:num_visible_object + num_non_visible_object] = np.transpose(object_feature_list, (1, 0, 2))
    # print(object_frame_feature.shape)
    return object_frame_feature, neighbor_matrix, m_xy


def generate_train_data(pra_file_path):
    '''
    Read data from $pra_file_path, and split data into clips with $total_frames length.
    Return: feature and adjacency_matrix
        feture: (N, C, T, V)
            N is the number of training data
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects.
    '''
    now_dict = get_frame_instance_dict(pra_file_path)
    # sorted() 函数对所有可迭代的对象进行排序操作
    frame_id_set = sorted(set(now_dict.keys()))
    # print(frame_id_set[:-total_frames+1])
    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    # 最后11帧因为没有标签,不进行预测
    # 对于其余每一帧
    for start_ind in frame_id_set[:-total_frames + 1]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames)
        observed_last = start_ind + history_frames - 1
        # ---------------------------core-------------------
        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)
        # --------------------------------------------------
        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list


def generate_test_data(pra_file_path):
    now_dict = get_frame_instance_dict(pra_file_path)
    frame_id_set = sorted(set(now_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []
    # get all start frame id
    start_frame_id_list = frame_id_set[::history_frames]
    # print(start_frame_id_list)
    for start_ind in start_frame_id_list:
        start_ind = int(start_ind)

        end_ind = int(start_ind + history_frames)
        # 当前帧
        observed_last = start_ind + history_frames - 1

        object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

        all_feature_list.append(object_frame_feature)
        all_adjacency_list.append(neighbor_matrix)
        all_mean_list.append(mean_xy)

    # (N, V, T, C) --> (N, C, T, V)
    all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list


def generate_data(pra_file_path_list, pra_is_train=True):
    all_data = []
    all_adjacency = []
    all_mean_xy = []

    for file_path in pra_file_path_list:
        # print(file_path)
        # 如果是训练数据
        if pra_is_train:
            now_data, now_adjacency, now_mean_xy = generate_train_data(file_path)
        else:
            now_data, now_adjacency, now_mean_xy = generate_test_data(file_path)
        # list.extend(seq)在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表
        all_data.extend(now_data)
        all_adjacency.extend(now_adjacency)
        all_mean_xy.extend(now_mean_xy)
    # 将list转化为numpy数组
    all_data = np.array(all_data)  # (N, C, T, V)=(5010, 11, 12, 70) Train
    all_adjacency = np.array(all_adjacency)  # (5010, 70, 70) Train
    all_mean_xy = np.array(all_mean_xy)  # (5010, 2) Train

    # Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
    # Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)
    print('all_data\t{}\nall_adjacency\t{}\nall_mean_xy\t{}\n'.format(np.shape(all_data), np.shape(all_adjacency),
                                                                      np.shape(all_mean_xy)))

    # # save training_data and trainjing_adjacency into a file.
    # if pra_is_train:
    #     save_path = 'train_data.pkl'
    # else:
    #     save_path = 'test_data.pkl'
    # with open(save_path, 'wb') as writer:
    #     # 序列化对象，将对象obj保存到文件file中去
    #     pickle.dump([all_data, all_adjacency, all_mean_xy], writer)


if __name__ == '__main__':
    # glob.glob() 将目录下的文件名全部读取了出来
    # os.path.join()连接两个或更多的路径名组件
    # sorted() 函数对所有可迭代的对象进行排序操作。
    train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    print('Generating Training Data.')
    generate_data(train_file_path_list, pra_is_train=True)

    # print('Generating Testing Data.')
    # generate_data(test_file_path_list, pra_is_train=False)
