import numpy as np
import os
import glob
import pickle
from scipy import spatial
import sys
import random
import time
from datetime import datetime
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.autograd import Variable

data_root = 'ApolloScape/'

history_frame = 6
future_frame = 6
total_frame = history_frame + future_frame
max_num_object = 120
neighbor_distance = 10
total_feature_dimention = 11
max_x = 1
max_y = 1

batch_size_train = 64
batch_size_val = 32
batch_size_test = 1

total_epoch = 50
dev = 'cuda:0'
work_dir = './trained_models'
log_file = os.path.join(work_dir, 'log_test.txt')
test_predict_result = 'prediction_result.txt'

if not os.path.exists(work_dir):
    os.makedirs(work_dir)


def my_print(pra_content):
    # open file as an append
    with open(log_file, 'a') as writer:
        print(pra_content)
        writer.write(pra_content + '\n')


def display_result(pra_results, pra_pref='Train_epoch'):
    all_overall_sum_list, all_overall_num_list = pra_results
    overall_sum_time = np.sum(all_overall_sum_list ** 0.5, axis=0)
    overall_num_time = np.sum(all_overall_num_list, axis=0)
    overall_loss_time = (overall_sum_time / overall_num_time)
    overall_log = '|{}|[{}] All_All: {}'.format(
        datetime.now(), pra_pref,
        ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
    my_print(overall_log)
    return overall_loss_time


def my_save_model(pra_model, pra_epoch):
    path = '{}/model_epoch_{:04}.pt'.format(work_dir, pra_epoch)
    torch.save(
        {
            'xin_graph_seq2seq_model': pra_model.state_dict(),
        },
        path)
    print('Successfull saved to {}'.format(path))


def my_load_model(pra_model, pra_path):
    checkpoint = torch.load(pra_path)
    pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
    print('Successfull loaded from {}'.format(pra_path))
    return pra_model


class ConvolutionSpatialGraph(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, t_kernel_size=1,
                 t_stride=1, t_padding=0, t_dilation=1, t_bias=True):
        super().__init__()
        self.kernal_size = kernel_size
        self.conv = nn.Conv2d(
            input_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            stride=(t_dilation, 1),
            padding=(t_padding, 0),
            dilation=(t_dilation, 1),
            bias=t_bias
        )

    def forward(self, x, A):
        print('A.shape ', A.shape)
        assert A.size(1) == self.kernal_size
        x = self.conv(x)
        # print('2', x.size())
        N, KC, T, V = x.size()
        x = x.view(N, self.kernal_size, KC // self.kernal_size, T, V)
        # print('3', x.size())
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))
        # print('3', x.size())
        return x.contiguous(), A


class Graph_Convolution_Block(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvolutionSpatialGraph(input_channels, output_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(output_channels, output_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (input_channels == output_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(output_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        # print('111111111111111')
        res = self.residual(x)
        x, A = self.gcn(x, A)
        # print('111111111111111')
        x= self.tcn(x) + res
        # print('111111111111111')
        return self.relu(x), res


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, is_cuda=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.is_cuda = is_cuda
        self.gru = nn.GRU(input_size, hidden_size * 30, num_layers, batch_first=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout, is_cuda=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.is_cuda = is_cuda
        self.gru = nn.GRU(hidden_size, output_size * 30, num_layers, batch_first=True)
        self.liner = nn.Linear(output_size * 30, output_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.liner(output)

        return output, hidden


class Seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, is_cuda=True):
        super().__init__()

        self.is_cuda = is_cuda
        self.encoderRnn = EncoderRNN(input_size, hidden_size, num_layers, is_cuda)
        self.decoderRnn = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, is_cuda)

    def forward(self, input, last_location, prediction_length, teacher_force_ratio=0, teacher_location=None):
        batch_size = input.shape[0]
        out_dim = self.decoderRnn.output_size
        self.predict_length = prediction_length

        outputs = torch.zeros((batch_size, self.predict_length, out_dim))

        if self.is_cuda == True:
            outputs = outputs.cuda()

        encoder_output, encoder_hidden = self.encoderRnn(input)
        decoder_input = last_location
        print('last_location ', last_location.shape)
        for t in range(prediction_length):
            decoder_output, decoder_hidden = self.decoderRnn(decoder_input, encoder_hidden)
            decoder_output += decoder_input
            outputs[:, t:t + 1] = decoder_output
            teacher_force = np.random.random() < teacher_force_ratio
            decoder_input = (teacher_location[:, t:t + 1] if (type(teacher_location) is not type(
                None)) and teacher_force else decoder_output)
        return outputs


class Model(nn.Module):
    def __init__(self, input_channels, graph_args, edge_importance_weight, **kwargs):
        super().__init__()

        self.input_channels = input_channels
        self.graph = Graph(**graph_args)
        A = np.zeros((graph_args['max_hop'] + 1, graph_args['num_node'], graph_args['num_node']))
        spatial_kernel_size = graph_args['max_hop']+1
        temperal_kernel_size = 5
        kernel_size = (temperal_kernel_size, spatial_kernel_size)
        self.hidden_features = 64
        self.st_gcn_networks = nn.ModuleList((
            nn.BatchNorm2d(input_channels),
            # def __init__(self, input_channels, output_channels, kernel_size, stride=1, dropout=0, residual=True):
            Graph_Convolution_Block(input_channels, 64, kernel_size, 1, residual=True, **kwargs),
            Graph_Convolution_Block(64, 64, kernel_size, 1, **kwargs),
            Graph_Convolution_Block(64, 64, kernel_size, 1, **kwargs),
        ))

        if edge_importance_weight:
            self.trainable_graph = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
            )
        else:
            self.trainable_graph = [1] * len(self.st_gcn_networks)

        self.num_node = graph_args['num_node']
        self.out_dim_for_one_node = 2

        self.seq2seq_car = Seq2seq(input_size=64, hidden_size=self.out_dim_for_one_node, num_layers=2, dropout=0.5,
                                   is_cuda=False)
        self.seq2seq_human = Seq2seq(input_size=64, hidden_size=self.out_dim_for_one_node, num_layers=2, dropout=0.5,
                                     is_cuda=False)
        self.seq2seq_bike = Seq2seq(input_size=64, hidden_size=self.out_dim_for_one_node, num_layers=2, dropout=0.5,
                                    is_cuda=False)

    def reshape_for_lstm(self, feature):
        N, C, T, V = feature.size()
        new_feature = feature.permute(0, 3, 2, 1).contiguous()
        new_feature = new_feature.view(N * V, T, C)
        return new_feature

    def reshape_from_lstm(self, predicted):
        NV, T, C = predicted.size()
        new_predicted = predicted.view(-1, self.num_node, T, C)
        new_predicted = new_predicted.permute(0, 3, 2, 1).contiguous()
        return new_predicted

    def forward(self, all_feature, fixed_graph, predicted_length, teacher_force_ratio=0, teacher_location=None):

        x = all_feature

        for gcn, importance in zip(self.st_gcn_networks, self.trainable_graph):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)
            else:

                # print('fixed_graph.shape ',fixed_graph.shape)
                # print('importance.shape ',importance.shape)
                x, _ = gcn(x, fixed_graph + importance)
        # print('111111111111111')
        graph_feature = self.reshape_for_lstm(x)
        last_location = self.reshape_for_lstm(all_feature[:, :2])

        if teacher_force_ratio > 0 and type(teacher_location) is not type(None):
            teacher_location = self.reshape_for_lstm(teacher_location)

        # def forward(self, input, last_location, prediction_length, teacher_force_ratio=0, teacher_location=None):
        now_predicted_car = self.seq2seq_car(
            input=graph_feature,
            last_location=last_location[:, -1:, :],
            prediction_length=predicted_length,
            teacher_force_ratio=teacher_force_ratio,
            teacher_location=teacher_location
        )
        now_predicted_car = self.reshape_from_lstm(now_predicted_car)

        now_predicted_human = self.seq2seq_human(
            input=graph_feature,
            last_location=last_location[:, -1:, :],
            prediction_length=predicted_length,
            teacher_force_ratio=teacher_force_ratio,
            teacher_location=teacher_location
        )
        now_predicted_human = self.reshape_from_lstm(now_predicted_human)

        now_predicted_bike = self.seq2seq_bike(
            input=graph_feature,
            last_location=last_location[:, -1:, :],
            prediction_length=predicted_length,
            teacher_force_ratio=teacher_force_ratio,
            teacher_location=teacher_location
        )
        now_predicted_bike = self.reshape_from_lstm(now_predicted_bike)

        now_predicted = (now_predicted_car + now_predicted_bike + now_predicted_human) / 3.

        return now_predicted


class Graph():
    def __init__(self, max_hop=1, num_node=120):
        self.max_node_num = num_node
        self.max_hop = max_hop

    def get_adjacency(self, input_adjacyncy_maxtix):
        self.hop_dis = np.zeros((self.max_node_num, self.max_node_num)) + np.inf
        transfer_matrix = [np.linalg.matrix_power(input_adjacyncy_maxtix, dim) for dim in range(self.max_hop + 1)]
        arrive_matrix = (np.stack(transfer_matrix) > 0)
        for d in range(2, -1, -1):
            self.hop_dis[arrive_matrix[d]] = d

        adjacency = np.zeros((self.max_node_num, self.max_node_num))
        for hop in range(self.max_hop + 1):
            adjacency[self.hop_dis == hop] = 1

        dl = np.sum(adjacency, axis=0)
        dn = np.zeros((self.max_node_num, self.max_node_num))
        for i in range(self.max_node_num):
            if dl[i] > 0:
                dn[i, i] = dl[i] ** (-1)
        # 将邻接矩阵归一化
        an = np.dot(adjacency, dn)

        # 将邻接矩阵按照max hop的维度分割
        output_matrix = np.zeros((self.max_hop+1, self.max_node_num, self.max_node_num))
        for i, hop in enumerate(range(self.max_hop+1)):
            output_matrix[i][self.hop_dis == hop] = an[self.hop_dis == hop]

        return output_matrix


class PredictionDataset(Dataset):
    def __init__(self, data_path, train_or_test='train', graph_args={}):
        self.data_path = data_path
        self.loader_data()

        total_data_list_num = len(self.all_feature)

        train_data_id_list = list(np.linspace(0, total_data_list_num-1, int(total_data_list_num * 0.8)).astype(int))

        val_data_id_list = list(set(list(range(total_data_list_num))) - set(train_data_id_list))

        self.train_or_test = train_or_test

        if train_or_test.lower() == 'train':
            self.all_feature = self.all_feature[train_data_id_list]
            self.all_adjacency = self.all_adjacency[train_data_id_list]
            self.all_mean_xy = self.all_mean_xy[train_data_id_list]
            # print(self.all_mean_xy.shape)
            # print(self.all_adjacency.shape)
            # print(self.all_feature.shape)

        if train_or_test.lower() == 'test':
            self.all_feature = self.all_feature[train_data_id_list]
            self.all_adjacency = self.all_adjacency[train_data_id_list]
            self.all_mean_xy = self.all_mean_xy[train_data_id_list]

        self.graph = Graph(**graph_args)

    def loader_data(self):
        with open(self.data_path, 'rb') as reader:
            [self.all_feature, self.all_adjacency, self.all_mean_xy] = pickle.load(reader)

    def __len__(self):
        return len(self.all_feature)

    def __getitem__(self, index):
        now_feature = self.all_feature[index]
        # print(now_feature.shape)
        now_mean_xy = self.all_mean_xy[index]
        # print(now_mean_xy.shape)
        if self.train_or_test == 'train' and np.random.random() > 0.5:
            # theta = 2 * np.pi * np.random.random()
            # angle_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)
            angle_mat = np.array(
                [[cos_angle, -sin_angle],
                 [sin_angle, cos_angle]])
            xy = now_feature[3:5, :, :]

            out_xy = np.einsum('ab,btv->atv', angle_mat, xy)

            now_mean_xy = np.matmul(angle_mat, now_mean_xy)

            num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0)  # get the number of valid data
            xy[:, :, :num_xy] = out_xy[:, :, :num_xy]
            now_feature[3:5, :, :] = xy

        now_adjacency_matrix = self.graph.get_adjacency(self.all_adjacency[index])

        # print(now_feature.shape, now_adjacency_matrix.shape, now_mean_xy.shape)
        return now_feature, now_adjacency_matrix, now_mean_xy


def generate_frame_instance_dict(file_path):
    with open(file_path, 'r') as reader:
        line_content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)

    data_dict = {}

    for line in line_content:
        line_frame_dict = data_dict.get(line[0], {})
        line_frame_dict[line[1]] = line
        data_dict[line[0]] = line_frame_dict
    return data_dict


def process_data(train_data_dict, start_frame_index, end_frame_index, current_frame_index):
    current_frame_object_list = list(train_data_dict[current_frame_index].keys())
    num_current_frame_object = len(current_frame_object_list)
    features_current_frame_object = np.array(list(train_data_dict[current_frame_index].values()))

    xy = features_current_frame_object[:, 3:5]
    # print(xy.shape)

    mean_xy = np.zeros_like(features_current_frame_object[0], dtype=float)
    m_xy = np.mean(xy, axis=0)
    mean_xy[3:5] = np.mean(xy, axis=0)
    distance_matrix = spatial.distance.cdist(xy, xy)
    # print(distance_matrix.shape)
    adjacency_matrix = np.zeros((max_num_object, max_num_object))
    # print(adjacency_matrix.shape)
    adjacency_matrix[:num_current_frame_object, :num_current_frame_object] = (
            distance_matrix < neighbor_distance).astype(int)

    all_object_in_frames = set(
        val for x in range(start_frame_index, end_frame_index) for val in train_data_dict[x].keys())
    miss_object_frame_list = list(all_object_in_frames - set(current_frame_object_list))
    num_miss_object_frame = len(miss_object_frame_list)
    object_feature_list = []
    for frame_index in range(start_frame_index, end_frame_index):
        features = {object_id: list(train_data_dict[frame_index][object_id] - mean_xy) + [1]
        if object_id in current_frame_object_list
        else list(train_data_dict[frame_index][object_id] - mean_xy) + [0]
                    for object_id in train_data_dict[frame_index]}

        now_frame_feature = np.array([features.get(vis_id, np.zeros(total_feature_dimention))
                                      for vis_id in current_frame_object_list + miss_object_frame_list])

        object_feature_list.append(now_frame_feature)

    object_feature_list = np.array(object_feature_list)
    object_frame_feature = np.zeros((max_num_object, total_frame, total_feature_dimention))
    object_frame_feature[:num_current_frame_object + num_miss_object_frame] = \
        np.transpose(object_feature_list, (1, 0, 2))

    return object_frame_feature, adjacency_matrix, m_xy


def generate_train_data(file_path):
    train_data_dict = generate_frame_instance_dict(file_path)
    train_data_frame_list = sorted(set(train_data_dict.keys()))

    all_data_feature = []
    all_adjacency_matrix = []
    all_mean_xy = []

    # start_frame_list=train_data_frame_list[::history_frame]

    for start_frame_index in train_data_frame_list[:-total_frame + 1]:
        start_frame_index = int(start_frame_index)
        end_frame_index = start_frame_index + total_frame
        current_frame_index = start_frame_index + history_frame - 1

        object_frame_feature, neighbor_matrix, mean_xy = \
            process_data(train_data_dict, start_frame_index, end_frame_index, current_frame_index)

        all_data_feature.append(object_frame_feature)
        all_adjacency_matrix.append(neighbor_matrix)
        all_mean_xy.append(mean_xy)

    train_data_feature = np.transpose(all_data_feature, (0, 3, 2, 1))
    train_data_adjacency = np.array(all_adjacency_matrix)
    train_data_mean_xy = np.array(all_mean_xy)

    # print(train_data_feature.shape, train_data_adjacency.shape)

    return train_data_feature, train_data_adjacency, train_data_mean_xy


# 产生数据,创建邻接矩阵,xy中心点
def generate_data(get_train_file_list, train_data=True):
    all_data = []
    all_adjacency_matrix = []
    all_mean_xy = []
    for file_path in get_train_file_list:
        if train_data:
            train_data_feature, train_data_adjacency, train_data_mean_xy = generate_train_data(file_path)
        all_data.extend(train_data_feature)
        all_adjacency_matrix.extend(train_data_adjacency)
        all_mean_xy.extend(train_data_mean_xy)
    all_data_array = np.array(all_data)
    all_adjacency_matrix_array = np.array(all_adjacency_matrix)
    all_mean_xy_array = np.array(all_mean_xy)

    # print(all_data_array.shape)
    # print(all_data_array.shape)
    # print(all_data_array.shape)

    print('all_data\t{}\nall_adjacency\t{}\nall_mean_xy\t{}\n'.format(
        np.shape(all_data_array), np.shape(all_adjacency_matrix_array), np.shape(all_mean_xy_array)))

    if train_data:
        save_path = 'train_dataset.pkl'

    with open(save_path, 'wb') as writer:
        pickle.dump([all_data_array, all_adjacency_matrix_array, all_mean_xy_array], writer)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # 每次得到的随机数是固定的。但是如果不加上torch.manual_seed，打印出来的数就不同
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def data_loader(path, t_batch_size=128, t_shuffle=False, train_or_test='train', t_drop_last=False):
    feeder = PredictionDataset(data_path=path, graph_args=graph_args, train_or_test='train')

    loader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=t_batch_size,
        shuffle=t_shuffle,
        num_workers=10,
        drop_last=t_drop_last,
    )
    return loader


def compute_RMSE(t_predict_result,t_ground_truth, mask,  erroe_order=2):
    predict_result = t_predict_result * mask
    ground_truth = t_ground_truth * mask
    xy = torch.sum(torch.abs(predict_result - ground_truth) ** erroe_order, dim=1)
    sum_time = torch.sum(xy, dim=-1)
    sum_num = mask.sum(dim=1).sum(dim=-1)
    return sum_time, sum_num


def preprocess_data(original_data, rascale_xy):
    feature_id = [3, 4, 9, 10]
    ori_data = original_data[:, feature_id].detach()
    data = ori_data.detach().clone()

    mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * mask.float()
    data[:, :2, 0] = 0

    object_type = original_data[:, 2:3]
    # ori_data=ori_data.float()
    data = data.float()
    ori_data = ori_data.float()
    object_type = object_type
    data[:, :2] = data[:, :2] / rascale_xy

    return data, ori_data, object_type


def train_model(model, data_loader, optimizer, epoch_log):
    model.train()
    rescale_xy = torch.ones((1, 2, 1, 1))
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y

    for i, (feature, A, _) in enumerate(data_loader):
        data, oridata, mask = preprocess_data(feature, rescale_xy)

        for frame_index in range(1, data.shape[2]):
            input_data = data[:, :, :frame_index, :]
            ground_truth_data = data[:, :2, frame_index:, :]
            output_mask = data[:, -1:, frame_index:, :]
            predicted_length = output_mask.shape[2]
            A = A.float()
            # def forward(self, all_feature,fixed_graph,predicted_length,teacher_force_ratio=0,teacher_location=None):
            predict_result = model(input_data, A, predicted_length, teacher_force_ratio=0, teacher_location=None)
            # print(predict_result.shape)
            # print(output_mask.shape)
            # print(ground_truth_data.shape)
            sum_time, sum_num = compute_RMSE(predict_result, ground_truth_data, output_mask, erroe_order=2)
            loss = torch.sum(sum_time) / torch.max(torch.sum(sum_num), torch.ones(1,))

            now_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            my_print('|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(
                datetime.now(), epoch_log, i, loss.data.item(), now_lr))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_trainval(t_model, train_path):
    #     def data_loader(path,t_batch_size=128,t_shuffle=False,train_or_test='train',t_drop_last=False):
    # print('11111111111')
    train_loader = data_loader(train_path, t_batch_size=batch_size_train, t_shuffle=False, t_drop_last=True,
                               train_or_test='train')
    # test_loader = data_loader(test_path, t_batch_size=batch_size_test, t_shuffle=False, t_drop_last=True,
    #                           train_or_test='all')
    # print('11111111111')
    # val_loader = data_loader(train_path, t_batch_size=batch_size_val, t_shuffle=False, t_drop_last=True,
    #                          train_or_test='train')

    optimizer = optim.Adam([{'params': model.parameters()}, ], )

    for epoch_index in range(total_epoch):
        # all_train_data = itertools.chain(train_loader, val_loader)

        my_print('#######################################Train')
        train_model(t_model, train_loader, optimizer,
                    epoch_log='Epoch:{:>4}/{:>4}'.format(epoch_index, total_epoch))

        my_save_model(t_model, epoch_index)
        my_print('#######################################Test')
        # display_result(val_model(t_model, val_loader), pra_pref='{}_Epoch{}'.format('Test', epoch_index))


if __name__ == '__main__':

    train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    # test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    # print('Generating Training Data.')
    # generate_data(train_file_path_list, True)

    # print('Generating Testing Data.')
    # generate_data(test_file_path_list, pra_is_train=False)

    graph_args = {'max_hop': 2, 'num_node': 120}
    # def __init__(self,input_channels,graph_args,edge_importance_weight,**kwargs):
    model = Model(input_channels=4, graph_args=graph_args, edge_importance_weight=True)
    # model.to(dev)

    # train and evaluate model
    # run_trainval(model, train_path='./train_data.pkl', test_path='./test_data.pkl')
    run_trainval(model, train_path='./train_dataset.pkl')


def val_model(pra_model, pra_data_loader):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []

    all_car_sum_list = []
    all_car_num_list = []
    all_human_sum_list = []
    all_human_num_list = []
    all_bike_sum_list = []
    all_bike_num_list = []
    # train model using training data
    for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        for now_history_frames in range(6, 7):
            input_data = data[:, :, :now_history_frames, :]  # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:, :2, now_history_frames:, :]  # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:, -1:, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:, :2, now_history_frames:, :]
            ori_output_last_loc = no_norm_loc_data[:, :2, now_history_frames - 1:now_history_frames, :]

            # for category
            cat_mask = ori_data[:, 2:3, now_history_frames:, :]  # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
                                  pra_teacher_forcing_ratio=0,
                                  pra_teacher_location=output_loc_GT)  # (N, C, T, V)=(N, 2, 6, 120)
            ########################################################
            # Compute details for training
            ########################################################
            predicted = predicted * rescale_xy
            # output_loc_GT = output_loc_GT*rescale_xy

            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind - 1:ind + 1], dim=-2)
            predicted += ori_output_last_loc

            ### overall dist
            # overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)
            overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)

            ### car dist
            car_mask = (((cat_mask == 1) + (cat_mask == 2)) > 0).float().to(dev)
            car_mask = output_mask * car_mask
            car_sum_time, car_num, car_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, car_mask)
            all_car_num_list.extend(car_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            car_x2y2 = car_x2y2.detach().cpu().numpy()
            car_x2y2 = car_x2y2.sum(axis=-1)
            all_car_sum_list.extend(car_x2y2)

            ### human dist
            human_mask = (cat_mask == 3).float().to(dev)
            human_mask = output_mask * human_mask
            human_sum_time, human_num, human_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, human_mask)
            all_human_num_list.extend(human_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            human_x2y2 = human_x2y2.detach().cpu().numpy()
            human_x2y2 = human_x2y2.sum(axis=-1)
            all_human_sum_list.extend(human_x2y2)

            ### bike dist
            bike_mask = (cat_mask == 4).float().to(dev)
            bike_mask = output_mask * bike_mask
            bike_sum_time, bike_num, bike_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, bike_mask)
            all_bike_num_list.extend(bike_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            bike_x2y2 = bike_x2y2.detach().cpu().numpy()
            bike_x2y2 = bike_x2y2.sum(axis=-1)
            all_bike_sum_list.extend(bike_x2y2)

    result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list)], pra_pref='car')
    result_human = display_result([np.array(all_human_sum_list), np.array(all_human_num_list)], pra_pref='human')
    result_bike = display_result([np.array(all_bike_sum_list), np.array(all_bike_num_list)], pra_pref='bike')

    result = 0.20 * result_car + 0.58 * result_human + 0.22 * result_bike
    overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), 'WS',
                                                ' '.join(['{:.3f}'.format(x) for x in list(result) + [np.sum(result)]]))
    my_print(overall_log)

    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    return all_overall_sum_list, all_overall_num_list
