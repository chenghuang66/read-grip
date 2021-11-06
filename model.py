import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, EncoderRNN
import numpy as np


# model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True
class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = np.ones((graph_args['max_hop'] + 1, graph_args['num_node'], graph_args['num_node']))
        # print(A.shape)
        # build networks
        spatial_kernel_size = np.shape(A)[0]
        #
        temporal_kernel_size = 5  # 9 #5 # 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # best
        self.st_gcn_networks = nn.ModuleList((
            nn.BatchNorm2d(in_channels),
            Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
        ))

        print("self.backbone: ", self.st_gcn_networks)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # print(self.st_gcn_networks)
        # print(self.edge_importance)
        # 节点数:120
        self.num_node = num_node = self.graph.num_node
        self.out_dim_per_node = out_dim_per_node = 2  # (x, y) coordinate
        # self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
        # self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
        # self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)

        self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                   isCuda=False)
        self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                     isCuda=False)
        self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5,
                                    isCuda=False)

    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous()  # to (N, V, T, C)
        now_feat = now_feat.view(N * V, T, C)
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T,
                                  self.out_dim_per_node)  # (N, T, V, C) ->  [(N, V, T, C)]
        # 重新排列数组维数
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous()  # [(N, V, T, C)]->(N, C, T, V)
        return now_feat

    # predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2],
    # 					  pra_teacher_forcing_ratio=0,
    # 					  pra_teacher_location=output_loc_GT)  # (N, C, T, V)=(N, 2, 6, 120)
    def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
        x = pra_x
        print('input x to bn : ',x.size())
        # forwad
        # _表示并不关心的有效变量
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)
                print('output x to bn : ',x.size())
            else:
                print('pra_A.shape ',pra_A.shape)
                print('importance.shape ',importance.shape)
                print('input x to gcn : ',x.shape)
                x, _ = gcn(x, pra_A + importance)


        # prepare for seq2seq lstm model
        
        graph_conv_feature = self.reshape_for_lstm(x)

        last_position = self.reshape_for_lstm(pra_x[:, :2])  # (N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

        if pra_teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
            pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        # now_predict.shape = (N, T, V*C)
        print('input x to lstm : ',graph_conv_feature.shape)
        now_predict_car = self.seq2seq_car(
            in_data=graph_conv_feature,
            last_location=last_position[:, -1:, :],
            pred_length=pra_pred_length,
            teacher_forcing_ratio=pra_teacher_forcing_ratio,
            teacher_location=pra_teacher_location)
        print('output x to lstm : ',now_predict_car.shape)
        now_predict_car = self.reshape_from_lstm(now_predict_car)  # (N, C, T, V)
        print('now_predict_car : ',now_predict_car.shape)
        now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                               pred_length=pra_pred_length,
                                               teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                               teacher_location=pra_teacher_location)
        print('output x to lstm : ',now_predict_human.shape)
        now_predict_human = self.reshape_from_lstm(now_predict_human)  # (N, C, T, V)
        print('now_predict_human : ',now_predict_human.shape)
        now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:, -1:, :],
                                             pred_length=pra_pred_length,
                                             teacher_forcing_ratio=pra_teacher_forcing_ratio,
                                             teacher_location=pra_teacher_location)
        now_predict_bike = self.reshape_from_lstm(now_predict_bike)  # (N, C, T, V)

        print('now_predict_bike : ',now_predict_bike.shape)
        now_predict = (now_predict_car + now_predict_human + now_predict_bike) / 3.

        print('now_predict : ',now_predict.shape)

        return now_predict


if __name__ == '__main__':
    model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
    print(model)
