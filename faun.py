#
# @Author: Songyang Zhang 
# @Date: 2019-03-11 18:18:00 
# @Last Modified by:   Songyang Zhang 
# @Last Modified time: 2019-03-11 18:18:00 
#

import torch

from torch import nn
from torch.nn import functional as F
import time
# from maskrcnn_benchmark.layers import FrozenBatchNorm2d
# from maskrcnn_benchmark.layers import Conv2d
import pytorch_analyser

class FAUNet(nn.Module):
    """
    Feature Augmentation with Efficient Graph Neural Network

    Args:
        in_channels:
            (int) channels number of the input feature map
        stride:
            (int) stride number for channel reducion
        num_kernel: 
            (int) number of kernel used for multi kernel approxiamtion
        num_layer: 
            (int) number of graph layer
        method: 
            (str): 'softmax', 'laplacian','gaussian','lowrank'
        spatial_sample:
            (bool): True is to use maxpooling to subsample the feature map for efficiency
                    False is not subsample
        norm_layer:
            (class): which normalization implementation for use
            nn.BatchNorm2d, FrozenBatchNorm2d

        query_normalize:
            (bool), default: True
        key_normalize:
            (bool), default: True
    Return:
    """
    def __init__(self, in_channels, stride=4, num_kernel=1, num_layer=1,latent_stride=2, method='lowrank',spatial_sample=False,norm_layer=nn.BatchNorm2d,query_normalize=True,key_normalize=True):
        super(FAUNet, self).__init__()
        self.num_layer = num_layer
        for i in range(num_layer):
            self.add_module('FAULayer_{}'.format(i), FAULayer(in_channels=in_channels,
                                                            stride=stride,
                                                            num_kernel=num_kernel,
                                                            method=method,
                                                            spatial_sample=spatial_sample,
                                                            norm_layer=norm_layer,
                                                            query_normalize=query_normalize,key_normalize=key_normalize,
                                                            latent_stride=latent_stride))
    def forward(self, conv_feature):
        for i in range(self.num_layer):
            conv_feature = eval('self.FAULayer_{}'.format(i))(conv_feature)
        return conv_feature

class FAULayer(nn.Module):
    """
    """
    def __init__(self, in_channels, stride, num_kernel,latent_stride, method, spatial_sample, norm_layer,query_normalize,key_normalize):
        super(FAULayer, self).__init__()
        self.query_normalize = query_normalize
        self.key_normalize = key_normalize
        self.num_kernel = num_kernel
        # Downchannels
        inter_channels = in_channels // stride
        ########### stride determines inter channels
        self.up_channel_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=inter_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(in_channels))
        self.f_query = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels,out_channels=inter_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(inter_channels))
        self.f_key = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels,out_channels=inter_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(inter_channels))
        #====================================
        if not method == 'lowrank':
            self.f_value = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels,out_channels=inter_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(inter_channels))
            self.graph_conv = nn.Sequential(
                                        nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU())

        if spatial_sample:
            self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2))
            self.f_key = nn.Sequential(self.f_key, self.max_pooling)
            if not method == 'lowrank':
                self.f_value = nn.Sequential(self.f_value, self.max_pooling)

        self.lambda_ = nn.Parameter(torch.zeros(1))

        self.method = method

        if self.method == 'softmax':
            pass
        elif self.method == 'lowrank':
            for i in range(num_kernel):
                self.add_module('FAUKernel_{}'.format(i), FAUKernel(in_channels=inter_channels,num_kernel=num_kernel, latent_stride=latent_stride,norm_layer=norm_layer))

            if num_kernel > 1:
                # self.multi_kernel_weight = nn.Parameter(torch.zeros(num_kernel))#nn.Linear(in_features=num_kernel,out_features=1,bias=False)
                self.multi_kernel_weight = nn.Parameter(torch.ones(num_kernel)/num_kernel)

            # self.lowrank_kernel = FAUKernel(in_channels=in_channels,stride=stride,num_kernel=num_kernel, latent_stride=latent_stride)
        #======================================
    def forward(self, conv_feature):
        """
        Args:
            conv_feature: (tensor) (B, C, H, W)
        Return:
            outpu_feature: (tensor) (B, C, H, W)
        """
        B, C, H, W = conv_feature.shape


        # theta_feature: B, C', H, W
        # phi_feature: B, C', H',W'
        qeury_feature, key_feature = self.f_query(conv_feature), self.f_key(conv_feature)

        # Normalize in channel dimension
        if self.query_normalize:
            qeury_feature = F.normalize(qeury_feature,dim=1)
        if self.key_normalize:
            key_feature = F.normalize(key_feature, dim=1)

        # Normalize the distance
        if self.method == 'lowrank':
            out_feature = []
            for i in range(self.num_kernel):
                out_feature.append(eval('self.FAUKernel_{}'.format(i))(qeury_feature, key_feature))
            if self.num_kernel > 1:
                out_feature = torch.stack(out_feature, dim=-1)
                # import pdb; pdb.set_trace()
                out_feature = torch.matmul(out_feature, self.multi_kernel_weight)
            else:
                out_feature = out_feature[0]

        # #=============================
        # elif self.method in ['softmax', 'laplacian', 'gaussian', 'constant']:
        #     value = self.f_value(conv_feature)
        #     value = value.view(B, -1, value.size(-2)*value.size(-1))# B, C', H'*W'
        #
        #     # Generate affinity matrix
        #     affinity_matrix = torch.matmul(qeury_feature.view(B, -1, qeury_feature.size(-2)*qeury_feature.size(-1)).permute(0,2,1),
        #                                     key_feature.view(B, -1, key_feature.size(-2)*key_feature.size(-1))) # B, HW, HW
        #     # Normalization
        #     # import pdb; pdb.set_trace()
        #
        #     if self.method == 'constant':
        #         affinity_matrix = affinity_matrix / H*W
        #     elif self.method == 'softmax':
        #         affinity_matrix = F.softmax(affinity_matrix, dim=-1)
        #     elif self.method == 'laplacian':
        #         sum_adj = 1 / (affinity_matrix.sum(2) + 1e-9)
        #         repeated_sum_adj = sum_adj.unsqueeze(-1) #.repeat(1, 1, L)
        #         affinity_matrix = repeated_sum_adj * affinity_matrix
        #     elif self.method == 'gaussian':
        #         #TODO
        #         raise NotImplementedError
        #     else:
        #         raise NotImplementedError
        #
        #     #  Feature Update
        #     out_feature = torch.bmm(affinity_matrix,value.permute(0, 2, 1))
        #     out_feature = out_feature.permute(0, 2, 1).view(B, -1, H, W)#B, C',H,W
        #
        #     out_feature = self.graph_conv(out_feature) # B, C', H, W
        # #================================

        out_feature = self.up_channel_conv(out_feature)
        # import pdb; pdb.set_trace()
        return conv_feature + out_feature*self.lambda_

class FAUKernel(nn.Module):
    def __init__(self, in_channels, num_kernel,latent_stride, norm_layer):
        super(FAUKernel, self).__init__()
        self.in_channels = in_channels
        # self.stride = stride
        self.latent_dim = in_channels // latent_stride
        ######## latent_dim is like prev inter channels, controlled by latent stride (set by faunet, 2)
        # Step1: Visible to Latent
        self.phi_func = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=self.latent_dim, kernel_size=1, stride=1,padding=0,bias=False),
                            norm_layer(self.latent_dim),
                            nn.ReLU()
        )
        # self.graph_conv = nn.Sequential(
                            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=1,bias=False),
        # )
        
        # Step2: Latent to Latent
        #TODO: Parametric latent massage passsing.

        # Step3: Latent to Visible
        self.phi_prime_func = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=self.latent_dim, kernel_size=1, # half the in channels
                            stride=1,padding=0,bias=False),
                            norm_layer(self.latent_dim),
                            nn.ReLU()
        )

    def forward(self, query_feature, key_feature):
        """
        Args:
            query_feature: (tensor), B, in_channel, H_query, W_query
            key_feature: (tensor), B, in_channel, H_key, W_key
            
        """
        B,_, H_key, W_key = key_feature.shape
        B,_, H_query, W_query = query_feature.shape

        # Step1: Visible-to-latent message passing
        ###### gav=aff(k): c h w - d h w - d hw
        ###### ln =gav \dot k: d hw . hw c - d c
        # import pdb; pdb.set_trace()
        graph_adj_v2l = self.phi_func(key_feature) # B, latent_dim, H, W
        graph_adj_v2l = F.normalize(graph_adj_v2l.view(B,-1,H_key*W_key), dim=2)# B, latent_dim, H_key*W_key
        # import pdb; pdb.set_trace()
        latent_nodes = torch.bmm(graph_adj_v2l, key_feature.view(B,-1,H_key*W_key).permute(0,2,1)) # B, latent_dim, in_channel
        latent_nodes_normalized = F.normalize(latent_nodes,dim=-1)
        # Step2: latent-to-latent message passing
        ###### afm=ln \dot ln: d c . c d - d d
        ###### ln =afm \dot ln: d d . d c - d c
        affinity_matrix = torch.bmm(latent_nodes_normalized, latent_nodes_normalized.permute(0,2,1)) # B, latent_dim, latent_dim
        # import pdb; pdb.set_trace()
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)
        # latent_nodes = torch.bmm(affinity_matrix/self.latent_dim, latent_nodes)
        latent_nodes = torch.bmm(affinity_matrix, latent_nodes)
        
        # Step3: latent-to-visible message passing
        ###### gal=aff(q): c h w - d h w - d hw
        ###### vn =ln \dot gal: c d . d hw - c hw - c h w
        graph_adj_l2v = self.phi_prime_func(query_feature)# B, latent, H_query*W_query
        graph_adj_l2v = F.normalize(graph_adj_l2v.view(B,-1,H_query*W_query),dim=1) # B, latent, H_query*W_query
        
        visible_nodes = torch.bmm(latent_nodes.permute(0,2,1), graph_adj_l2v).view(B,-1,H_query,W_query) # B, in_channel, H_query, W_query

        return visible_nodes




class FAUCore(nn.Module):
    def v2l(self, key_feature, func):
        """
        visible to latent
        :param key_feature: b,c,h,w
        :param func: transform k to d dim, b,c,hw => b,d,hw
        :return: b,d,c
        """
        B, _, H_key, W_key = key_feature.shape
        # Step1: Visible-to-latent message passing
        ###### gav=aff(k): c h w - d h w - d hw
        ###### ln =gav \dot k: d hw . hw c - d c
        # import pdb; pdb.set_trace()
        graph_adj_v2l = func(key_feature)  # B, latent_dim, H, W
        graph_adj_v2l = F.normalize(graph_adj_v2l.view(B, -1, H_key * W_key), dim=2)  # B, latent_dim, H_key*W_key
        # import pdb; pdb.set_trace()
        latent_nodes = torch.bmm(graph_adj_v2l,
                                 key_feature.reshape((B, -1, H_key * W_key)).permute(0, 2, 1))  # B, latent_dim, in_channel

        return latent_nodes

    def l2l(self, latent_nodes):
        latent_nodes_normalized = F.normalize(latent_nodes, dim=-1)
        # Step2: latent-to-latent message passing
        ###### afm=ln \dot ln: d c . c d - d d
        ###### ln =afm \dot ln: d d . d c - d c => bt,d,c => b, td, c =>
        affinity_matrix = torch.bmm(latent_nodes_normalized,
                                    latent_nodes_normalized.permute(0, 2, 1))  # B, latent_dim, latent_dim
        # import pdb; pdb.set_trace()
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)
        # latent_nodes = torch.bmm(affinity_matrix/self.latent_dim, latent_nodes)
        latent_nodes = torch.bmm(affinity_matrix, latent_nodes)  # d,c

        return latent_nodes

    def l2v(self, query_feature, latent_nodes, func):
        """
        latent to visible
        :param query_feature: b,c,h,w
        :param latent_nodes: b,d,c
        :param func: transform q to d dim, b,c,hw => b,d,hw
        :return: b,c,h,w
        """
        # Step3: latent-to-visible message passing
        ###### gal=aff(q): c h w - d h w - d hw
        ###### vn =ln \dot gal: c d . d hw - c hw - c h w
        graph_adj_l2v = func(query_feature)  # B, latent, H_query*W_query
        B, _, H_query, W_query = query_feature.shape
        graph_adj_l2v = F.normalize(graph_adj_l2v.view(B, -1, H_query * W_query), dim=1)  # B, latent, H_query*W_query

        visible_nodes = torch.bmm(latent_nodes.permute(0, 2, 1), graph_adj_l2v).view(B, -1, H_query,
                                                                                     W_query)  # B, in_channel, H_query, W_query
        return visible_nodes

class FAUKernel_3d(FAUCore):
    def __init__(self, kq_channels, inner_kq_channels, numlayer=1, latent_dim1=64, latent_dim2=64, norm_layer=nn.BatchNorm2d, query_normalize=True, key_normalize=True):
        super(FAUKernel_3d, self).__init__()
        self.num_layer = numlayer
        self.kq_channels = kq_channels
        self.inner_kq_channels = inner_kq_channels
        self.hw_latent_dim = latent_dim1
        self.t_latent_dim = latent_dim2
        self.query_normalize = query_normalize
        self.key_normalize = key_normalize
        # kq - c, latent - d
        self.hw_v2l = nn.Sequential(
            nn.Conv2d(in_channels=kq_channels, out_channels=self.hw_latent_dim, kernel_size=1, stride=1, padding=0,
                      bias=False),
            norm_layer(self.hw_latent_dim),
            nn.ReLU()
        ) # c => d1
        self.hw_l2v = nn.Sequential(
            nn.Conv2d(in_channels=kq_channels, out_channels=self.hw_latent_dim, kernel_size=1, stride=1, padding=0,
                      bias=False),
            norm_layer(self.hw_latent_dim),
            nn.ReLU()
        ) # c => d1
        self.t_v2l = nn.Sequential(
            nn.Conv2d(in_channels=self.inner_kq_channels, out_channels=self.t_latent_dim, kernel_size=1, stride=1, padding=0,
                      bias=False),
            norm_layer(self.t_latent_dim),
            nn.ReLU()
        ) # c => d2
        self.t_l2v = nn.Sequential(
            nn.Conv2d(in_channels=self.inner_kq_channels, out_channels=self.t_latent_dim, kernel_size=1, stride=1, padding=0,
                      bias=False),
            norm_layer(self.t_latent_dim),
            nn.ReLU()
        ) # c => d2
        self.f_k_latent = nn.Sequential(
                                    nn.Conv2d(in_channels=kq_channels, out_channels=self.inner_kq_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                    norm_layer(self.inner_kq_channels)) # c => c
        self.f_q_latent = nn.Sequential(
                                    nn.Conv2d(in_channels=kq_channels, out_channels=self.inner_kq_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                    norm_layer(self.inner_kq_channels))

        self.recover_latent = nn.Sequential(
                                    nn.Conv2d(in_channels=self.inner_kq_channels, out_channels=self.kq_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                    norm_layer(self.kq_channels))

    def forward(self, query_feature, key_feature):
        """
        Args:
            query_feature: (tensor), B, in_channel, t, H_query, W_query
            key_feature: (tensor), B, in_channel, t, H_key, W_key

        Return:
            b,t,c,h,w

        """
        b,c,t,hq,wq = query_feature.shape
        b,c,t,hk,wk = key_feature.shape
        query_feature = query_feature.permute(0,2,1,3,4)
        query_feature = query_feature.reshape((b*t,c,hq,wq))

        key_feature = key_feature.permute(0,2,1,3,4).reshape((b*t,c,hk,wk))
        latent_hw = self.v2l(key_feature, self.hw_v2l)
        for _ in range(self.num_layer):
            latent_hw = self.l2l(latent_hw)

        latent_hw = latent_hw.permute(0,2,1).view(b*t,c,-1,1) # bt,c,d1
        #================
        q, k = self.f_q_latent(latent_hw), self.f_k_latent(latent_hw) # bt,c1,d1,1
        # Normalize in channel dimension
        if self.query_normalize:
            q = F.normalize(q, dim=1)
        if self.key_normalize:
            k = F.normalize(k, dim=1)
        _,c1,d1,_ = q.shape
        q = q.view(b,t,c1,d1).permute(0,2,1,3) # b,c1,t,d1
        k = k.view(b, t, c1, d1).permute(0, 2, 1, 3)  # b,c1,t,d1
        latent_t = self.v2l(k, self.t_v2l) # (b,d2,c1)
        for _ in range(self.num_layer):
            latent_t = self.l2l(latent_t)
        latent_hw = self.l2v(q, latent_t, self.t_l2v)  # b,c1,t,d1
        latent_hw = self.recover_latent(latent_hw) # b,c,t,d1
        #================
        latent_hw = latent_hw.permute(0,2,3,1).reshape((b*t,d1,self.kq_channels)) # bt,d1,c
        visible_nodes = self.l2v(query_feature, latent_hw, self.hw_l2v) # bt,c,hq,wq

        return visible_nodes.view(b,t,c,hq,wq).permute(0,2,1,3,4)


class FAUKernel_thw(nn.Module): # kq_channel, latent channel
    def __init__(self, kq_channels, latent_dim=64, norm_layer=nn.BatchNorm3d):
        super(FAUKernel_thw, self).__init__()
        self.in_channels = kq_channels
        # self.stride = stride
        self.latent_dim = latent_dim
        ######## latent_dim is like prev inter channels, controlled by latent stride (set by faunet, 2)
        # Step1: Visible to Latent
        self.phi_func = nn.Sequential(
            nn.Conv3d(in_channels=kq_channels, out_channels=self.latent_dim, kernel_size=1, stride=1, padding=0,
                      bias=False),
            norm_layer(self.latent_dim),
            nn.ReLU()
        )

        self.phi_prime_func = nn.Sequential(
            nn.Conv3d(in_channels=kq_channels, out_channels=self.latent_dim, kernel_size=1,  # half the in channels
                      stride=1, padding=0, bias=False),
            norm_layer(self.latent_dim),
            nn.ReLU()
        )

    def forward(self, query_feature, key_feature):
        """
        Args:
            query_feature: (tensor), B, in_channel, t, H_query, W_query
            key_feature: (tensor), B, in_channel, t, H_key, W_key

        """
        B, _, t, H_key, W_key = key_feature.shape
        B, _, t, H_query, W_query = query_feature.shape

        # Step1: Visible-to-latent message passing
        ###### gav=aff(k): c h w - d h w - d hw
        ###### ln =gav \dot k: d hw . hw c - d c
        # import pdb; pdb.set_trace()
        graph_adj_v2l = self.phi_func(key_feature)  # B, latent_dim, t,H, W
        graph_adj_v2l = F.normalize(graph_adj_v2l.view(B, -1, t*H_key * W_key), dim=2)  # B, latent_dim, tH_key*W_key
        # import pdb; pdb.set_trace()
        latent_nodes = torch.bmm(graph_adj_v2l,
                                 key_feature.view(B, -1, t*H_key * W_key).permute(0, 2, 1))  # B, latent_dim, in_channel
        latent_nodes_normalized = F.normalize(latent_nodes, dim=-1)
        # Step2: latent-to-latent message passing
        ###### afm=ln \dot ln: d c . c d - d d
        ###### ln =afm \dot ln: d d . d c - d c
        affinity_matrix = torch.bmm(latent_nodes_normalized,
                                    latent_nodes_normalized.permute(0, 2, 1))  # B, latent_dim, latent_dim
        # import pdb; pdb.set_trace()
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)
        # latent_nodes = torch.bmm(affinity_matrix/self.latent_dim, latent_nodes)
        latent_nodes = torch.bmm(affinity_matrix, latent_nodes)

        # Step3: latent-to-visible message passing
        ###### gal=aff(q): c h w - d h w - d hw
        ###### vn =ln \dot gal: c d . d hw - c hw - c h w
        graph_adj_l2v = self.phi_prime_func(query_feature)  # B, latent, H_query*W_query
        graph_adj_l2v = F.normalize(graph_adj_l2v.view(B, -1, t*H_query * W_query), dim=1)  # B, latent, H_query*W_query

        visible_nodes = torch.bmm(latent_nodes.permute(0, 2, 1), graph_adj_l2v).view(B, -1, t, H_query,
                                                                                     W_query)  # B, in_channel, H_query, W_query

        return visible_nodes


class FAUKernel_thw2(FAUCore): # slower
    def __init__(self, kq_channels, numlayer=1, latent_dim=64, norm_layer=nn.BatchNorm2d):
        super(FAUKernel_thw2, self).__init__()
        self.num_layer = numlayer
        self.in_channels = kq_channels
        # self.stride = stride
        self.latent_dim = latent_dim
        ######## latent_dim is like prev inter channels, controlled by latent stride (set by faunet, 2)
        # Step1: Visible to Latent
        self.phi_func = nn.Sequential(
            nn.Conv2d(in_channels=kq_channels, out_channels=self.latent_dim, kernel_size=1, stride=1, padding=0,
                      bias=False),
            norm_layer(self.latent_dim),
            nn.ReLU()
        )

        self.phi_prime_func = nn.Sequential(
            nn.Conv2d(in_channels=kq_channels, out_channels=self.latent_dim, kernel_size=1,  # half the in channels
                      stride=1, padding=0, bias=False),
            norm_layer(self.latent_dim),
            nn.ReLU()
        )

    def forward(self, query_feature, key_feature):
        """
        Args:
            query_feature: (tensor), B, in_channel, t, H_query, W_query
            key_feature: (tensor), B, in_channel, t, H_key, W_key

        """
        B, c, t, H_key, W_key = key_feature.shape
        B, c, t, H_query, W_query = query_feature.shape
        key_feature = key_feature.view((B,c,t*H_key,W_key))
        query_feature = query_feature.reshape((B, c, t * H_query, W_query))
        latent_nodes = self.v2l(key_feature, self.phi_func)
        for _ in range(self.num_layer):
            latent_nodes = self.l2l(latent_nodes)

        visible_nodes = self.l2v(query_feature, latent_nodes, self.phi_prime_func)
        visible_nodes = visible_nodes.view(B, -1, t, H_query, W_query)  # B, in_channel, H_query, W_query

        return visible_nodes

class FAULayer_3d(nn.Module):
    """
    """
    def __init__(self, in_channels, kernel, norm_layer=nn.BatchNorm3d, query_normalize=True, key_normalize=True,
                 kq_channels=64, num_kernel=1):
        super(FAULayer_3d, self).__init__()
        self.query_normalize = query_normalize
        self.key_normalize = key_normalize
        self.num_kernel = num_kernel
        # Downchannels
        # kq_channels = in_channels // kq_stride
        ########### stride determines inter channels
        self.up_channel_conv = nn.Sequential(
                                    nn.Conv3d(in_channels=kq_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(in_channels))
        self.f_query = nn.Sequential(
                                    nn.Conv3d(in_channels=in_channels,out_channels=kq_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(kq_channels))
        self.f_key = nn.Sequential(
                                    nn.Conv3d(in_channels=in_channels,out_channels=kq_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                    norm_layer(kq_channels))
        for i in range(num_kernel):
            self.add_module('FAUKernel_{}'.format(i), kernel)

        if num_kernel > 1:
            self.multi_kernel_weight = nn.Parameter(torch.ones(num_kernel)/num_kernel)

        self.lambda_ = nn.Parameter(torch.zeros(1))

    def forward(self, conv_feature):
        """
        Args:
            conv_feature: (tensor) (B, C, t,H, W)
        Return:
            outpu_feature: (tensor) (B, C, t,H, W)
        """
        qeury_feature, key_feature = self.f_query(conv_feature), self.f_key(conv_feature) # b,t,c0,h,w

        # Normalize in channel dimension
        if self.query_normalize: # for 5d
            qeury_feature = F.normalize(qeury_feature,dim=1)
        if self.key_normalize:
            key_feature = F.normalize(key_feature, dim=1)

        out_feature = []
        for i in range(self.num_kernel):
            out_feature.append(eval('self.FAUKernel_{}'.format(i))(qeury_feature, key_feature)) # 5d
        if self.num_kernel > 1:
            out_feature = torch.stack(out_feature, dim=-1)
            # import pdb; pdb.set_trace()
            out_feature = torch.matmul(out_feature, self.multi_kernel_weight)
        else:
            out_feature = out_feature[0]

        out_feature = self.up_channel_conv(out_feature)
        # import pdb; pdb.set_trace()
        return conv_feature + out_feature*self.lambda_

if __name__ == "__main__":
    # data_input = torch.randn(8,1024,60,60)
    # # Lowrank
    # network = FAUNet(in_channels=1024, stride=4, num_kernel=2,
    #                 num_layer=1,latent_stride=2,
    #                 method='lowrank',spatial_sample=False,
    #                 norm_layer=nn.BatchNorm2d,
    #                 query_normalize=True,
    #                 key_normalize=True)
    # out = network(data_input)

    inp = torch.randn(1,192,8,28,28)

    kq_channels = 64
    c1 = c2 = 32
    d1 = 64
    d2 = 32
    d3 = 256
    kernel1 = FAUKernel_3d(c1, c2, latent_dim1=d1, latent_dim2=d2) # 28x28 - 64, td1=8x64 - 32
    kernel2 = FAUKernel_thw(c1, latent_dim=d3) # 8x28x28=6400
    ks = [kernel1,kernel2]
    for k in ks:
        s = time.time()
        net = FAULayer_3d(in_channels=192, kernel=k, kq_channels=c1)
        # blob_dict, tracked_layers = pytorch_analyser.analyse(net, inp)
        # pytorch_analyser.save_csv(tracked_layers,'analysis.csv')
        out = net(inp)
        dur = time.time() - s
        total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(dur, out.shape, total_params_trainable)
    # import pdb; pdb.set_trace()
    # total_params = sum(p.numel() for p in network.parameters())
    #
    # # for key, values in network.named_parameters():
    #     # print(key, values.numel())
    # # Total Trainable Parameters
    # total_params_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)

    # print("Total Parameters: {}, Total Trainable Parameters: {}".format(total_params, total_params_trainable))
    # import pdb; pdb.set_trace()


    # data_input = torch.randn(8,1024,60,60)
    # # Softmax
    # network = FAUNet(in_channels=1024, stride=4, num_kernel=1, 
    #                 num_layer=1,method='softmax',spatial_sample=False,
    #                 norm_layer=nn.BatchNorm2d,
    #                 query_normalize=True,
    #                 key_normalize=True)
    # out = network(data_input)
    # # import pdb; pdb.set_trace()
    # total_params = sum(p.numel() for p in network.parameters())

    # # for key, values in network.named_parameters():
    #     # print(key, values.numel())
    # # Total Trainable Parameters
    # total_params_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)

    # print("Total Parameters: {}, Total Trainable Parameters: {}".format(total_params, total_params_trainable))
    # import pdb; pdb.set_trace()


    # data_input = torch.randn(8,1024,60,60)
    # # Laplacian
    # network = FAUNet(in_channels=1024, stride=4, num_kernel=1, 
    #                 num_layer=1,method='softmax',spatial_sample=False,
    #                 norm_layer=nn.BatchNorm2d,
    #                 query_normalize=True,
    #                 key_normalize=True)
    # out = network(data_input)
    # # import pdb; pdb.set_trace()
    # total_params = sum(p.numel() for p in network.parameters())

    # # for key, values in network.named_parameters():
    #     # print(key, values.numel())
    # # Total Trainable Parameters
    # total_params_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)

    # print("Total Parameters: {}, Total Trainable Parameters: {}".format(total_params, total_params_trainable))
    # import pdb; pdb.set_trace()