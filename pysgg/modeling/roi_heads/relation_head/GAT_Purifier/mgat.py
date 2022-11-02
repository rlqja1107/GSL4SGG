import sys
sys.path.append('')
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.utils.sparse import dense_to_sparse
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
from pysgg.modeling.roi_heads.relation_head.hgrcnn.Feature_Extracter import PairwiseFeatureExtractor



class MaskGAT(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskGAT, self).__init__()
        self.cfg = cfg

        self.dim = cfg.MODEL.ROI_RELATION_HEAD.RGCN.H_DIM

        self.topk = cfg.MGAT.TOPK

        self.feat_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.FEATURE_UPDATE_STEP
        self.score_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.SCORES_UPDATE_STEP
        num_classes_obj = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes_pred = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)

        self.rel_embedding = nn.Sequential(
            nn.Linear(self.pairwise_feature_extractor.pooling_dim, self.dim*2),
            nn.ReLU(True),
            nn.Linear(self.dim*2, self.dim),
            nn.ReLU(True) 
        )
        self.obj_embedding = nn.Sequential(
            nn.Linear(self.pairwise_feature_extractor.pooling_dim, self.dim*2),
            nn.ReLU(True),
            nn.Linear(self.dim*2, self.dim),
            nn.ReLU(True)
             
        )
    
        if self.feat_update_step > 0:
            self.mgat = MaskGATLayer(self.dim, self.dim, self.dim, self.dim, cfg)


        self.rel_classifier = build_classifier(self.dim, num_classes_pred) # Linear Layer
        self.obj_classifier = build_classifier(self.dim, num_classes_obj)  # 양방향이면 pred 2배     
        
        if self.score_update_step > 0:
            self.mgat_score = MaskGATLayer(num_classes_obj, num_classes_pred, num_classes_obj, num_classes_pred, cfg)

        self.obj_logit_layer = nn.Linear(num_classes_obj, num_classes_obj) 
        self.pred_logit_layer = nn.Linear(num_classes_pred, num_classes_pred)
        

        self.init_classifier_weight()


    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()


    def forward(self, inst_features, rel_union_features, proposals, rel_pair_inds, rel_gt_binarys=None, logger=None):

        # Define
        nf, ef = self.pairwise_feature_extractor( 
            inst_features,
            rel_union_features,
            proposals,
            rel_pair_inds,
        )

        g_rel, g_skip = self._get_map_idxs(proposals, rel_pair_inds)

        nf = self.obj_embedding(nf) # Node Feature
        ef = self.rel_embedding(ef) # Edge Feature


        for _ in range(self.feat_update_step):
            nf, ef = self.mgat(g_rel, g_skip,nf, ef)
            nf = F.elu(nf)
            ef = F.elu(ef)
        
        obj_class_logits = self.obj_classifier(nf)
        pred_class_logits = self.rel_classifier(ef)

        # Logit Layer
        for _ in range(self.score_update_step):
            obj_class_logits, pred_class_logits = self.mgat_score(g_rel, g_skip,obj_class_logits, pred_class_logits)
            obj_class_logits = F.elu(obj_class_logits)
            pred_class_logits = F.elu(pred_class_logits)

        return obj_class_logits, pred_class_logits

    

    def _get_map_idxs(self, proposals, proposal_pairs):
        
        offset = 0

        obj_num = sum([len(proposal) for proposal in proposals])
        rel_inds = []    
        obj_obj_map = torch.FloatTensor(obj_num, obj_num).fill_(0)
    
        
        for proposal, rel_ind_i in zip(proposals, proposal_pairs):
            rel_ind_i = rel_ind_i.detach().clone()
            rel_ind_i += offset
            rel_inds.append(rel_ind_i)
            
            obj_obj_map_i = (1 - torch.eye(len(proposal))).float()
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(proposal)] = obj_obj_map_i

            offset += len(proposal)


        obj_obj_edge_index = dense_to_sparse(obj_obj_map)[0].cuda()
        rel_inds = torch.cat(rel_inds, 0).T
        
        return rel_inds, obj_obj_edge_index


class MaskGATLayer(MessagePassing):

    def __init__(self, node_in_channels, edge_in_channels, node_out_channels, edge_out_channels, cfg):
        super(MaskGATLayer, self).__init__()


        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels

        self.node_out_channels = node_out_channels
        self.edge_out_channels = edge_out_channels

        self.is_topk = cfg.MGAT.IS_TOPK
        if self.is_topk:
            self.topk = cfg.MGAT.TOPK
        else:
            self.relative_rank = cfg.MGAT.RELATIVE_RANK / 100

        self.negative_slope = 0.2
        
        # Relation Embedding: attn sub2rel + attn ob2rel    ->  u_new_ij =  u_ij + attn sub2rel + attn ob2rel 로 변경
        self.sub2rel_msg = nn.Linear(2*node_in_channels, edge_out_channels) # (x_i, x_j  -> m_ij #sub)
        self.sub2rel_attn = nn.Linear(edge_out_channels, 1)

        self.obj2rel_msg = nn.Linear(2*node_in_channels, edge_out_channels) # (x_i, x_j -> m_ij #obj)
        self.obj2rel_attn = nn.Linear(edge_out_channels, 1)

        # Message_sub :  rel2sub * [Relation Embedding, x_j]
        self.rel2sub = nn.Linear(node_in_channels+edge_out_channels, node_out_channels)
        self.rel2sub_attn = nn.Linear(node_out_channels, 1)
        # Message_obj :  rel2obj * [Relation Embedding, x_j]
        self.rel2obj = nn.Linear(node_in_channels+edge_out_channels, node_out_channels)
        self.rel2obj_attn = nn.Linear(node_out_channels, 1)
        # Message_skip :  skip_msg * [x_i, x_j]
        self.skip_msg = nn.Linear(2*node_in_channels, node_out_channels) # (x_i, x_j) -> m_ij #skip
        self.skip_attn = nn.Linear(node_out_channels, 1) # m_ij -> scalar


    def forward(self, edge_index, edgeskip_index, nf, ef, size=None):

        return self.propagate(edge_index, edgeskip_index, nf, ef)


    def message(self, edgeindex, x_i, x_j, x_ij, message_type, num_nodes):


        if message_type == 'sub2rel':
            message = torch.cat([x_i, x_j], -1)
            message = self.sub2rel_msg(message)
            e_ij = F.leaky_relu(self.sub2rel_attn(message))
            alpha_ij = self.softmax(e_ij, edgeindex[1, :])
            message = x_ij + alpha_ij * message

        elif message_type == 'obj2rel':
            message = torch.cat([x_i, x_j], -1)
            message = self.obj2rel_msg(message)
            e_ij = F.leaky_relu(self.obj2rel_attn(message))
            alpha_ij = self.softmax(e_ij, edgeindex[0, :])
            message = x_ij + alpha_ij * message


        elif message_type == 'rel2sub':
            message = torch.cat([x_i, x_ij], -1)
            message = self.rel2sub(message)
            e_ij = F.leaky_relu(self.rel2sub_attn(message))
            mask_e_ij = self.generate_purifier(e_ij, edgeindex, num_nodes, j=1)
            alpha_ij = self.softmax(mask_e_ij, edgeindex[0, :])
            message = alpha_ij * message

        elif message_type == 'rel2obj':
            message = torch.cat([x_i, x_ij], -1)
            message = self.rel2obj(message)
            e_ij = F.leaky_relu(self.rel2obj_attn(message))
            mask_e_ij = self.generate_purifier(e_ij, edgeindex, num_nodes, j=0)
            alpha_ij = self.softmax(mask_e_ij, edgeindex[1, :])
            message = alpha_ij * message

        elif message_type == 'skip':
            message = torch.cat([x_i, x_j], -1)
            message = self.skip_msg(message)
            e_ij = F.leaky_relu(self.skip_attn(message))
            mask_e_ij = self.generate_purifier(e_ij, edgeindex, num_nodes, j=0)
            alpha_ij = self.softmax(mask_e_ij, edgeindex[1, :])
            message = alpha_ij * message
        
        return message


    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr = None,
                  dim_size = None) -> Tensor:
        if ptr is not None:
            ptr = self.expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce = 'sum')


    def expand_left(self, src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
        for _ in range(dims + dim if dim < 0 else dim):
            src = src.unsqueeze(0)
        return src

    def propagate(self, edge_index, edgeskip_index, nf, ef, size = None):

        size = self.__check_input__(edge_index, size)


        # Otherwise, run both functions in separation.

        # Define dst(i) and src (j)
        x_i = nf[edge_index[1, :]]; x_j=nf[edge_index[0, :]]; x_ij=ef
        sub2rel_x_j = nf[edge_index[0, :]] # 수정부분 - Subject Embedding
        obj2rel_x_j = nf[edge_index[1, :]] # 수정부분 - Object Embedding
        # Generate Relation Embedding
        sub2rel_msg = self.message(edge_index, x_i, sub2rel_x_j, x_ij, 'sub2rel', nf.size(0))
        obj2rel_msg = self.message(edge_index, x_j, obj2rel_x_j, x_ij, 'obj2rel', nf.size(0))

        rel_embedding = (sub2rel_msg + obj2rel_msg) / 2

        sub_msg = self.message(edge_index, x_j, None, rel_embedding, 'rel2sub', nf.size(0))
        obj_msg = self.message(edge_index, x_i, None, rel_embedding, 'rel2obj', nf.size(0))

        # Define dst(i) and src (j)
        xskip_i = nf[edgeskip_index[1, :]]; xskip_j = nf[edgeskip_index[0, :]]
        skip_msg = self.message(edgeskip_index, xskip_i, xskip_j, None, 'skip', nf.size(0))
        
        # Aggregate
        sub_agg = self.aggregate(sub_msg, index = edge_index[0,:], ptr= None, dim_size=nf.size(0))
        obj_agg = self.aggregate(obj_msg, index = edge_index[1,:], ptr= None, dim_size=nf.size(0))
        skip_agg = self.aggregate(skip_msg, index = edgeskip_index[1, :], dim_size=nf.size(0))
        
        # Update
        sub_update = nf + sub_agg
        obj_update = nf + obj_agg
        skip_update = nf + skip_agg

        node_embedding = (sub_update + obj_update + skip_update)/3

        return node_embedding, rel_embedding

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out


    def maybe_num_nodes(self, edge_index, num_nodes=None):
        if num_nodes is not None:
            return num_nodes
        elif isinstance(edge_index, Tensor):
            return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        else:
            return max(edge_index.size(0), edge_index.size(1))


    def softmax(self, src: Tensor, index= None,
            ptr = None, num_nodes = None,
            dim: int = 0) -> Tensor:
   
        if ptr is not None:
            dim = dim + src.dim() if dim < 0 else dim
            size = ([1] * dim) + [-1]
            ptr = ptr.view(size)
            src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
            out = (src - src_max).exp()
            out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
        elif index is not None:
            N = self.maybe_num_nodes(index, num_nodes) # The number of nodes in batch
            src_max = scatter(src, index, dim, dim_size=N, reduce='max') # Find max values among the neighbor(src)
            src_max = src_max.index_select(dim, index) # fill neighbor nodes with max values.
            out = (src - src_max).exp()
            out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError

        return out / (out_sum + 1e-16)


    def generate_purifier(self, e_ij: Tensor, edge_index= None, num_nodes = None, dim: int = 0, j=0) -> Tensor:
        
        mask = -1e15*torch.ones((num_nodes, num_nodes)).cuda()

        mask[edge_index[0].detach(), edge_index[1].detach()] = e_ij.squeeze().detach()
        i1, i2 = (-1, 1) if j == 1 else (1, -1)
        if not self.is_topk:
            
            survive_n_edge = torch.round(torch.sum(mask!=-1e15, j) * self.relative_rank).long()
            sort_idx = torch.argsort(mask, j, descending=True)
            idx = torch.gather(sort_idx, j, survive_n_edge.view(i1,i2))
            indices_to_remove = mask <= torch.gather(mask, j, idx).view(i1, i2)
        else:
            indices_to_remove = mask <= torch.min(torch.topk(mask, self.topk, dim=j)[0], j)[0].view(i1, i2) # torch clamp
        
        mask[indices_to_remove] = -1e15

        mattn = mask[edge_index[0], edge_index[1]]

        return mattn.unsqueeze(1).cuda()

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)




def build_mgat_model(cfg, in_channels):
    return MaskGAT(cfg, in_channels)
