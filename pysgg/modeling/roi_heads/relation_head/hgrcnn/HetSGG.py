import sys
import json
sys.path.append('')
import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_scatter import scatter_add

from ..nn import MultiHeadAttention

from pysgg.modeling.roi_heads.relation_head.utils_motifs import encode_box_info
from pysgg.modeling.roi_heads.relation_head.model_msg_passing import PairwiseFeatureExtractor

from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier




class HetSGG(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(HetSGG, self).__init__()
        
        self.cfg = cfg
        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.RGCN.NUM_RELATION
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.num_bases = cfg.MODEL.ROI_RELATION_HEAD.RGCN.N_BASES
        self.dim = cfg.MODEL.ROI_RELATION_HEAD.RGCN.H_DIM
        self.score_update_step = cfg.MODEL.ROI_RELATION_HEAD.RGCN.SCORE_UPDATE_STEP
        self.feat_update_step = cfg.MODEL.ROI_RELATION_HEAD.RGCN.FEATURE_UPDATE_STEP
        self.geometry_feat_dim = 128
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.RGCN.H_DIM
        num_classes_obj = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes_pred = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        if 'vg' in self.cfg.DATA_DIR:
            self.vg_cat_dict = json.load(open(f'{cfg.DATA_DIR}/{cfg.MODEL.ROI_RELATION_HEAD.RGCN.CATEGORY_FILE}.json', 'r'))
        else:
            self.vg_cat_dict = json.load(open(cfg.DATA_DIR+'/annotations/OI-SGG-Category.json', 'r'))

        self.vg_map_arr = self.compute_category_mapping_array()

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = "sgdet"
            
        if cfg.MODEL.ROI_RELATION_HEAD.RGCN.IS_ASSIGN_GT_CAT:
            self.mode = 'predcls'

        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}

        self.gt2pred = []

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

        self.rel_classifier = build_classifier(self.hidden_dim, num_classes_pred) # Linear Layer
        self.obj_classifier = build_classifier(self.hidden_dim, num_classes_obj)

        if self.feat_update_step > 0:
            self.rgcn = RGCNLayer(self.dim, self.dim, self.dim, self.dim, self.num_bases, self.n_reltypes, cfg=cfg)
            
        if self.score_update_step > 0:
            self.rgcn_score = RGCNLayer(num_classes_obj, num_classes_pred, num_classes_obj, num_classes_pred, num_bases=self.num_bases, num_relations=self.n_reltypes, cfg=cfg)

        self.init_classifier_weight()


    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()


    def compute_category_mapping_array(self):
        key = list(self.vg_cat_dict['labelidx_to_catidx'].keys())
        value = list(self.vg_cat_dict['labelidx_to_catidx'].values())
        vg_map_arr = np.array([key, value], dtype = int).transpose()
        vg_map_arr = np.array(vg_map_arr[np.argsort(vg_map_arr[:,0])])
        return torch.LongTensor(vg_map_arr).cuda()


    def forward(self,      
        inst_features,
        rel_union_features,
        proposals,
        rel_pair_inds,
        rel_gt_binarys=None,
        logger=None, is_training=True):

        nf, ef = self.pairwise_feature_extractor( 
            inst_features,
            rel_union_features,
            proposals,
            rel_pair_inds,
        )
        g_rel, etype_rel, etype_rel_inv = self._get_map_idxs(proposals, rel_pair_inds, is_training)

        # edge_norm_rel = self.edge_normalization(g_rel, etype_rel, nf.size(0), self.n_reltypes)
        # edge_norm_rel_inv = self.edge_normalization(g_rel[[1, 0], :], etype_rel_inv, nf.size(0), self.n_reltypes)
        edge_norm_rel = None
        edge_norm_rel_inv = None

        edge_type_list = [etype_rel, etype_rel_inv]
        edge_norm_list = [edge_norm_rel, edge_norm_rel_inv]

        nf = self.obj_embedding(nf) # Node Feature
        ef = self.rel_embedding(ef) # Edge Feature

        for _ in range(self.feat_update_step):
            nf, ef = self.rgcn(g_rel, nf, ef, edge_type_list, edge_norm_list)
            nf = F.elu(nf)
            ef = F.elu(ef)

        # Relationship Classifier
        pred_class_logits = self.rel_classifier(ef)
        obj_class_logits = self.obj_classifier(nf)

        pred_logits = [pred_class_logits]

        # Logit Layer
        for _ in range(self.score_update_step):
            obj_class_logits, pred_class_logits = self.rgcn_score(g_rel, obj_class_logits, pred_class_logits, edge_type_list, edge_norm_list)
            obj_class_logits = F.elu(obj_class_logits)
            pred_class_logits = F.elu(pred_class_logits)
            pred_logits.append(pred_class_logits)


        return obj_class_logits, pred_class_logits, None


    def spatial_embedding(self, proposals):
        """
        Compute the Spatial Information for each proposal
        """
        pos_embed = []
        for proposal in proposals:
            pos_embed.append(encode_box_info([proposals,]))
        pos_embed = torch.cat(pos_embed)
        return pos_embed


    def edge_normalization(self, rel_inds, etype, n_entity, n_rels):
        one_hot = F.one_hot(etype, num_classes = n_rels).to(torch.float)
        deg = scatter_add(one_hot, rel_inds[0], dim = 0, dim_size = n_entity)
        index = etype + torch.arange(len(rel_inds[0])).cuda() * (n_rels)
        edge_norm = 1 / deg[rel_inds[0]].view(-1)[index]
        return edge_norm

  

    def _get_map_idxs(self, proposals, rel_pair_inds, is_training):
        """
        (P,P) : 0, (P,H) : 1, (P,A) : 2
        (H,P) : 3, (H,H) : 4, (H,A) : 5
        (A,P) : 6, (A,H) : 7, (A,A) : 8
        """
        offset = 0
        rel_inds = []

        edge_types = []
        edge_types_inv = []
        
        for proposal, rel_pair_ind in zip(proposals, rel_pair_inds):

            # Generate Graph
            rel_ind_i = rel_pair_ind.detach().clone()
            rel_ind_i += offset
            rel_inds.append(rel_ind_i)

            # Get Node Type for each entity
            if self.mode == 'sgcls':
                if self.cfg.MODEL.ROI_RELATION_HEAD.RGCN.CLASS_AGG == 'max':
                    proposal_category = self.vg_map_arr[proposal.extra_fields['pred_labels'].long(), 1]
                else:
                    proposal_category = proposal.extra_fields['category_scores'].max(1)[1].detach()
            elif self.mode == 'sgdet':
                # if is_training: # Training => Give GT Type
                #     proposal_category = self.vg_map_arr[proposal.extra_fields['labels'].long(), 1]
                # else:
                proposal_category = proposal.extra_fields['category_scores'].max(1)[1].detach()
            else:
                proposal_category = self.vg_map_arr[proposal.extra_fields['labels'].long(), 1] # if 'predcls', give GT Category
            

            edge_type = torch.LongTensor([self.obj2rtype[(s.item(),d.item())] for (s, d) in proposal_category[rel_ind_i.detach()-offset]])
            edge_type_inv = torch.LongTensor([self.obj2rtype[(d.item(), s.item())] for (s, d) in proposal_category[rel_ind_i.detach()-offset]])

            edge_types.append(edge_type)
            edge_types_inv.append(edge_type_inv)

            offset += len(proposal)

        rel_inds = torch.cat(rel_inds, 0).T
       
        edge_types = torch.cat(edge_types).cuda()
        edge_types_inv = torch.cat(edge_types_inv).cuda()

        return rel_inds, edge_types, edge_types_inv


class RGCNLayer(MessagePassing):

    def __init__(self, node_in_channels, edge_in_channels, node_out_channels, edge_out_channels, num_bases, num_relations, cfg, bias=False):
        super(RGCNLayer, self).__init__()

        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.node_out_channels = node_out_channels
        self.edge_out_channels = edge_out_channels
        self.bias = bias

        self.num_relations = num_relations
        self.num_bases = num_bases
        self.n_ntypes = int(sqrt(self.num_relations))
        self.use_attn_agg = cfg.MODEL.ROI_RELATION_HEAD.RGCN.USE_ATTN_AGG

        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}
        self.rtype2obj = dict(zip(self.obj2rtype.values(), self.obj2rtype.keys()))

        # Check Dimension
        self.sub2rel_basis = nn.Parameter(torch.Tensor(num_bases, node_in_channels, edge_out_channels)) # *2
        self.sub2rel_att =  nn.Parameter(torch.Tensor(num_relations, num_bases))

        self.entity2rel_attn = nn.Linear(edge_out_channels, 1, bias = False)

        
        self.obj2rel_basis = nn.Parameter(torch.Tensor(num_bases, node_in_channels, edge_out_channels)) # *2
        self.obj2rel_att =  nn.Parameter(torch.Tensor(num_relations, num_bases)) 
        # self.obj2rel_attn = nn.ModuleList([nn.Linear(edge_out_channels, 1) for _ in range(self.num_relations)])

        self.rel2obj_basis = nn.Parameter(torch.Tensor(num_bases, edge_in_channels, node_out_channels)) # 2번 째 node_in_channels + edge_in_channels
        self.rel2obj_att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.rel2obj_attn = nn.ModuleList([nn.Linear(node_out_channels, 1) for _ in range(self.num_relations)])
        
        self.rel2sub_basis = nn.Parameter(torch.Tensor(num_bases, edge_in_channels, node_out_channels))
        self.rel2sub_att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.rel2sub_attn = nn.ModuleList([nn.Linear(node_out_channels, 1) for _ in range(self.num_relations)])

        if self.use_attn_agg:
            self.sub2rel_salayer = MultiHeadAttention(n_head=4, d_model=edge_out_channels, d_k=edge_out_channels, d_v=edge_out_channels, 
                                    dropout=0.1, normalize_before=True)
            self.obj2rel_salayer = MultiHeadAttention(n_head=4, d_model=edge_out_channels, d_k=edge_out_channels, d_v=edge_out_channels, 
                                    dropout=0.1, normalize_before=True)
            self.rel2sub_salayer = MultiHeadAttention(n_head=4, d_model=node_out_channels, d_k=node_out_channels, d_v=node_out_channels, 
                                    dropout=0.1, normalize_before=True)
            self.rel2obj_salayer = MultiHeadAttention(n_head=4, d_model=node_out_channels, d_k=node_out_channels, d_v=node_out_channels, 
                                    dropout=0.1, normalize_before=True)

        # if self.bias: # True
        #     self.sub2rel_bias = nn.Parameter(torch.Tensor(edge_out_channels))
        #     self.obj2rel_bias = nn.Parameter(torch.Tensor(edge_out_channels))
        #     self.rel2sub_bias = nn.Parameter(torch.Tensor(node_out_channels))
        #     self.rel2obj_bias = nn.Parameter(torch.Tensor(node_out_channels))
        #     self.skip_bias = nn.Parameter(torch.Tensor(node_out_channels))

        # else:
            # self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        
        layers = [self.sub2rel_basis.data, self.sub2rel_att,
                    self.obj2rel_basis.data, self.obj2rel_att,
                    self.rel2sub_basis.data, self.rel2sub_att,
                    self.rel2obj_basis.data, self.rel2obj_att,
                    self.entity2rel_attn.weight.data
                    ]

        for layer in layers:
            nn.init.xavier_uniform_(layer)

        if self.bias:

            biases = [self.sub2rel_bias, self.obj2rel_bias, self.rel2sub_bias, self.rel2obj_bias, self.skip_bias]

            for bias in biases:
                stdv = 1. / sqrt(bias.shape[0])
                bias.data.uniform_(-stdv, stdv)
            

    def forward(self, edge_index, nf, ef, edge_type_list, edge_norm_list, size=None):

        return self.propagate(edge_index, nf, ef, edge_type_list, edge_norm_list)



    def message(self, edgeindex_i, x_i, x_j, x_ij, edge_type, message_type):
        
            # Generate Edge Mask for RGCN
        edge_mask = torch.zeros((edge_type.size(0), self.num_relations)).cuda()
        for i in range(self.num_relations):
            edge_mask[:, i] += (edge_type == i).float().cuda()
            
        if message_type == 'sub2rel':

            W = torch.matmul(self.sub2rel_att, self.sub2rel_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.node_in_channels, self.edge_out_channels)

            src_feat = torch.mul(x_j.unsqueeze(2), edge_mask.unsqueeze(1))
            # dest_feat = torch.mul(x_i.unsqueeze(2), edge_mask.unsqueeze(1))
            message = src_feat
            # message = torch.cat([src_feat, dest_feat], 1)

            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...]
                m_r = torch.matmul(message[..., rel], W_r)
                Message.append(m_r)

        elif message_type == 'obj2rel':
            W = torch.matmul(self.obj2rel_att, self.obj2rel_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.node_in_channels, self.edge_out_channels)

            src_feat = torch.mul(x_j.unsqueeze(2), edge_mask.unsqueeze(1)) # j가 src로 Neighbor
            # dest_feat = torch.mul(x_i.unsqueeze(2), edge_mask.unsqueeze(1))
            # message = torch.cat([src_feat, dest_feat], 1)
            message = src_feat

            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...] # 1024, 1024 # x_j : 8192, 1024  edge_mask: 8192, 16
                # m_r = torch.matmul(message[..., rel], W_r) + self.obj2rel_bias
                m_r = torch.matmul(message[..., rel], W_r)

                Message.append(m_r)  
            

        elif message_type == 'rel2sub':
            W = torch.matmul(self.rel2sub_att, self.rel2sub_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.edge_in_channels, self.node_out_channels)

            src_feat = torch.mul(x_ij.unsqueeze(2), edge_mask.unsqueeze(1)) # j가 src로 Neighbor
            # dest_feat = torch.mul(x_i.unsqueeze(2), edge_mask.unsqueeze(1))
            # message = torch.cat([src_feat, dest_feat], 1)
            message = src_feat
            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...] # 1024, 1024 # x_j : 8192, 1024  edge_mask: 8192, 16
                # m_r = torch.matmul(message[..., rel], W_r) + self.rel2sub_bias
                m_r = torch.matmul(message[..., rel], W_r)
                eij_r = F.leaky_relu(self.rel2sub_attn[rel](m_r))
                eij_r += (edge_mask[:, rel, None]-1) * 1e8
                alpha_ij_r = self.softmax(eij_r, edgeindex_i)
                alpha_ij_r *= edge_mask[:, rel, None]
                Message.append(alpha_ij_r*m_r)  ## Compute Attention for each relation r respectively.

        elif message_type == 'rel2obj':
            W = torch.matmul(self.rel2obj_att, self.rel2obj_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.edge_in_channels, self.node_out_channels)

            src_feat = torch.mul(x_ij.unsqueeze(2), edge_mask.unsqueeze(1)) # j가 src로 Neighbor
            # dest_feat = torch.mul(x_i.unsqueeze(2), edge_mask.unsqueeze(1))
            # message = torch.cat([src_feat, dest_feat], 1)
            message = src_feat
            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...] # 1024, 1024 # x_j : 8192, 1024  edge_mask: 8192, 16
                # m_r = torch.matmul(message[..., rel], W_r) + self.rel2obj_bias
                m_r = torch.matmul(message[..., rel], W_r)
                eij_r = F.leaky_relu(self.rel2obj_attn[rel](m_r))
                eij_r += (edge_mask[:, rel, None]-1) * 1e8
                alpha_ij_r = self.softmax(eij_r, edgeindex_i)
                alpha_ij_r *= edge_mask[:, rel, None]
                Message.append(alpha_ij_r*m_r)  ## Compute Attention for each relation r respectively.

        return torch.stack(Message, -1)


    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr = None,
                  dim_size = None) -> Tensor:
        if ptr is not None:
            ptr = self.expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            # return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce = 'sum')
            return segment_csr(inputs, ptr, reduce=self.aggr)


    def expand_left(self, src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
        for _ in range(dims + dim if dim < 0 else dim):
            src = src.unsqueeze(0)
        return src


    def propagate(self, edge_index, nf, ef, edge_type_list, edge_norm_list, size=None):
        size = self.__check_input__(edge_index, size)
   
        # Define dst(i) and src (j)
        x_i = nf[edge_index[1, :]]; x_j=nf[edge_index[0, :]]; x_ij=ef

        sub2rel_x_j = nf[edge_index[0, :]] # 수정부분 - Subject Embedding
        obj2rel_x_j = nf[edge_index[1, :]] # 수정부분 - Object Embedding

        edge_type_rel, edge_type_rel_inv = edge_type_list

        # Generate Relation Embedding
        sub2rel_msg = torch.sum(self.message(edge_index[1, :], None, sub2rel_x_j, x_ij, edge_type_rel, 'sub2rel'), -1)
        obj2rel_msg = torch.sum(self.message(edge_index[0, :], None, obj2rel_x_j, x_ij, edge_type_rel_inv, 'obj2rel') , -1)

        rel_embedding = x_ij + (sub2rel_msg+obj2rel_msg)/2
        
        sub_msg = self.message(edge_index[0, :], x_i, None, rel_embedding, edge_type_rel_inv, 'rel2sub')
        obj_msg = self.message(edge_index[1, :], x_j, None, rel_embedding, edge_type_rel, 'rel2obj')

        # Aggregate
        """
        Option 1: Node-AGG: ATTN*sum / REL-AGG: relation * edge_norm
        Option 2: Node-AGG: ATTN*sum / REL-AGG: relation * REL_ATTN
        Option 3: Node-Rel AGG: ATTn* sum
        """
        node_mask_sub = torch.zeros((nf.size(0), self.num_relations)).cuda()
        node_mask_obj = torch.zeros((nf.size(0), self.num_relations)).cuda()
        sub_agg = []
        obj_agg = []

        for rel in range(self.num_relations):
            sub_agg.append(self.aggregate(sub_msg[..., rel], index = edge_index[0, :], ptr= None, dim_size=nf.size(0)))
            obj_agg.append(self.aggregate(obj_msg[..., rel], index = edge_index[1, :], ptr= None, dim_size=nf.size(0)))
            node_mask_sub[:, rel] += self.aggregate((edge_type_rel_inv == rel).float()[..., None], index=edge_index[0, :], ptr=None, dim_size=nf.size(0)).squeeze()
            node_mask_obj[:, rel] += self.aggregate((edge_type_rel == rel).float()[..., None], index=edge_index[1, :], ptr=None, dim_size=nf.size(0)).squeeze()
       
        node_mask_sub_gt = node_mask_sub.gt(0).float()
        node_mask_obj_gt = node_mask_obj.gt(0).float()

        sub_agg = torch.stack(sub_agg, -1) # node_size * n_out_channel * rel
        obj_agg = torch.stack(obj_agg, -1)  # node_size * n_out_channel * rel
        
        node_mask_sub_sum = node_mask_sub_gt.sum(1).view(-1,1)
        node_mask_sub_sum[node_mask_sub_sum == 0.0] = 1.0
        node_mask_obj_sum = node_mask_obj_gt.sum(1).view(-1,1)
        node_mask_obj_sum[node_mask_obj_sum == 0.0] = 1.0
        sub_agg = torch.sum(sub_agg, -1) / node_mask_sub_sum
        obj_agg = torch.sum(obj_agg, -1) / node_mask_obj_sum

        # Update
        node_embedding = nf + (sub_agg + obj_agg)/2

        return node_embedding, rel_embedding


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
            N = self.maybe_num_nodes(index, num_nodes)
            src_max = scatter(src, index, dim, dim_size=N, reduce='max')
            src_max = src_max.index_select(dim, index)
            out = (src - src_max).exp()
            out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError

        return out / (out_sum + 1e-16)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
