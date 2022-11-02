from calendar import c
import copy

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pysgg.modeling.make_layers import make_fc
from pysgg.utils.comm import get_rank
from pysgg.config import cfg
from pysgg.modeling.roi_heads.relation_head.model_msg_passing import (
    PairwiseFeatureExtractor,
)
from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    make_relation_confidence_aware_module,
)
from pysgg.structures.boxlist_ops import squeeze_tensor
from pysgg.modeling.utils import cat
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.utils import clamp_probs

class MessagePassingUnit_v2(nn.Module):
    def __init__(self, input_dim, filter_dim=128):
        super(MessagePassingUnit_v2, self).__init__()
        self.w = nn.Linear(input_dim, filter_dim, bias=True)
        self.fea_size = input_dim
        self.filter_size = filter_dim

    def forward(self, unary_term, pair_term):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        # print '[unary_term, pair_term]', [unary_term, pair_term]
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = torch.sigmoid(gate.sum(1))
        # print 'gate', gate
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])

        return output, gate


def reverse_sigmoid(x):
    new_x = x.clone()
    new_x[x > 0.999] = x[x > 0.999] - (x[x > 0.999].clone().detach() - 0.999)
    new_x[x < 0.001] = x[x < 0.001] + (-x[x < 0.001].clone().detach() + 0.001)
    return torch.log((new_x) / (1 - (new_x)))


class MessagePassingUnit_v1(nn.Module):
    def __init__(self, input_dim, filter_dim=64):
        """

        Args:
            input_dim:
            filter_dim: the channel number of attention between the nodes
        """
        super(MessagePassingUnit_v1, self).__init__()
        self.w = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, filter_dim, bias=True),
        )

        self.fea_size = input_dim
        self.filter_size = filter_dim

        self.gate_weight = nn.Parameter(
            torch.Tensor(
                [
                    0.5,
                ]
            ),
            requires_grad=True,
        )
        self.aux_gate_weight = nn.Parameter(
            torch.Tensor(
                [
                    0.5,
                ]
            ),
            requires_grad=True,
        )

    def forward(self, unary_term, pair_term, attn_value):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        paired_feats = torch.cat([unary_term, pair_term], 1)

        gate = torch.sigmoid(self.w(paired_feats))
        if gate.shape[1] > 1:
            gate = gate.mean(1)  # average the nodes attention between the nodes
        # print 'gate', gate
        output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])
        output *= attn_value.view(-1, 1) # Attn Value
        return output, gate


class MessageFusion(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MessageFusion, self).__init__()
        self.wih = nn.Linear(input_dim, input_dim, bias=True)
        self.whh = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout: # False
            output = F.dropout(output, training=self.training)
        return output


class LearnableRelatednessGating(nn.Module):
    def __init__(self):
        super(LearnableRelatednessGating, self).__init__()
        cfg_weight = cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.LEARNABLE_SCALING_WEIGHT
        self.alpha = nn.Parameter(torch.Tensor([cfg_weight[0]]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([cfg_weight[1]]), requires_grad=False)

    def forward(self, relness):
        relness = torch.clamp(self.alpha * relness - self.alpha * self.beta, min=0, max=1.0)
        return relness


class GSL4SGG(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        hidden_dim=1024,
        dropout=False,
        gate_width=128,
        use_kernel_function=False,
    ):
        super(GSL4SGG, self).__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        self.pooling_dim = self.pairwise_feature_extractor.pooling_dim
        # GraphCAD
        self.l2r = nn.Sequential(
            nn.Linear(3*self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 1)
        )
        self.edge_skip_alpha = nn.Parameter(torch.rand(1))
        self.n_layer = self.cfg.MODEL.ROI_RELATION_HEAD.GSL_MODULE.ITER

        self.rel_aware_on = (
            self.cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE
        )
        self.rel_aware_module_type = (
            self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD
        )

        self.num_rel_cls = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.share_parameters_each_iter = self.cfg.MODEL.ROI_RELATION_HEAD.GSL_MODULE.SHARE_PARAMETER
        
        if self.rel_aware_on:
            
            self.pretrain_pre_clser_mode = False

            self.apply_gt_for_rel_conf = self.cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT

            self.filter_the_mp_instance = (
                self.cfg.MODEL.ROI_RELATION_HEAD.GSL_MODULE.FILTER_VALID_NODE
            )
        
        # decrease the dimension before mp
        self.obj_downdim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim), # 2048
            nn.ReLU(True),
        )
        self.rel_downdim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim),
            nn.ReLU(True),
        )
        self.obj_pair2rel_fuse = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 2),
            make_fc(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )
        self.padding_feature = nn.Parameter(
            torch.zeros((self.hidden_dim)), requires_grad=False
        )
        MessagePassingUnit = MessagePassingUnit_v1
        param_set_num = self.n_layer
        if self.share_parameters_each_iter:
            param_set_num = 1
        self.gate_sub2pred = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        self.gate_obj2pred = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        self.gate_pred2sub = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        self.gate_pred2obj = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )

        self.object_msg_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(param_set_num)]
        )  #
        self.pred_msg_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(param_set_num)]
        )

        self.forward_time = 0


    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val


    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs, combine_weight):

        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]
        rel_prop_pairs_relness_batch = []

        for idx, (prop, rel_ind_i) in enumerate(zip(proposals, rel_pair_idxs)):
            rel_ind_i = copy.deepcopy(rel_ind_i)
            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)
        
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)
        
        rel_prop_pairs_relness_sorted_idx = []
        
        subj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        obj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        offset = 0
        rel_prop_pairs_all = torch.arange(len(rel_inds_batch_cat)).cuda()
        for c_w, rel_pair_idx in zip(combine_weight, rel_pair_idxs):
            subj_pred_map[rel_inds_batch_cat[offset:offset+c_w.shape[0],0], rel_prop_pairs_all[offset:offset+c_w.shape[0]]] = c_w
            obj_pred_map[rel_inds_batch_cat[offset:offset+c_w.shape[0],1], rel_prop_pairs_all[offset:offset+c_w.shape[0]]] = c_w
            offset += c_w.shape[0]
            
            

        return (
            rel_inds_batch_cat,
            subj_pred_map,
            obj_pred_map
        )

    # Here, we do all the operations out of loop, the loop is just to combine the features
    # Less kernel evoke frequency improve the speed of the model
    def prepare_message(
        self,
        target_features,
        source_features,
        select_mat,
        gate_module,
        attn_value
    ):
        feature_data = []

        transfer_list = (select_mat > 0).nonzero(as_tuple=False)

        source_indices = transfer_list[:, 1] # Predicate
        target_indices = transfer_list[:, 0] # Entity
        source_f = torch.index_select(source_features, 0, source_indices)
        target_f = torch.index_select(target_features, 0, target_indices) # Entity


        transferred_features, weighting_gate = gate_module(target_f, source_f, attn_value)
        aggregator_matrix = torch.zeros(
            (target_features.shape[0], transferred_features.shape[0]),
            dtype=weighting_gate.dtype,
            device=weighting_gate.device,
        )

        for f_id in range(target_features.shape[0]):
            if select_mat[f_id, :].data.sum() > 0:
                # average from the multiple sources
                feature_indices = squeeze_tensor(
                    (transfer_list[:, 0] == f_id).nonzero(as_tuple=False)
                )  # obtain source_relevant_idx
                # (target, source_relevant_idx)
                aggregator_matrix[f_id, feature_indices] = 1
        # (target, source_relevant_idx) @ (source_relevant_idx, feat-dim) => (target, feat-dim)
        aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)
        avg_factor = aggregator_matrix.sum(dim=1)
        vaild_aggregate_idx = avg_factor != 0
        avg_factor = avg_factor.unsqueeze(1).expand(
            avg_factor.shape[0], aggregate_feat.shape[1]
        )
        aggregate_feat[vaild_aggregate_idx] /= avg_factor[vaild_aggregate_idx]

        feature_data = aggregate_feat
        return feature_data # 각 Entity에 Aggregated Message가 존재

    def pairwise_rel_features(self, augment_obj_feat, rel_pair_idxs):
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.size(0), 2, self.hidden_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)

        obj_pair_feat4rel_rep = torch.cat(
            (head_rep[rel_pair_idxs[:, 0]], tail_rep[rel_pair_idxs[:, 1]]), dim=-1
        )

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(
            obj_pair_feat4rel_rep
        )  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(         
        self,
        inst_features,
        rel_union_features,
        proposals,
        rel_pair_inds,
        rel_gt_binarys=None,
        logger=None,
    ):

        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(
            inst_features,
            rel_union_features,
            proposals,
            rel_pair_inds,
        )
        num_obj = [len(prop) for prop in proposals]
        rel_inds_batch_cat = []
        offset = 0
        # gt_rel_idx = []
        for idx, (prop, rel_ind_i) in enumerate(zip(proposals, rel_pair_inds)):
            rel_ind_i = copy.deepcopy(rel_ind_i)
            # gt_rel_idx.append(torch.nonzero(rel_gt_binarys[idx]) + offset)
            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)
        # gt_rel_idx = torch.cat(gt_rel_idx, 0)
        n_pair = [rel_pair.shape[0] for rel_pair in rel_pair_inds]
        
        inst_feature4iter = [self.obj_downdim_fc(augment_obj_feat)]
        rel_feature4iter = [self.rel_downdim_fc(rel_feats)]        
        
        previous_edge_weight = torch.ones([rel_inds_batch_cat.shape[0]]).cuda()
        link_likelihood_list = []
        residual_weight_list = []
        for l in range(self.n_layer):
            # Parameter Share하는지 확인
            if self.share_parameters_each_iter:
                l = 0
            
            if self.cfg.MODEL.ROI_RELATION_HEAD.GSL_MODULE.INPUT_FORMAT == 'minus':
                # centroid 계산
                centroid = inst_feature4iter[l].split(num_obj)
                centroid = [torch.mean(c.view(1, -1, inst_feature4iter[l].shape[1]), dim=1).repeat(len(c),1) for c in centroid]
                residual_node_features = inst_feature4iter[l] - cat(centroid)
                residual_node_sub = residual_node_features[rel_inds_batch_cat[:,0]] # subject
                residual_node_obj = residual_node_features[rel_inds_batch_cat[:,1]] # object
                sim_vec = torch.cat((torch.abs(inst_feature4iter[l][rel_inds_batch_cat[:,0]] - inst_feature4iter[l][rel_inds_batch_cat[:,1]]), residual_node_sub, residual_node_obj), dim = 1)
            else:
                centroid = inst_feature4iter[l].split(num_obj)
                centroid = [torch.mean(c.view(1, -1, inst_feature4iter[l].shape[1]), dim=1).repeat(len(rel_pair_inds[i]),1) for i, c in enumerate(centroid)]
                centroid = cat(centroid)
                sim_vec = torch.cat((inst_feature4iter[l][rel_inds_batch_cat[:,0]], inst_feature4iter[l][rel_inds_batch_cat[:,1]], centroid), dim = 1)
                
            # Link Likelihood
            prob_score = self.l2r(sim_vec)
            pre_adj = torch.sigmoid(prob_score)
            link_likelihood_list.append(pre_adj)
            
            # Make sparse graph
            sampled_edge = self.gumbel_softmax_(pre_adj, temperature=0.6)
            
            # Residual Connection
            if self.cfg.MODEL.ROI_RELATION_HEAD.GSL_MODULE.RESIDUAL_CONNECTION:
                combine_weight = self.edge_skip_alpha * (sampled_edge.view(-1) * previous_edge_weight) + (1-self.edge_skip_alpha) * (sampled_edge.view(-1) * pre_adj.view(-1)) # 전자: 이전 Weight (sampled_edge_gt)
            else:
                combine_weight = sampled_edge.view(-1)
            previous_edge_weight = combine_weight
            residual_weight_list.append(combine_weight)
            combine_weight = combine_weight.split(n_pair)
            # Message Passing 때 Valid한 Node만 남길 것인지
            valid_inst_idx = []
            if self.filter_the_mp_instance:
                for p in proposals:
                    valid_inst_idx.append(p.get_field("pred_scores") > 0.03)
            else:
                for p in proposals:
                    valid_inst_idx.append(p.get_field("pred_scores") > -0.1)        
                            
            if len(valid_inst_idx) > 0:
                valid_inst_idx = torch.cat(valid_inst_idx, 0)
            else:
                valid_inst_idx = torch.zeros(0)
            (
                batchwise_rel_pair_inds,
                subj_pred_map,
                obj_pred_map
            ) = self._prepare_adjacency_matrix(
                proposals, rel_pair_inds, combine_weight
            )
            idx_nonzero = torch.nonzero(subj_pred_map > 0.0)
            attn_value = subj_pred_map[idx_nonzero[:,0], idx_nonzero[:,1]]
            object_sub = self.prepare_message( # Entity가 Subect일 때 Message Aggregate
                inst_feature4iter[l],
                rel_feature4iter[l],
                subj_pred_map, # pred to sub
                self.gate_pred2sub[l],
                attn_value
            )
            
            idx_nonzero = torch.nonzero(obj_pred_map > 0.0)
            attn_value = obj_pred_map[idx_nonzero[:,0], idx_nonzero[:,1]]
            object_obj = self.prepare_message( # Entity가 Object일 때 Message Aggregate
                inst_feature4iter[l],
                rel_feature4iter[l],
                obj_pred_map, # pred to obj
                self.gate_pred2obj[l],
                attn_value
            )   
            
            GRU_input_feature_object = (object_sub + object_obj) / 2.0
            inst_feature4iter.append(inst_feature4iter[l]+self.object_msg_fusion[l](GRU_input_feature_object, inst_feature4iter[l]))
            combine_weight_cat = torch.cat(combine_weight)
            indices_sub = batchwise_rel_pair_inds[:, 0]
            indices_obj = batchwise_rel_pair_inds[:, 1]

            valid_sub_inst_in_pairs = valid_inst_idx[indices_sub]
            valid_obj_inst_in_pairs = valid_inst_idx[indices_obj]
                
            valid_inst_pair_inds = (valid_sub_inst_in_pairs) & (
                valid_obj_inst_in_pairs
            )
            
            indices_sub = indices_sub[valid_inst_pair_inds]
            # attn_sub = combine_weight_cat[valid_inst_pair_inds]
            indices_obj = indices_obj[valid_inst_pair_inds]
            attn_value = combine_weight_cat[valid_inst_pair_inds]

            feat_sub2pred = torch.index_select(inst_feature4iter[l], 0, indices_sub)
            feat_obj2pred = torch.index_select(inst_feature4iter[l], 0, indices_obj)        

            valid_pairs_rel_feats = torch.index_select(
                rel_feature4iter[l],
                0,
                squeeze_tensor(valid_inst_pair_inds.nonzero(as_tuple=False)),
            )
            phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[l]( # Sub => Pred의 Message
                valid_pairs_rel_feats, feat_sub2pred, attn_value # 여기서부터 수정
            )
            phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[l]( # Obj => Pred의 Message
                valid_pairs_rel_feats, feat_obj2pred, attn_value
            )
            GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
            next_stp_rel_feature4iter = self.pred_msg_fusion[l](
                GRU_input_feature_phrase, valid_pairs_rel_feats
            )
            padded_next_stp_rel_feats = rel_feature4iter[l].clone()
            padded_next_stp_rel_feats[
                valid_inst_pair_inds
            ] += next_stp_rel_feature4iter

            rel_feature4iter.append(padded_next_stp_rel_feats)
            

        final_ent_feats = inst_feature4iter[-1]
        final_rel_feats = rel_feature4iter[-1]
        return final_ent_feats, final_rel_feats, link_likelihood_list, residual_weight_list



    # def sample_gumbel(self, shape, eps=1e-20):
    #     U = torch.rand(shape).cuda()
    #     return -Variable(torch.log(-torch.log(U + eps) + eps))


    # def gumbel_softmax_sample(self, score, temperature):
    #     y = score + self.sample_gumbel(score.size())
    #     # denominator = (torch.exp(-y[:,0]/temperature)+torch.exp(-y[:,1]/temperature))
    #     # prob_non_edge = torch.exp(-y[:,0]/temperature) / denominator
    #     # prob_edge = torch.exp(-y[:,1]/temperature) / denominator
    #     # return torch.cat([prob_non_edge.view(-1, 1), prob_edge.view(-1,1)], dim =1)
    #     y = torch.sigmoid(y/temperature)
    #     return y


    def gumbel_softmax_(self, score, temperature):
        soft_sample = RelaxedBernoulli(temperature=temperature, probs=score).sample()
        soft_sample = clamp_probs(soft_sample)
        hard_sample = QuantizeBernoulli.apply(soft_sample)
        # y = self.gumbel_softmax_sample(score, temperature)
        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        return hard_sample
    
class QuantizeBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        hard_value = soft_value.round()
        hard_value._unquantize = soft_value
        return hard_value

    @staticmethod
    def backward(ctx, grad):
        return grad