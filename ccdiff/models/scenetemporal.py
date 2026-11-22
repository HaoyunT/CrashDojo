#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import math
import os
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccdiff.models.context_encoders import MapEncoderPtsMA
from ccdiff.models.diffuser_helpers import SinusoidalPosEmb


def apply_directed_interaction_edge_cut(neighbor_mask, edge_cut_mask):
    """Mask receiver<-sender attention edges for every modeled time step."""
    if edge_cut_mask is None:
        return neighbor_mask
    if neighbor_mask.ndim != 4 or edge_cut_mask.ndim != 3:
        raise ValueError("expected neighbor [B,M,M,T] and edge-cut [B,M,M] masks")
    if neighbor_mask.shape[:3] != edge_cut_mask.shape:
        raise ValueError("edge-cut mask shape does not match neighbor mask")
    return neighbor_mask | edge_cut_mask[..., None].to(
        device=neighbor_mask.device, dtype=torch.bool
    )


def apply_semantic_assertive_token(embeddings, receiver_mask, scale=1.0):
    """Apply the fixed receiver token used by M2-e fine-tuning.

    This is conditioning only: no sampled action or trajectory coordinate is
    edited, and no parameter is introduced into the checkpoint contract.
    """
    if receiver_mask is None:
        return embeddings
    if embeddings.ndim != 4 or receiver_mask.ndim != 2:
        raise ValueError("expected embeddings [T,B,M,D] and receiver mask [B,M]")
    if embeddings.shape[1:3] != receiver_mask.shape:
        raise ValueError("assertive receiver mask shape disagrees with decoder")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("assertive token scale must be finite and positive")
    channel = torch.arange(
        1, embeddings.shape[-1] + 1,
        device=embeddings.device,
        dtype=torch.float32,
    )
    code = torch.sin(channel * np.sqrt(2.0)) + torch.cos(
        channel * np.sqrt(3.0)
    )
    code = code - code.mean()
    code = code / code.square().mean().clamp_min(1e-8).sqrt()
    receiver = receiver_mask.to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    bias = receiver.unsqueeze(0).unsqueeze(-1) * code.to(
        dtype=embeddings.dtype
    ).view(1, 1, 1, -1)
    return embeddings + bias * scale


def apply_directed_interaction_edge_delay(neighbor_feat, edge_delay_steps):
    """Delay selected receiver<-sender edge features without editing trajectories.

    ``edge_delay_steps`` is an integer tensor with shape ``[B, M, M]``.  A
    positive value makes the receiver attend to an older representation of
    that sender.  The beginning of the sequence is padded by repeating the
    earliest available representation, so tensor shapes and availability
    contracts remain unchanged.
    """
    if edge_delay_steps is None:
        return neighbor_feat
    if neighbor_feat.ndim != 5 or edge_delay_steps.ndim != 3:
        raise ValueError("expected neighbor [B,M,M,T,K] and delay [B,M,M]")
    if neighbor_feat.shape[:3] != edge_delay_steps.shape:
        raise ValueError("edge-delay tensor shape does not match neighbor features")
    if torch.any(edge_delay_steps < 0):
        raise ValueError("edge-delay steps must be non-negative")

    num_steps = neighbor_feat.shape[-2]
    time_index = torch.arange(num_steps, device=neighbor_feat.device)
    source_index = time_index.view(1, 1, 1, num_steps) - edge_delay_steps.to(
        device=neighbor_feat.device, dtype=torch.long
    ).unsqueeze(-1)
    source_index = source_index.clamp(min=0, max=num_steps - 1)
    gather_index = source_index.unsqueeze(-1).expand_as(neighbor_feat)
    return torch.gather(neighbor_feat, dim=-2, index=gather_index)


def apply_directed_future_intent_hold(
    neighbor_fut_feat, neighbor_hist_feat, future_intent_hold_mask
):
    """Replace only selected directed future beliefs with last observed state.

    The model still receives the untouched real history for every edge. For a
    selected ``receiver <- sender`` edge, only its denoising-time future
    interaction feature is replaced by the sender's final observed relative
    state repeated over the prediction horizon. No sampled action or trajectory
    coordinate is edited.
    """
    if future_intent_hold_mask is None:
        return neighbor_fut_feat
    if neighbor_fut_feat.ndim != 5 or neighbor_hist_feat.ndim != 5:
        raise ValueError("expected future/history neighbor features [B,M,M,T,K]")
    if neighbor_fut_feat.shape[:3] != neighbor_hist_feat.shape[:3]:
        raise ValueError("future/history directed edge shapes disagree")
    if neighbor_fut_feat.shape[-1] != neighbor_hist_feat.shape[-1]:
        raise ValueError("future/history neighbor feature widths disagree")
    if future_intent_hold_mask.ndim != 3:
        raise ValueError("expected future-intent hold mask [B,M,M]")
    if neighbor_fut_feat.shape[:3] != future_intent_hold_mask.shape:
        raise ValueError("future-intent hold mask shape does not match edges")
    if neighbor_hist_feat.shape[-2] < 1:
        raise ValueError("future-intent hold requires at least one history step")

    held = neighbor_hist_feat[..., -1:, :].expand(
        *neighbor_hist_feat.shape[:3], neighbor_fut_feat.shape[-2],
        neighbor_hist_feat.shape[-1]
    )
    mask = future_intent_hold_mask.to(
        device=neighbor_fut_feat.device, dtype=torch.bool
    )[..., None, None]
    return torch.where(mask, held, neighbor_fut_feat)


def apply_directed_yield_belief(
    future_positions,
    future_speeds,
    last_observed_positions,
    initial_speeds,
    yield_deceleration_mps2,
    dt,
):
    """Install a directed, physically parameterized yielding *belief*.

    For a selected ``receiver <- sender`` edge, the receiver internally
    extrapolates the sender along its otherwise unchanged predicted chord while
    assuming a constant, non-negative deceleration.  This function operates on
    relation features before they enter the SceneTransformer.  It never edits
    the sampled action bank or any executed trajectory.

    All tensors use directed ``[batch, receiver, sender, ...]`` order.  A zero
    entry in ``yield_deceleration_mps2`` is exact identity.
    """
    if future_positions.ndim != 5 or future_positions.shape[-1] != 2:
        raise ValueError("future positions must be [B,M,M,T,2]")
    if future_speeds.shape != future_positions.shape[:-1]:
        raise ValueError("future speed shape does not match future positions")
    if last_observed_positions.shape != future_positions.shape[:3] + (2,):
        raise ValueError("last observed positions must be [B,M,M,2]")
    if initial_speeds.shape != future_positions.shape[:3]:
        raise ValueError("initial speeds must be [B,M,M]")
    if yield_deceleration_mps2.shape != future_positions.shape[:3]:
        raise ValueError("yield-deceleration tensor must be [B,M,M]")
    if float(dt) <= 0.0:
        raise ValueError("yield-belief dt must be positive")
    if torch.any(yield_deceleration_mps2 < 0):
        raise ValueError("yield deceleration must be non-negative")

    deceleration = yield_deceleration_mps2.to(
        device=future_positions.device, dtype=future_positions.dtype
    )
    active = deceleration > 0
    if not bool(active.any().item()):
        return future_positions, future_speeds

    horizon = future_positions.shape[-2]
    time = torch.arange(
        1, horizon + 1, device=future_positions.device,
        dtype=future_positions.dtype,
    ) * float(dt)
    last_position = last_observed_positions.to(
        device=future_positions.device, dtype=future_positions.dtype
    )
    last_speed = initial_speeds.to(
        device=future_positions.device, dtype=future_positions.dtype
    ).clamp_min(0.0)

    assumed_distance = (
        last_speed.unsqueeze(-1) * time
        - 0.5 * deceleration.unsqueeze(-1) * time.square()
    ).clamp_min(0.0)
    assumed_speed = (
        last_speed.unsqueeze(-1) - deceleration.unsqueeze(-1) * time
    ).clamp_min(0.0)

    displacement = future_positions - last_position.unsqueeze(-2)
    original_distance = torch.linalg.vector_norm(displacement, dim=-1)
    scale = torch.minimum(
        torch.ones_like(original_distance),
        assumed_distance / original_distance.clamp_min(1.0e-6),
    )
    believed_positions = (
        last_position.unsqueeze(-2) + displacement * scale.unsqueeze(-1)
    )
    believed_speeds = torch.minimum(
        future_speeds, assumed_speed.to(dtype=future_speeds.dtype)
    )

    position_mask = active[..., None, None]
    speed_mask = active[..., None]
    return (
        torch.where(position_mask, believed_positions, future_positions),
        torch.where(speed_mask, believed_speeds, future_speeds),
    )


def expand_directed_attention_mask(edge_mask, num_steps, num_heads, device):
    """Expand a ``receiver <- sender`` mask for batched multi-head attention.

    PyTorch flattens attention masks in ``(batch, head)`` order.  SceneTransformer
    additionally folds time into the attention batch, so the required order is
    ``(scene, time, head, receiver, sender)``.  Keeping this conversion in one
    helper prevents silently reversing the direction or interleaving scenes.
    """
    if edge_mask is None:
        return None
    if edge_mask.ndim != 3 or edge_mask.shape[-1] != edge_mask.shape[-2]:
        raise ValueError("expected directed edge mask [B,M,M]")
    if int(num_steps) <= 0 or int(num_heads) <= 0:
        raise ValueError("attention time/head counts must be positive")
    batch_size, num_agents, _ = edge_mask.shape
    return (
        edge_mask.to(device=device, dtype=torch.bool)
        .view(batch_size, 1, 1, num_agents, num_agents)
        .expand(batch_size, int(num_steps), int(num_heads), num_agents, num_agents)
        .reshape(batch_size * int(num_steps) * int(num_heads), num_agents, num_agents)
    )

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class SceneTransformer(nn.Module):
    '''
    SceneTransformer Class.
    '''
    def __init__(self, d_k=128, d_edge=32, L_enc=1, dropout=0.0, 
                 k_attr=7, neighbor_inds=[0,1,2,3], edge_attr_separation=[], 
                 map_attr=2, num_heads=16, L_dec=1, tx_hidden_size=384, use_map_lanes=False, 
                 transition_dim=6, output_dim=2, agent_hist_embed_method='transformer', 
                 neigh_hist_embed_method=None, map_embed_method='transformer', 
                 interaction_edge_speed_repr='rel_vel', single_cond_feat=False, 
                 mask_social_interaction=False, mask_edge=False, social_attn_radius=50, 
                 all_interactive_social=False, mask_time=True, layer_num_per_edge_decoder=1, 
                 attn_combination='sum'):
        '''
        d_edge: applies only when neigh_hist_embed_method is 'edge_interaction'

        k_attr: number of attributes of history trajectories
        neighbor_inds: neighbor indices to be used for edge modeling (default correspondence: x, y, cos, sin)
        edge_attr_separation: if not empty, the edge attributes will be separated into different edge attrs before encoding.
        map_attr: number of points per road segment
        output_dim: output final dimension of the prediction

        tx_hidden_size: hidden size of FNN for each transformer encoder
        use_map_lanes: if map is used
        data_centric: 'agent' or 'scene'. If 'agent', an agent-level mask applies to social attention layer
        
        single_cond_feat: if True, use a single conditional feature consists of map, agent_hist, and neigh_hist.

        all_interactive_social: if True, all social attention layer will involve interactions. If False, only the first social attention layer will involve interactions.
        mask_time: if True, mask future time steps in the social attention layer
        layer_num_per_edge_decoder: number of layers per edge decoder

        attn_combination: how to combine attention and x.
        '''
        super(SceneTransformer, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.k_attr = k_attr
        self.map_attr = map_attr
        self.neighbor_inds = neighbor_inds
        self.edge_attr_separation = edge_attr_separation
        self.d_k = d_k
        self.d_edge = d_edge
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec = L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_lanes = use_map_lanes

        self.transition_dim = transition_dim
        self.output_dim = output_dim
        self.agent_hist_embed_method = agent_hist_embed_method
        self.neigh_hist_embed_method = neigh_hist_embed_method
        self.map_embed_method = map_embed_method
        self.interaction_edge_speed_repr = interaction_edge_speed_repr
        self.single_cond_feat = single_cond_feat
        self.mask_social_interaction = mask_social_interaction
        self.mask_edge = mask_edge
        self.social_attn_radius = social_attn_radius
        self.all_interactive_social = all_interactive_social
        self.mask_time = mask_time
        self.layer_num_per_edge_decoder = layer_num_per_edge_decoder
        self.attn_combination = attn_combination

        # not encode map when it is already included in cond feat
        if self.single_cond_feat or self.map_embed_method == 'cnn_local_patch':
            self.use_map_lanes = False



        # ============================== Neighbor ENCODERS ==============================
        if len(self.edge_attr_separation) == 0:
            self.neighbor_hist_feat_encoder = nn.Sequential(
                init_(nn.Linear(len(neighbor_inds), self.d_edge)),
                nn.Mish(),
                init_(nn.Linear(self.d_edge, self.d_edge)),
            )
            self.neighbor_fut_feat_encoder = nn.Sequential(
                init_(nn.Linear(len(neighbor_inds), self.d_edge)),
                nn.Mish(),
                init_(nn.Linear(self.d_edge, self.d_edge)),
            )
        else:
            self.neighbor_hist_feat_encoder = MultipleInputEmbedding(in_channels=[len(attrs) for attrs in self.edge_attr_separation], out_channel=self.d_edge)
            self.neighbor_fut_feat_encoder = self.neighbor_hist_feat_encoder

        # ============================== INPUT ENCODERS ==============================
        # agent history input encoder
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        # agent noise input encoder
        input_dim = self.transition_dim
        if self.neigh_hist_embed_method == 'interaction_edge_and_input':
            input_dim += 32
        if self.map_embed_method == 'cnn_local_patch':
            input_dim += 32
        self.agents_noisy_input_encoder = nn.Sequential(init_(nn.Linear(input_dim, self.d_k)))
        # ============================== Time ENCODER ==============================
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.d_k),
            nn.Linear(self.d_k, self.d_k * 4),
            nn.Mish(),
            nn.Linear(self.d_k * 4, self.d_k),
        )
        # ============================== AutoBot-Joint ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self.use_map_lanes:
            self.map_encoder = MapEncoderPtsMA(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        # ============================== AutoBot-Joint DECODER ==============================

        self.self_temporal_attn_decoder_layers = []
        self.social_attn_decoder_layers = []
        self.temporal_attn_decoder_layers = []
        for i in range(self.L_dec):
            tx_self_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.self_temporal_attn_decoder_layers.append(nn.TransformerEncoder(tx_self_encoder_layer, num_layers=2))

            tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))

            # agent-agent coord curstomized social attention
            if self.neigh_hist_embed_method == 'interaction_edge':
                if i == 0 or self.all_interactive_social:
                    tx_social_decoder_layer = SocialDecoderLayer(embed_dim=self.d_k, edge_dim=self.d_edge, num_heads=self.num_heads, dropout=self.dropout, attn_combination=self.attn_combination)
                    self.social_attn_decoder_layers.append(SocialTransformerDecoder(decoder_layer=tx_social_decoder_layer, num_layers=self.layer_num_per_edge_decoder, norm=nn.LayerNorm(self.d_k)))
                else:
                    tx_social_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
                    self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_social_encoder_layer, num_layers=1))
            else: # agent-coord naive self social attention
                tx_social_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
                self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_social_encoder_layer, num_layers=1))

        # self.temporal_attn_decoder_map_layers = nn.ModuleList(self.temporal_attn_decoder_map_layers)
        self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
        self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)
        self.self_temporal_attn_decoder_layers = nn.ModuleList(self.self_temporal_attn_decoder_layers)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, self.output_dim)))

        self.train()

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T_obs, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_obs, B, M, d_k)
        '''        
        T_obs, B, M, _ = agents_emb.shape
        # (B, T_obs, M) -> (B*M, T_obs)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.

        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * M, -1)),
                                src_key_padding_mask=agent_masks)
        return agents_temp_emb.view(T_obs, B, M, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer, aux_info=None):
        '''
        :param agents_emb: (T_obs, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_obs, B, M, d_k)
        '''
        T_obs, B, M, _ = agents_emb.shape
        full_visibility_mask = None if aux_info is None else aux_info.get(
            "interaction_full_visibility_cut_mask"
        )
        agent_interaction_mask = expand_directed_attention_mask(
            full_visibility_mask, T_obs, self.num_heads, agents_emb.device
        )
        if self.mask_social_interaction:
            self_only_mask = (1-torch.eye(M, device=agents_emb.device)).bool()
            if agent_interaction_mask is None:
                agent_interaction_mask = self_only_mask
            else:
                agent_interaction_mask = agent_interaction_mask | self_only_mask
        # (T_obs, B, M, d_k) -> (M, B*T_obs, d_k)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(M, B * T_obs, -1)
        
        # Original: agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, M))
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.reshape(-1, M), mask=agent_interaction_mask)
        agents_soc_emb = agents_soc_emb.view(M, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    
    def self_temporal_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T_all, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_all, B, M, d_k)
        '''        
        T_all, B, M, _ = agents_emb.shape
        T_obs = agent_masks.shape[1]
        T_fut = T_all - T_obs
        # (B, T_obs, M) -> (B*M, T_obs)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        # padding with current availability for future steps
        # (B*M, T_obs) -> (B*M) -> (B*M, 1) -> (B*M, T_fut)
        agent_masks_fut = agent_masks[:,-1].unsqueeze(-1).repeat(1,T_fut)
        # (B*M, T_obs), (B*M, T_fut) -> (B*M, T_all)
        agent_masks = torch.cat([agent_masks, agent_masks_fut], dim=-1)

        # Ensure agent's that don't exist don't throw NaNs.
        agent_masks[:, T_obs-1][agent_masks.sum(-1) == T_all] = False  
        
        if self.mask_time:
            agent_masks[:, 0] = False  # Ensure agent's that don't exist don't throw NaNs.
            # [T_all, T_all], time mask for only later steps consider earlier steps.
            time_masks = self.generate_decoder_mask(seq_len=T_all, device=agents_emb.device)
        else:
            time_masks = None

        # (T_all, B, M, d_k) -> (T_all, B*M, d_k)
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_all, B * M, -1)),
                                src_key_padding_mask=agent_masks, mask=time_masks)
        # (T_all, B*M, d_k) -> (T_all, B, M, d_k)
        return agents_temp_emb.view(T_all, B, M, -1)

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        '''
        :param agents_emb: (T_fut, B, M, d_k)
        :param context: (T_obs, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_fut, B, M, d_k)
        '''
        T_fut = agents_emb.shape[0]
        T_obs, B, M, _ = context.shape
        if self.mask_time:
            # [T_fut, T_fut], time mask for only later steps consider earlier steps.
            time_masks = self.generate_decoder_mask(seq_len=T_fut, device=agents_emb.device)
        else:
            time_masks = None
        # (B, T_obs, N) -> (BxM, T_obs)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_emb = self.pos_encoder(agents_emb.reshape(T_fut, -1, self.d_k))  # [T_fut, BxM, H]

        context = context.reshape(-1, B*(M), self.d_k) # [T_obs, BxM, H]

        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks)
        agents_temp_emb = agents_temp_emb.view(T_fut, B, M, -1)

        return agents_temp_emb

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer, aux_info=None):
        '''
        :param agents_emb: (T_all, B, M, d_k)
        :param agent_masks: (B, T_hist, M), T_all = 1 when mlp is used
        :return: (T_all, B, M, d_k)
        '''
        T_all, B, M, _ = agents_emb.shape
        T_hist = agent_masks.shape[1]
        T_fut = T_all - T_hist
        full_visibility_mask = None if aux_info is None else aux_info.get(
            "interaction_full_visibility_cut_mask"
        )
        agent_interaction_mask = expand_directed_attention_mask(
            full_visibility_mask, T_all, self.num_heads, agents_emb.device
        )
        if self.mask_social_interaction:
            self_only_mask = (1-torch.eye(M, device=agents_emb.device)).bool()
            if agent_interaction_mask is None:
                agent_interaction_mask = self_only_mask
            else:
                agent_interaction_mask = agent_interaction_mask | self_only_mask

        # ------------------- Agent Mask -------------------
        # (B, T_hist, M) -> (B, 1, M) -> (B, T_fut, M) 
        agent_fut_masks = agent_masks[:, -1:, :].repeat(1, T_fut, 1) # take last timestep of all agents.
        # (B, T_all, M) -> (B*T_all, M)
        agent_masks = torch.cat([agent_masks, agent_fut_masks], dim=-2).view(B*T_all, M)


        # (T_all, B, M, d_k) -> (M, BxT_all, d_k)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(M, B * T_all, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks, mask=agent_interaction_mask)
        # (M, BxT_all, d_k) -> (M, B, T_all, d_k) -> (T_all, B, M, d_k)
        agents_soc_emb = agents_soc_emb.view(M, B, T_all, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    
    def social_attn_with_edge_decoder_fn(self, agents_emb, agent_masks, layer, aux_info, attn_name="", T_total=83, motion_dist='ttc', use_graph_cond=False):
        '''
        :param agents_emb: (T_all, B, M, d_k)
        :param agent_masks: (B, T_hist, M), T_all = 1 when mlp is used
        :param aux_info: dict
        :return: (T_all, B, M, d_k)
        '''
        T_all, B, M, _ = agents_emb.shape
        neighbor_feat = apply_directed_interaction_edge_delay(
            aux_info['neighbor_feat'], aux_info.get("interaction_edge_delay_steps")
        )
        # T_hist = agent_masks.shape[1]
        # T_fut = T_all - T_hist

        # ------------------- Attention Mask -------------------
        # (M, M) -> (M, M, 1) -> (M, M, M) -> (M, M*M)
        attn_block_mask = torch.eye(M, device=agents_emb.device).unsqueeze(-1).repeat(1,1,M).reshape(M, M*M)
        attn_block_mask = (1 - attn_block_mask).to(torch.bool)
        # (M, M*M) -> (1, M, M*M) -> (B*T_all*num_heads, M, M*M)
        attn_block_mask = attn_block_mask.unsqueeze(0).repeat(B*T_all*self.num_heads, 1, 1)
        
        # (M, M) -> (M, 1, M) -> (M, M, M) -> (M, M*M)
        attn_self_unmask = torch.eye(M, device=agents_emb.device).unsqueeze(1).repeat(1,M,1).view(M,M*M)
        attn_self_unmask = (1 - attn_self_unmask).type(torch.bool)
        # (M, M*M) -> (B*T_all*self.num_heads, M, M*M)
        attn_self_unmask = attn_self_unmask.unsqueeze(0).repeat(B*T_all*self.num_heads, 1, 1)
        attn_self_unmask = attn_self_unmask | attn_block_mask
        
        if self.mask_social_interaction:
            attn_mask = attn_self_unmask
        else:
            # (B, M1, M2, T_all)
            neighbor_mask = (1.0 - neighbor_feat[...,-1]).type(torch.bool).to(agents_emb.device) # available or not
            neighbor_mask = apply_directed_interaction_edge_cut(
                neighbor_mask, aux_info.get("interaction_edge_cut_mask")
            )
            neighbor_mask = apply_directed_interaction_edge_cut(
                neighbor_mask,
                aux_info.get("interaction_full_visibility_cut_mask"),
            )
            # print(neighbor_mask.shape)
            # print(neighbor_mask[0][0])
            # print(aux_info['neighbor_feat'].shape)
            # (B, M1, M2, T_all) -> (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
            attn_mask = neighbor_mask.permute(0, 3, 1, 2).unsqueeze(-2).repeat(1, 1, 1, M, 1).view(B*T_all, M, M*M)
            # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_heads,1,1).view(B*T_all*self.num_heads, M, M*M)
            
            # (B, M1, M2, T_all, 2)
            if self.interaction_edge_speed_repr in ['rel_vel_per_step']:
                neighbor_pos = neighbor_feat[...,(-3,-2)]
                # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
                neighbor_dist = torch.norm(neighbor_pos, dim=-1).permute(0,3,1,2)
            elif self.interaction_edge_speed_repr in ['rel_vel_new']:
                neighbor_pos = neighbor_feat[...,(-3,-2)]
                # (B, M1, M2, T_all, 2) -> (B, T_all, 2, M1) -> (B, M1, T_all, 2) -> (B, M1, 1, T_all, 2) -> (B, M1, M2, T_all, 2)
                self_pos = torch.diagonal(neighbor_pos, dim1=1, dim2=2).permute(0,3,1,2).unsqueeze(2).repeat(1,1,M,1,1)
                # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
                neighbor_dist = torch.norm(neighbor_pos-self_pos, dim=-1).permute(0,3,1,2)
            else: # with both distance and ttc info, used in CCDiff's case
                # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
                if motion_dist == 'dist': 
                    neighbor_pos = neighbor_feat[...,(0,1)]
                    neighbor_dist = torch.norm(neighbor_pos, dim=-1).permute(0,3,1,2)
                elif motion_dist == 'ttc': 
                    neighbor_dist = neighbor_feat[..., -4].permute(0,3,1,2) # TTC values
                else: 
                    raise NotImplementedError("{} not implemented".format(motion_dist))
            
            if not use_graph_cond: 
                neighbor_dist_mask = neighbor_dist > self.social_attn_radius 
            else: 
                neighbor_dist_mask = torch.zeros_like(neighbor_dist, dtype=torch.bool)

            # (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
            neighbor_dist_mask = neighbor_dist_mask.unsqueeze(-2).repeat(1,1,1,M,1).view(B*T_all, M, M*M)
            # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
            neighbor_dist_mask = neighbor_dist_mask.unsqueeze(1).repeat(1,self.num_heads,1,1).view(B*T_all*self.num_heads, M, M*M)

            # (B*T_all*num_heads, M1, M1*M2)
            attn_mask = attn_mask | attn_block_mask | neighbor_dist_mask
            attn_mask = attn_mask & attn_self_unmask

            if self.attn_combination == 'gate':
                # mask all self-attentions as we use gate to combine self and neighbors
                attn_self_mask =  ~attn_self_unmask
                attn_mask = attn_mask | attn_self_mask
                if_nan = torch.sum(attn_mask, dim=-1) == M*M
                attn_mask[if_nan] = attn_mask[if_nan] & attn_self_unmask[if_nan]


        # ------------------- Query -------------------
        # Use the same delayed representation for both edge attributes and
        # attention masks.  Applying delay only to the radius/TTC mask is a
        # no-op whenever the edge remains inside the attention radius.
        num_hist_steps = aux_info['neighbor_hist_feat'].shape[-2]
        edge_feat = neighbor_feat[..., self.neighbor_inds]
        edge_hist_feat = edge_feat[..., :num_hist_steps, :]
        edge_fut_feat = edge_feat[..., num_hist_steps:, :]
        if self.mask_edge:
            edge_hist_feat = torch.zeros_like(edge_hist_feat, device=edge_hist_feat.device)
            edge_fut_feat = torch.zeros_like(edge_fut_feat, device=edge_fut_feat.device)

        # (B, M, M, T_hist, edge_attr) -> (B*M*M*T_hist, edge_attr)
        edge_hist_feat = edge_hist_feat.reshape(-1, len(self.neighbor_inds))
        # (B, M, M, T_fut, edge_attr) -> (B*M*M*T_fut, edge_attr)
        edge_fut_feat = edge_fut_feat.reshape(-1, len(self.neighbor_inds))

        if len(self.edge_attr_separation) > 0:
            edge_hist_feat = [edge_hist_feat[...,attrs] for attrs in self.edge_attr_separation]
            edge_fut_feat = [edge_fut_feat[...,attrs] for attrs in self.edge_attr_separation]

        # (B*M*M*T_hist, edge_attr) -> (B*M*M*T_hist, d_edge)
        edge_hist_feat = self.neighbor_hist_feat_encoder(edge_hist_feat)
        # (B*M*M*T_fut, edge_attr) -> (B*M*M*T_fut, d_edge)
        edge_fut_feat = self.neighbor_fut_feat_encoder(edge_fut_feat)

        # (B*M*M*T_hist, d_edge) -> (B, M, M, T_hist, d_edge)
        edge_hist_feat = edge_hist_feat.view(B, M, M, -1, self.d_edge)
        # (B*M*M*T_fut, d_edge) -> (B, M, M, T_fut, d_edge)
        edge_fut_feat = edge_fut_feat.view(B, M, M, -1, self.d_edge)

        # (B, M, M, T_all, d_edge)
        edge_feat = torch.cat([edge_hist_feat, edge_fut_feat], dim=-2)
        # (B, M, M, T_all, d_edge) -> (M, M, B, T_all, d_edge) -> (M*M, BxT_all, d_edge)
        edge = edge_feat.permute(1, 2, 0, 3, 4).reshape(M*M, B * T_all, self.d_edge)

        # (T_all, B, M, K_neigh) -> (M, B, T_all, K_neigh) -> (M, B*T_all, K_neigh)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(M, B * T_all, -1)
        # (M, B*T_all, d_k) -> (1, M, B*T_all, d_k) -> (M, M, B*T_all, d_k) -> (M*M, B*T_all, d_k)
        agents_emb_expand = agents_emb.unsqueeze(0).repeat(M, 1, 1, 1).view(M*M, B*T_all, self.d_k)
        # (M*M, B*T_all, d_k+d_edge)
        agents_emb_expand_and_edge = torch.cat([agents_emb_expand, edge], dim=-1)
        
        # agents_soc_emb = layer(agents_emb, agents_emb_expand_and_edge, memory_mask=attn_mask, memory_key_padding_mask=agent_masks, attn_name=attn_name)
        agents_soc_emb, attn_weights = layer(agents_emb, agents_emb_expand_and_edge, memory_mask=attn_mask, attn_name=attn_name, T_total=T_total)
        # (M, BxT_all, d_k) -> (M, B, T_all, d_k) -> (T_all, B, M, d_k)
        agents_soc_emb = agents_soc_emb.view(M, B, T_all, -1).permute(2, 1, 0, 3)

        return agents_soc_emb, attn_weights

    def forward(self, x_noisy, aux_info, t_inp, use_cond=True, motion_dist='ttc', ):
        '''
        : param x_noisy: (B, M, T_fut, k_output)
        : param aux_info: 
            agent_hist_feat mlp:(B, M, 1, d_k), transformer:(B, M, T_hist, d_k)
            map_feat cnn:(B, M, 1, d_k), transformer:(B, M, S_seg, d_k)

        : param t_inp: (B)
        : param use_cond: bool, if the causal condition is used.
                          When False, dummy causal graph is used

        :return:
            out: [B, M, (T_hist+)T_fut, output_dim]
        '''
        # print('FORWARD: ', x_noisy.isnan().any())
        B, M, T_fut, _ = x_noisy.shape
        attn_weights = None
        # if we put everything (subset of {map, agent_hist, neighbor_hist}) into cond_feat
        if self.single_cond_feat:
            # agent_hist_feat = aux_info['cond_feat'].unsqueeze(-2)
            agent_hist_feat = aux_info['agent_hist_feat']
            map_feat = aux_info['map_feat']
        else:
            agent_hist_feat = aux_info['agent_hist_feat']
            map_feat = aux_info['map_feat']

        # Process agent history information
        if self.agent_hist_embed_method == 'transformer':
            # extract agent history vectors
            # Encode all input observations
            # (B, M, T, k_attr+1) -> (B, T, M, k_attr+1)
            agent_hist_feat = agent_hist_feat.permute(0, 2, 1, 3)
            # (B, M, T, k_attr+1) -> (B, T, M, k_attr), (B, T, M)
            agents_tensor, opps_masks = agent_hist_feat[..., :-1], (1.0 - agent_hist_feat[..., -1]).type(torch.BoolTensor).to(agent_hist_feat.device) 

            # (B, T, M, k_attr) -> (T, B, M, d_k)
            agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

            # Process through AutoBot's encoder
            for i in range(self.L_enc):
                agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
                agents_emb = self.social_attn_fn(
                    agents_emb, opps_masks, layer=self.social_attn_layers[i],
                    aux_info=aux_info,
                )
            # (T, B, M, d_k)
        elif self.agent_hist_embed_method == 'mlp':
            # (B, M, 1, d_k) -> (1, B, M, d_k)
            agents_emb = agent_hist_feat.permute(2, 0, 1, 3)
            # dummy mask (B, 1, M)
            opps_masks = torch.zeros((B, 1, M)).type(torch.BoolTensor).to(agents_emb.device)
        elif self.agent_hist_embed_method == 'concat':
            # extract agent history vectors
            # Encode all input observations
            # (B, M, T_hist, k_attr+1) -> (B, T_hist, M, k_attr+1)
            agent_hist_feat = agent_hist_feat.permute(0, 2, 1, 3)
            # (B, T_hist, M, k_attr+1) -> (B, T_hist, M, k_attr), (B, T_hist, M)
            agents_tensor, opps_masks = agent_hist_feat[..., :-1], (1.0 - agent_hist_feat[..., -1]).type(torch.BoolTensor).to(agent_hist_feat.device)

            # (B, T_hist, M, k_attr) -> (T_hist, B, M, d_k)
            agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)
            # TBD: think about how to deal with availability field properly
            # print('agents emb: ', agents_emb.shape, 'add noise')
            # agents_emb += 5.0 * torch.randn_like(agents_emb)
        else:
            raise NotImplementedError
        # print("AGENT EMB: ", agents_emb.isnan().any())
        # Process map information
        if self.use_map_lanes:
            if self.map_embed_method == 'transformer':
                # (S_seg, B, M, d_k) -> (S_seg, B, M, d_k+1) add availability dimension
                nan_inds = torch.sum(torch.isnan(map_feat).float(), dim=-1) > 0
                avail = torch.ones_like(map_feat[...,0])
                avail[nan_inds] = 0
                # TBD: currently convert nan values to 0.0, maybe try other values?
                map_feat = torch.nan_to_num(map_feat, nan=0.0)
                map_feat = torch.cat([map_feat, avail.unsqueeze(-1)], dim=-1)

                # extract map vectors
                # (B, M, S_seg, P_per_seg, map_attr) -> (S_seg, B, M, d_k)
                orig_map_features, orig_road_segs_masks = self.map_encoder(map_feat)
                # (S_seg, B, M, d_k) -> (S_seg, BxM, d_k)
                map_features = orig_map_features.view(-1, B * M, self.d_k)
                # (B, M, S_seg) -> (BxM, S_seg)
                road_segs_masks = orig_road_segs_masks.view(B * M, -1)
            elif self.map_embed_method == 'cnn':
                # (B, M, 1, d_k) -> (1, BxM, d_k)
                map_features = map_feat.permute(2, 0, 1, 3).reshape(1, B*M, -1)
                # dummy mask (B*M, 1)
                road_segs_masks = torch.zeros((B*M, 1)).type(torch.BoolTensor).to(map_features.device)
            else:
                raise NotImplementedError

                

        # AutoBot-Joint Decoding
        # (B, M, T_fut, output_dim) -> (B, T_fut, M, output_dim)
        dec_parameters = x_noisy.permute(0, 2, 1, 3)
        # (B, T_fut, M, output_dim) -> (T_fut, B, M, d_k)
        agents_dec_emb = self.agents_noisy_input_encoder(dec_parameters).permute(1, 0, 2, 3)
        agents_dec_emb = apply_semantic_assertive_token(
            agents_dec_emb,
            aux_info.get("semantic_assertive_receiver_mask"),
            1.0,
        )
        if self.agent_hist_embed_method == 'concat':
            # (T_fut, B, M, d_k) -> (T_hist+T_fut, B, M, d_k)
            agents_dec_emb = torch.cat([agents_emb, agents_dec_emb], dim=0)
            T_hist = agents_emb.shape[0]
            T_total = T_hist + T_fut
        else:
            T_total = T_fut
        # Add time embedding for the denoising step k
        # (B) -> (B, d_k) -> (1, B, 1, d_k)
        time_embed = self.time_mlp(t_inp).unsqueeze(0).unsqueeze(-2)
        agents_dec_emb = agents_dec_emb + time_embed
        # The raster CNN and process_cond_mlp already produce this condition,
        # but the previous implementation never consumed it. Add the fused
        # map/history context to every denoising token so gradients and map
        # information can reach the generated trajectory.
        cond_feat = aux_info.get('cond_feat')
        if self.single_cond_feat:
            if cond_feat is None:
                raise RuntimeError('single_cond_feat=True requires aux_info[cond_feat]')
            if cond_feat.dim() == 3:
                cond_token = cond_feat.unsqueeze(-2)
            elif cond_feat.dim() == 4 and cond_feat.shape[-2] == 1:
                cond_token = cond_feat
            else:
                raise RuntimeError('Unexpected fused condition shape {}'.format(tuple(cond_feat.shape)))
            if cond_token.shape[:2] != (B, M) or cond_token.shape[-1] != self.d_k:
                raise RuntimeError(
                    'Unexpected condition shape {}; expected [B={}, M={}, 1, d_k={}]'.format(
                        tuple(cond_token.shape), B, M, self.d_k
                    )
                )
            agents_dec_emb = agents_dec_emb + cond_token.permute(2, 0, 1, 3)
        # print('AGENT DECODE #1: ', agents_dec_emb.isnan().any())

        for d in range(self.L_dec):
            # Cross-Attention with Map
            if self.use_map_lanes and d == 1:
                # (T_total, B, M, d_k) -> (T_total, B*M, d_k)
                agents_dec_emb = agents_dec_emb.reshape(T_total, -1, self.d_k)
                agents_dec_emb_map = self.map_attn_layers(query=agents_dec_emb, key=map_features, value=map_features, key_padding_mask=road_segs_masks)[0]
                agents_dec_emb = agents_dec_emb + agents_dec_emb_map
                # (T_total, B*M, d_k) -> (T_total, B, M, d_k)
                agents_dec_emb = agents_dec_emb.reshape(T_total, B, M, -1)
            # agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, map_features, opps_masks, layer=self.self.temporal_attn_decoder_map_layers[d])
            if self.agent_hist_embed_method in ['mlp', 'transformer']:
                agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, agents_emb, opps_masks, layer=self.temporal_attn_decoder_layers[d])
            # add this extra self temporal layer to capture the temporal dynamics of the agents
            agents_dec_emb = self.self_temporal_attn_decoder_fn(agents_dec_emb, opps_masks, layer=self.self_temporal_attn_decoder_layers[d])

            if self.neigh_hist_embed_method == 'interaction_edge':
                if d == 0 or self.all_interactive_social:
                    if 'attn_name' not in aux_info or aux_info['attn_name']=='':
                        attn_name_str = ""
                    else:
                        attn_name_str = aux_info['attn_name']+"_"+str(d)
                    agents_dec_emb, attn_weights = self.social_attn_with_edge_decoder_fn(agents_dec_emb, opps_masks, layer=self.social_attn_decoder_layers[d], 
                                                                                         aux_info=aux_info, attn_name=attn_name_str, T_total=T_total,
                                                                                         motion_dist=motion_dist, use_graph_cond=use_cond)
                else:
                    agents_dec_emb = self.social_attn_decoder_fn(
                        agents_dec_emb, opps_masks,
                        layer=self.social_attn_decoder_layers[d], aux_info=aux_info,
                    )
            else:
                agents_dec_emb = self.social_attn_decoder_fn(
                    agents_dec_emb, opps_masks,
                    layer=self.social_attn_decoder_layers[d], aux_info=aux_info,
                )
        # print('AGENT DECODE #2: ', agents_dec_emb.isnan().any())
        # (T_total, B, M, d_k) -> (T_total*B*M, output_dim)
        out = self.output_model(agents_dec_emb.reshape(-1, self.d_k))
        # (T_total*B*M, output_dim) -> (T_total, B, M, output_dim) -> (B, M, T_total, output_dim)
        out = out.reshape(T_total, B, M, -1).permute(1, 2, 0, 3)
        pred_info = {'attn_weights': attn_weights}
        # print('OUT: ', out.isnan().any())

        return out, pred_info

class SocialTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, attn_name="", T_total=83):
        output = tgt

        for i, mod in enumerate(self.layers):
            if attn_name != "":
                attn_name_str = attn_name+"_"+str(i)
            else:
                attn_name_str = ""
            output, attn_weights_i = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask, attn_name=attn_name_str, T_total=T_total)
            # pick the first layer's attn_weights
            if i == 0:
                attn_weights = attn_weights_i

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights

class SocialDecoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 edge_dim: int,
                 num_heads: int = 16,
                 dropout: float = 0.1,
                 attn_combination: str = 'sum') -> None:
        super(SocialDecoderLayer, self).__init__()
        self.self_attn_with_edge = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, kdim=embed_dim+edge_dim, vdim=embed_dim+edge_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attn_combination = attn_combination
        if self.attn_combination == 'gate':
            self.lin_self = nn.Linear(embed_dim, embed_dim)
            self.lin_ih = nn.Linear(embed_dim, embed_dim)
            self.lin_hh = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                attn_name: str = "",
                T_total: int = 83) -> torch.Tensor:
        '''
        Input:
            src: (M, B*T_all, d_k)
            memory: (M, M, B*T_all, d_edge)
            memory_mask: (B*T_all*num_heads, M, M*M)
            memory_key_padding_mask: (B*T_all, M*M)
            attn_name: str, the name of attention weights to be visualized. If "", no visualization.
        Output:
            x: (M, B*T_all, d_k)
            attn_weights: (M, M)
        '''
        x, x_expand_and_edge = tgt, memory

        # always return weights but not necessarily visualize them
        attn, attn_weights = self._sa_block(self.norm1(x), x_expand_and_edge, key_padding_mask=memory_key_padding_mask, attn_mask=memory_mask, need_weights=True)
        if attn_name != '':
            visualize_attn(attn_weights, attn_name, T_total)

        BT, M, MM = attn_weights.size()
        B = BT // T_total
        attn_weights = attn_weights.reshape(B, T_total, M, MM)
        row_indices = torch.arange(M).view(-1, 1).expand(-1, M)
        col_indices = (torch.arange(M).view(-1, 1) * M + torch.arange(M).view(1, -1)).expand(M, -1)
        # (B, S, M, MM) -> (B, S, M, M)
        attn_weights_selected = attn_weights[:, :, row_indices, col_indices]

        if self.attn_combination == 'gate':
            gate = torch.sigmoid(self.lin_ih(attn) + self.lin_hh(x))
            x = attn + gate * (self.lin_self(x) - attn)
        else:
            x = x + attn
        x = x + self._ff_block(self.norm2(x))

        return x, attn_weights_selected

    def _sa_block(self,
                  x: torch.Tensor,
                  x_expand_and_edge: torch.Tensor,
                  key_padding_mask: Optional[torch.Tensor],
                  attn_mask: Optional[torch.Tensor],
                  need_weights: bool=False) -> torch.Tensor:

        x, attn_weights = self.self_attn_with_edge(x, x_expand_and_edge, x_expand_and_edge, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=need_weights)

        return self.dropout1(x), attn_weights

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)

def visualize_attn(attn_weights, attn_name='', T_total=83):
    from matplotlib import pyplot as plt
    T_hist = 31
    ts = [31, 51]
    S = len(ts)
    BT, M, MM = attn_weights.size()
    B = BT // T
    # (BT, M, MM) -> (B, T, M, MM) -> (T, M, MM) -> (S, M, MM)
    attn_weights = attn_weights.reshape(B, T_total, M, MM).mean(0)[ts, ...]
    # (S, M, MM)
    # attn_weights = attn_weights.view(S, M, M, M)

    row_indices = torch.arange(M).view(-1, 1).expand(-1, M)
    col_indices = (torch.arange(M).view(-1, 1) * M + torch.arange(M).view(1, -1)).expand(M, -1)
    # (S, M, MM) -> (S, M, M)
    attn_weights = attn_weights[:, row_indices, col_indices]

    fig_size = (int(MM//3), int(M*S//3))
    fig_size_2 = (M, M*S)
    
    # Plot the attention weights for each timstep
    _, axs = plt.subplots(nrows=S, ncols=1, figsize=fig_size_2)
    for i, ax in enumerate(axs):
        attn_weights_i = attn_weights[i].cpu().detach().numpy()
        ax.imshow(attn_weights_i, cmap='hot', interpolation='nearest', origin='lower')
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels([i for i in range(M)], fontsize=40)
        ax.set_yticklabels([i for i in range(M)], fontsize=40)
        ax.set_title(f'Planning Timestep {ts[i]-T_hist}', fontsize=50)

        # add annotations
        for k in range(attn_weights_i.shape[0]):
            for j in range(attn_weights_i.shape[1]):
                ax.annotate(str(round(attn_weights_i[k, j], 2)), xy=(j, k), 
                            horizontalalignment='center',
                            verticalalignment='center')
    plt.tight_layout()
    dir = 'nusc_results/attn_weights/'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(dir+'attn_weights_'+attn_name+'.png')
    plt.close()

class MultipleInputEmbedding(nn.Module):
    def __init__(self,
                    in_channels: List[int],
                    out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                            nn.LayerNorm(out_channel),
                            nn.ReLU(inplace=True),
                            nn.Linear(out_channel, out_channel))
                for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
