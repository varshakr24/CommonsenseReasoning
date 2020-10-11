from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn import Identity


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class KVMem_Att_layer(nn.Module):
	def __init__(self, config):
		super(KVMem_Att_layer, self).__init__()
		self.commonsense_encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=100, batch_first=True, bidirectional=True, dropout=config.hidden_dropout_prob)
		self.commonsense_wk = nn.Linear(200, config.hidden_size)
		self.commonsense_wv = nn.Linear(200, config.hidden_size)
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

		self.output = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, bert_output, commonsense, attention_mask, commonsense_mask, commonsense_shape):
		b_size, num_cand, num_path, path_len = commonsense_shape
		lstm_output, (h, c) = self.commonsense_encoder(commonsense)
		encoded_cs = h.permute(1, 0, 2)
		encoded_cs = encoded_cs.contiguous().view(b_size*num_cand*num_path, -1)
		cs_key = self.commonsense_wk(encoded_cs).view(b_size*num_cand, num_path, -1)
		cs_value = self.commonsense_wv(encoded_cs).view(b_size*num_cand, num_path, -1)

		query_layer = self.transpose_for_scores(bert_output)
		key_layer = self.transpose_for_scores(cs_key)
		value_layer = self.transpose_for_scores(cs_value)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)

		mask_cs = (1.0 - commonsense_mask.float()) * -10000.0
		mask_dqa = (1.0 - attention_mask.float()) * -10000.0
		joint_mask = mask_cs.unsqueeze(1) + mask_dqa.unsqueeze(2)    # batch * seq_len * num_paths
		joint_mask = joint_mask.unsqueeze(1)   # batch * 1 * seq_len * num_paths
		attention_scores = attention_scores + joint_mask
		attention_probs = nn.Softmax(dim=-1)(attention_scores)
		attention_probs = self.att_dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		cs_attended = self.output(context_layer)
		cs_attended = self.dropout(cs_attended)
		cs_attended = self.LayerNorm(cs_attended + bert_output)
		return cs_attended