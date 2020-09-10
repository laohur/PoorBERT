"""PyTorch ALBERT model. """
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from configs import Constants
from .activition import wing, WingNorm, LinearNorm, SqrtNorm, relu0, penalized_tanh, ptanh
from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_albert import AlbertConfig
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)
import time
t0=time.time()
def time_cost(tag):
    t1=time.time()
    cost=t1-t0
    logger.info(f" {tag}  cost {cost} ")
    return t1

ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'albert-base': "",
    'albert-large': "",
    'albert-xlarge': "",
    'albert-xxlarge': "",
}
def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    if not os.path.exists(tf_path+'/checkpoint'):
        tf_path = tf_path + "/variables/variables"
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.replace("attention_1","attention")
        name = name.replace("ffn_1","ffn")
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            elif re.fullmatch(r'[A-Za-z]+_+[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] in ['LayerNorm', 'attention', 'ffn'] and len(l) >= 2:
                l = ["_".join(l[:-1])]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]

        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu,"relu0":relu0, "swish": swish, "gelu_new": gelu_new,"wing":wing,"ptanh":ptanh}
NORM2FN = {"layer_norm":torch.nn.LayerNorm,"linear_norm":LinearNorm,"wing_norm":WingNorm,"sqrt_norm":SqrtNorm}

class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, type_ids=None, position_ids=None):
        batchsize,seq_length = input_ids.size()[:2]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_ids = position_ids.unsqueeze(0).repeat(batchsize,1)
        if type_ids is None:
            type_ids = torch.zeros_like(input_ids)
        if len(input_ids.size())>=3:
            words_embeddings=input_ids
        else:
            words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        type_embeddings = self.type_embeddings(type_ids)
        embeddings = words_embeddings + position_embeddings + type_embeddings
        # embeddings = wing(embeddings)
        # embeddings = self.LayerNorm(words_embeddings)+self.LayerNorm(position_embeddings)+self.LayerNorm(task_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AlbertSelfAttention(nn.Module):
    def __init__(self, config):
        super(AlbertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions   # default False
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.act_fn=ACT2FN[config.hidden_act]
        self.LayerNorm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):  # batch*128*768 -> batch*16*128*4
        # x=self.act_fn(x)
        # x=self.LayerNorm(x)
        # new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  #16*12*128*128
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.LayerNorm(attention_scores)/4+0.5

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # batch*12*128*64

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # batch*128*12*64
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  #batch*128*758
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs  # hidden

class AlbertSelfOutput(nn.Module):
    def __init__(self, config):
        super(AlbertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.norm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states  # 1 tensor

class AlbertAttention(nn.Module):
    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.self = AlbertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        # attention_output =  self_outputs[0]
        outputs = (attention_output,self_outputs)
        return outputs  # attn hidden

class AlbertOutput(nn.Module):
    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states  # 1 tensor

class AlbertIntermediate(nn.Module):
    def __init__(self, config):
        super(AlbertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = AlbertOutput(config)
        self.LayerNorm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        # intermediate_output = self.LayerNorm(intermediate_output)
        output = self.output(intermediate_output)
        return output  # hidden

class AlbertFFN(nn.Module):
    def __init__(self, config):
        super(AlbertFFN, self).__init__()
        self.intermediate = AlbertIntermediate(config)

    def forward(self, attention_output):
        output = self.intermediate(attention_output)
        return output  # 1 tensor

class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = AlbertAttention(config)
        self.ffn = AlbertFFN(config)
        self.LayerNorm = NORM2FN[config.norm](config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm_1 = NORM2FN[config.norm](config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self.LayerNorm(attention_outputs[0] )+hidden_states
        ffn_output = self.ffn(attention_output)
        ffn_output = self.LayerNorm_1(ffn_output)+attention_output
        outputs = (ffn_output,) + attention_outputs[1:] # add attentions if we output them
        return outputs  # hidden atten

class AlbertGroup(nn.Module):
    def __init__(self, config):
        super(AlbertGroup, self).__init__()
        self.inner_group_num = config.inner_group_num
        self.inner_group = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask, head_mask):
        layer_attentions = ()
        layer_hidden_states = ()
        for inner_group_idx in range(self.inner_group_num):
            layer_module = self.inner_group[inner_group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask)
            hidden_states = layer_outputs[0]
            layer_attentions = layer_attentions + (layer_outputs[1],)
            layer_hidden_states = layer_hidden_states + (hidden_states,)
        return (layer_hidden_states, layer_attentions)  # hidden atten

class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super(AlbertTransformer, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.group = nn.ModuleList([AlbertGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask, head_mask):
        all_hidden_states = ()
        all_attentions = ()
        for layer_idx in range(self.num_hidden_layers):
            # if self.output_hidden_states and layer_idx == 0:
            #     all_hidden_states = all_hidden_states + (hidden_states,)
            group_idx = int(layer_idx / self.num_hidden_layers * self.num_hidden_groups)
            layer_module = self.group[group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[layer_idx])
            hidden_states = layer_outputs[0][-1]
            if self.output_attentions:
                all_attentions = all_attentions + layer_outputs[1]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + layer_outputs[0]
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)   # hidden

class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.embedding_hidden_mapping_in = nn.Linear(self.embedding_size, self.hidden_size)
        self.transformer = AlbertTransformer(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        if self.embedding_size != self.hidden_size:
            prev_output = self.embedding_hidden_mapping_in(hidden_states)
        else:
            prev_output = hidden_states
        outputs = self.transformer(prev_output, attention_mask, head_mask)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)   # hidden

class AlbertPooler(nn.Module):
    def __init__(self, config):
        super(AlbertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # self.activation = ACT2FN[config.hidden_act]
        # self.LayerNorm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        # pooled_output = self.LayerNorm(pooled_output)
        return pooled_output   # first hidden

class AlbertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(AlbertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states  # seq

class AlbertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(AlbertLMPredictionHead, self).__init__()
        self.transform = AlbertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.embedding_size,config.vocab_size,bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # seq vocab

class AlbertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(AlbertOnlyMLMHead, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class AlbertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(AlbertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class AlbertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(AlbertPreTrainingHeads, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score   # seq cls

class AlbertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # elif isinstance(module, AlbertLayerNorm):
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


ALBERT_START_DOCSTRING = r"""    The ALBERT model was proposed in
    `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations`_
    by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. 
    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.
    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1909.11942
    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module
    Parameters:
        config (:class:`~transformers.ALbertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, ALBERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:
            (a) For sequence pairs:
                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``
            (b) For single sequences:
                ``tokens:         [CLS] the dog is hairy . [SEP]``
                ``token_type_ids:   0   0   0   0  0     0   0``
            ALBert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare Albert Model transformer outputting raw hidden-states without any specific head on top.",
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertModel(AlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """

    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = AlbertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, type_ids=None, position_ids=None, head_mask=None):
        batch,seq=input_ids.size()[:2]
        if attention_mask is None:
            # attention_mask = torch.ones_like(input_ids)
            attention_mask = torch.ones((batch,seq),dtype=input_ids.dtype,device=input_ids.device)
        if type_ids is None:
            # task_type_ids = torch.zeros_like(input_ids)
            type_ids = torch.zeros((batch, seq), dtype=input_ids.dtype, device=input_ids.device)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, type_ids=type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)  # seq [cls]

@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training:
    a `masked language modeling` head and a `next sentence prediction (classification)` head. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForPreTraining(AlbertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForPreTraining, self).__init__(config)
        self.bert = AlbertModel(config)

        self.predictions = AlbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, config.sentence_relation_size)  # relation
        self.activation = nn.Tanh()  # poller nn.Tanh()
        self.norm = NORM2FN[config.norm](config.embedding_size, eps=config.layer_norm_eps)

        # self.cls = AlbertPreTrainingHeads(config)
        # self.tasks=config.tasks
        self.char_labeler = nn.Linear(config.hidden_size, self.config.char_lable_size)   # char
        self.word_labeler = nn.Linear(config.hidden_size, self.config.word_lable_size)   # word
        # self.modify_labeler = nn.Linear(config.hidden_size, 2)  # mofidy
        self.qa_labeler = nn.Linear(config.hidden_size, 2)  # qa

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, next_sentence_label=None, tasks=[]):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            type_ids=type_ids)
        # return outputs
        sequence_output, pooled_output, hidden_states = outputs[:3]
        # pooled_output = self.bert.pooler(pooled_output)
        seq2=hidden_states[len(hidden_states)//2]
        # seq2=hidden_states[-1]
        pooled2 = self.bert.pooler(seq2)  # layer-2  pooled output
        logits={}
        if Constants.SCORE_MASK in tasks:
            prediction_scores = [self.predictions(sequence_output),self.predictions(seq2)]
            logits[Constants.SCORE_MASK]=prediction_scores
        if Constants.SCORE_CHAR in tasks:
            logits[Constants.SCORE_CHAR]=self.char_labeler(hidden_states[0])
        if Constants.SCORE_WORD in tasks:
            logits[Constants.SCORE_WORD]=self.word_labeler(hidden_states[1])
        if Constants.SCORE_MODIFY in tasks:
            logits[Constants.SCORE_MODIFY]=[self.modify_labeler(pooled_output),self.modify_labeler(pooled2)]
        if Constants.SCORE_RELATION in tasks:
            logits[Constants.SCORE_RELATION] = [self.seq_relationship( pooled_output),self.seq_relationship( pooled2)]
        if Constants.SCORE_QA in tasks:
            logits[Constants.SCORE_QA] = [self.qa_labeler(pooled_output),self.qa_labeler(pooled2)]
        # outputs = (prediction_scores, seq_relationship_score,char_score,word_score) + outputs[2:]  # add hidden states and attention if they are here
        outputs =[outputs,logits]

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)  # seq cls

@add_start_docstrings("""Bert Model with a `language modeling` head on top. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForMaskedLM(AlbertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForMaskedLM, self).__init__(config)

        self.bert = AlbertModel(config)
        self.cls = AlbertOnlyMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForNextSentencePrediction(AlbertPreTrainedModel):
    r"""
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next sequence prediction (classification) loss.
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        seq_relationship_scores = outputs[0]
    """

    def __init__(self, config):
        super(AlbertForNextSentencePrediction, self).__init__(config)

        self.bert = AlbertModel(config)
        self.cls = AlbertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                next_sentence_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            type_ids=type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output+0.1)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForMultipleChoice, self).__init__(config)

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)

class AlbertForTokenClassification(AlbertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            type_ids=type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                active_labels = labels.reshape(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]
    """

    def __init__(self, config):
        super(AlbertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
