import torch
from torch import nn
import config
import math
from typing import NamedTuple
import numpy as np

def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])

def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)

    pis = torch.cat(
        [torch.nn.functional.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    ) 
    costs = torch.cat(costs, 1)

    mincosts, argmincosts = costs.min(-1)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

def compute_in_batches(f, calc_batch_size, *args, n=None):
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            return None
        return torch.cat(chunks, dim)

    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)

class ExpBaseline():
    def __init__(self, beta):
        self.beta = beta
        self.v = None

    def eval(self, x, c):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()
        self.v = v.detach()  
        return self.v, 0

    def state_dict(self):
        return {
            'v': self.v
        }
        
    def load_state_dict(self, state_dict):
        self.v = state_dict['v']
        
    def wrap_dataset(self, dataset):
        return dataset
    
    def unwrap_batch(self, batch):
        return batch, None
    
    def get_learnable_parameters(self):
        return []

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)


        Q = torch.matmul(qflat, self.W_query).view(shp_q)

        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))


        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)


        return out

class AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key], 
            glimpse_val=self.glimpse_val[:, key], 
            logit_key=self.logit_key[key]
        )

class BatchNormalization(nn.Module):
    def __init__(self, embed_dim):
        super(BatchNormalization, self).__init__()
        self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())

class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            BatchNormalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                )
            ),
            BatchNormalization(embed_dim)
        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()
        
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    def forward(self, x):
        h = self.layers(x)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
        
        
class AttentionModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 mask_inner=True,
                 mask_logits=True,
                 shrink_size=None):
        
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.tanh_clipping = config.TANH_CLIPPING
        self.temp = 1.0
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.problem = problem
        self.n_heads = config.N_HEADS
        self.shrink_size = shrink_size
        
        step_context_dim = embedding_dim + 1
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        node_dim = 3
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=config.N_HEADS,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, input): 
        embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)

        ll = self._calc_log_likelihood(_log_p, pi, mask)

        return cost, ll, pi

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)
    
    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            log_p[mask] = 0
        return log_p.sum(1)

    def _init_embed(self, input):
        features = ('demand',)
        return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['coordinates'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)
        batch_size = state.ids.size(0)
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    
            log_p, mask = self._get_log_p(fixed, state)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  

            state = state.update(selected)

            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], pi),  
            (input, self.embedder(self._init_embed(input))[0]),  
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        selected = probs.multinomial(1).squeeze(1)
        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            selected = probs.multinomial(1).squeeze(1)
        return selected

    def _precompute(self, embeddings, num_steps=1):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        mask = state.get_mask()

        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()
        return torch.cat(
                (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.capacity - state.used_capacity[:, :, None]
                ),
                -1
            )
    
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
