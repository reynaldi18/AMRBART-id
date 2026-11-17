# coding: utf-8
"""M3/M4: MBart AMR parser + head biaffine dependency parsing di atas output encoder.

Parameter sharing implisit: encoder dipakai bersama; head hanya dipanggil bila batch membawa
`dep_heads`/`dep_rels` (training M3/M4). generate() tidak pernah memanggilnya, dan memuat
checkpoint ini dengan MBartForConditionalGeneration biasa cukup mengabaikan kunci `dep_head.*`
-> jalur decoding identik dengan M1/M2.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_outputs import Seq2SeqLMOutput
from model_interface.modeling_bart import MBartForConditionalGeneration

IGNORE = -100
NEG = -1e9


@dataclass
class Seq2SeqLMOutputWithDep(Seq2SeqLMOutput):
    dep_loss: Optional[torch.FloatTensor] = None
    # [arc_loss, rel_loss, benar_arc, benar_arc_dan_label, jumlah_dependen]
    dep_stats: Optional[torch.FloatTensor] = None


class _ScaleGrad(torch.autograd.Function):
    """Identitas di forward; gradien dikali `scale` di backward (warm-up gradien ke encoder)."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.linear(x)))


class BiaffineDepHead(nn.Module):
    """Dozat & Manning (2017): skor busur biaffine + klasifikasi label biaffine (diberi kepala emas)."""

    def __init__(self, d_model, n_rels, arc_dim=500, rel_dim=100, dropout=0.33):
        super().__init__()
        self.n_rels = n_rels
        self.arc_dep = _MLP(d_model, arc_dim, dropout)
        self.arc_head = _MLP(d_model, arc_dim, dropout)
        self.rel_dep = _MLP(d_model, rel_dim, dropout)
        self.rel_head = _MLP(d_model, rel_dim, dropout)
        # nol di awal (seperti Dozat & Manning): head tidak mengirim gradien ke encoder di step 0
        self.arc_U = nn.Parameter(torch.zeros(arc_dim + 1, arc_dim))
        self.rel_U = nn.Parameter(torch.zeros(n_rels, rel_dim + 1, rel_dim + 1))

    @staticmethod
    def _add_one(x):
        return torch.cat([x, x.new_ones(x.shape[:-1] + (1,))], dim=-1)

    def forward(self, hidden, dep_heads, dep_rels):
        dep_mask = dep_heads >= 0
        cand = dep_heads != IGNORE
        n = dep_mask.sum()
        zero = hidden.sum() * 0.0
        if n == 0:
            return zero, torch.zeros(5, device=hidden.device)

        with torch.cuda.amp.autocast(enabled=False):
            h = hidden.float()
            b_idx, i_idx = dep_mask.nonzero(as_tuple=True)
            gold = dep_heads[b_idx, i_idx]

            # busur: S[b,i,j] = [d_i;1]^T U h_j
            ad = self._add_one(self.arc_dep(h))                       # B,T,a+1
            ah = self.arc_head(h)                                     # B,T,a
            S = torch.bmm(ad @ self.arc_U, ah.transpose(1, 2))         # B,T,T
            s = S[b_idx, i_idx]                                       # N,T
            s = s.masked_fill(~cand[b_idx], NEG)
            s[torch.arange(s.shape[0], device=s.device), i_idx] = NEG  # bukan kepala bagi diri sendiri
            arc_loss = F.cross_entropy(s, gold)

            # label: s_r = [d_i;1]^T U_r [h_gold;1]
            rd = self._add_one(self.rel_dep(h)[b_idx, i_idx])          # N,r+1
            rh = self._add_one(self.rel_head(h)[b_idx, gold])          # N,r+1
            sr = torch.einsum("ni,kij,nj->nk", rd, self.rel_U, rh)     # N,R
            rgold = dep_rels[b_idx, i_idx]
            rel_loss = F.cross_entropy(sr, rgold)

            with torch.no_grad():
                arc_ok = s.argmax(-1) == gold
                las_ok = arc_ok & (sr.argmax(-1) == rgold)
                stats = torch.stack([arc_loss.detach(), rel_loss.detach(), arc_ok.sum().float(),
                                     las_ok.sum().float(), n.float()])
        return arc_loss + rel_loss, stats


class MBartForAMRParsingMTL(MBartForConditionalGeneration):
    """MBartForConditionalGeneration + head dependency biaffine. `config.mtl_n_rels` wajib ada."""

    def __init__(self, config):
        super().__init__(config)
        self.dep_head = BiaffineDepHead(
            config.d_model,
            config.mtl_n_rels,
            arc_dim=getattr(config, "mtl_arc_dim", 500),
            rel_dim=getattr(config, "mtl_rel_dim", 100),
            dropout=getattr(config, "mtl_dropout", 0.33),
        )
        # faktor gradien L_dep yang masuk ke encoder; diatur Trainer (warm-up lambda)
        self.mtl_encoder_grad_scale = 1.0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        dep_heads=None,
        dep_rels=None,
    ):
        want_dep = dep_heads is not None
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if want_dep else return_dict,
        )
        if not want_dep:
            return out
        enc = out.encoder_last_hidden_state
        scale = float(self.mtl_encoder_grad_scale)
        if self.training and scale != 1.0:
            enc = _ScaleGrad.apply(enc, scale)
        dep_loss, dep_stats = self.dep_head(enc, dep_heads, dep_rels)
        return Seq2SeqLMOutputWithDep(**out, dep_loss=dep_loss, dep_stats=dep_stats)
