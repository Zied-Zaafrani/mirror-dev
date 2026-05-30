"""
Autoregressive drug sequence decoder (Phase 4).

Architecture:
  Encoder memory: 3 tokens — [patient_repr, note_proj(note), lab_proj(lab)]
  Decoder:        1 TransformerDecoderLayer (causal self-attn + cross-attn to memory)
  Ordering:       Rare-first (ascending training frequency) for teacher forcing.
                  Stable ordering ensures consistent learning targets.
  SGM highway:    gate * hidden + (1-gate) * global_drug_emb — reduces exposure bias
                  so inference (free generation) matches training (teacher forcing).
  VITA blending:  logits = λ*AR_logits + (1-λ)*static_logits at each step.
                  static_logits = MultiHeadCopyPredictor + optional RetrievalHead output.
                  Prevents base model signal from being discarded during AR fine-tuning.
  Training:       Teacher forcing, cross-entropy per step, frozen base model.
  Inference:      Greedy generation with duplicate suppression.

Sources:
  L1  (HI-DR)       — pure Transformer decoder, beam_size=4
  L17 (COGNet)      — rare-first ordering + soft copy gate; rare-first best of 4 orderings
  L18 (SGM)         — SGM highway gate: global embedding blending for exposure bias reduction
  L19 (VITA)        — per-step λ blending of AR and static logits
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressivePredictor(nn.Module):
    """Autoregressive drug sequence predictor.

    Sits on top of a frozen base MIRROR model. At each generation step,
    takes previous drugs as context and outputs logits over all num_drugs drugs.

    Training: teacher forcing on rare-first sorted sequences.
    Inference: greedy generation with de-duplication.
    """

    def __init__(
        self,
        hidden_dim:     int,
        num_drugs:      int,
        note_input_dim: int   = 768,
        lab_input_dim:  int   = 36,
        num_heads:      int   = 4,
        dropout:        float = 0.2,
        max_seq_len:    int   = 35,
        drug_freq:      np.ndarray | None = None,  # (num_drugs,) training frequency
        vita_lambda_init: float = 0.15,  # start mostly-static (15% AR); gate learns to increase AR as quality improves
        # F4 (Run 23): per-step retrieval bias inside the AR loop. Run 22's generate()
        # had no access to retrieval by construction (signature lacked similar_*),
        # so h4's contribution collapsed the moment the AR head took over. These
        # control how neighbour multi-hots feed back into every decode step.
        retrieval_att_tau:  float = 10.0,  # softmax temperature over k neighbours
        retrieval_ar_weight: float = 0.5,  # scale of retrieval bias added to step logits
    ):
        super().__init__()
        self.num_drugs  = num_drugs
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.retrieval_att_tau = float(retrieval_att_tau)
        self.retrieval_ar_weight = float(retrieval_ar_weight)

        # Drug token embedding — vocab: SOS(0) + num_drugs drug tokens
        # Drug i uses token index i+1; SOS = 0.
        self.drug_embed = nn.Embedding(num_drugs + 1, hidden_dim, padding_idx=None)
        self.sos_embed  = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Drug GCN memory injection (HI-DR / COGNet pattern):
        # x = drug_embed(token) + drug_memory_proj(drug_reprs[drug_idx])
        # drug_reprs are from the frozen base model's DrugGNN, cached before the loop.
        # This injects the drug knowledge graph signal into every decoder step,
        # so the AR decoder sees both learned token identity and graph-structured drug context.
        self.drug_memory_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Memory projections: note and lab → hidden_dim
        self.note_mem_proj = nn.Sequential(
            nn.Linear(note_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.lab_mem_proj = nn.Sequential(
            nn.Linear(lab_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional embedding for the target sequence
        self.pos_embed = nn.Embedding(max_seq_len + 2, hidden_dim)

        # Transformer decoder layer (1 layer is sufficient — HI-DR uses 1)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )

        # Output projection: hidden → drug scores
        self.out_proj = nn.Linear(hidden_dim, num_drugs)

        # SGM exposure bias highway gate (L18)
        # gate = sigmoid(W * hidden); output = gate*hidden + (1-gate)*global_emb
        self.sgm_gate   = nn.Linear(hidden_dim, hidden_dim)
        self.global_emb = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # VITA per-step gate (L19) — upgraded to match VITA/HI-DR/COGNet reference.
        # Reference code: generate_prob = sigmoid(W_z(dec_hidden)) — a LINEAR layer
        # that produces a per-position, per-sample gate from the decoder hidden state.
        # This allows early steps (rarest drugs, highest uncertainty) to weight AR
        # and static logits differently than late steps (common drugs, more confident).
        # Bias initialised to logit(vita_lambda_init) so gate ≈ vita_lambda_init at
        # epoch 0. Default is 0.15 — AR starts at ~15 % weight while its logits are
        # still random, then grows as the decoder earns confidence. The prior default
        # of 0.7 caused gradient collapse in Run 21 E3/E4 (gate stuck at 0.671).
        self.vita_gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.vita_gate.weight)  # no hidden-state dependence initially
        nn.init.constant_(
            self.vita_gate.bias,
            float(np.log(vita_lambda_init / (1 - vita_lambda_init)))
        )  # sigmoid(bias) ≈ vita_lambda_init at init

        # vita_lambda kept for backward-compat when loading old checkpoints (not used).
        self.vita_lambda = nn.Parameter(
            torch.tensor(float(np.log(vita_lambda_init / (1 - vita_lambda_init))))
        )

        # Rare-first drug ordering: argsort ascending by training frequency
        # rare_first_order[i] = drug index that is the i-th rarest
        if drug_freq is not None:
            order = np.argsort(drug_freq)  # ascending = rarest first
            self.register_buffer("rare_first_order", torch.from_numpy(order).long())
            # Inverse map: drug_to_rank[d] = rank of drug d in rare-first order
            inv = np.zeros_like(order)
            inv[order] = np.arange(len(order))
            self.register_buffer("drug_to_rank", torch.from_numpy(inv).long())
        else:
            # Fallback: identity ordering
            order = np.arange(num_drugs)
            self.register_buffer("rare_first_order", torch.arange(num_drugs).long())
            self.register_buffer("drug_to_rank",     torch.arange(num_drugs).long())

        self.dropout = nn.Dropout(dropout)

    # ─────────────────────────────────────────
    # Drug GCN memory cache
    # ─────────────────────────────────────────

    def cache_drug_reprs(self, drug_reprs: torch.Tensor) -> None:
        """Cache projected DrugGNN representations for decoder input injection.

        Call once before each training/eval loop using drug reprs from the
        frozen base model. The projected cache is added to token embeddings
        at every decoder step: x = drug_embed(token) + drug_memory[drug_idx].

        Stored as a plain attribute (not a buffer) so it is NOT saved in
        state_dict — it is always recomputed fresh from the current
        drug_memory_proj weights and the base model's DrugGNN output.

        Args:
            drug_reprs: (num_drugs, hidden_dim) — DrugGNN output, on device
        """
        self._drug_memory_cache = self.drug_memory_proj(drug_reprs).detach()  # (num_drugs, H)

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _build_memory(
        self,
        patient_repr: torch.Tensor,   # (B, H)
        note_embed:   torch.Tensor,   # (B, 768)
        lab_vector:   torch.Tensor,   # (B, lab_dim)
        has_note:     torch.Tensor,   # (B,)
        has_lab:      torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Build 3-token cross-attention memory: (B, 3, hidden_dim)."""
        note_mem = self.note_mem_proj(note_embed) * has_note.unsqueeze(1)
        lab_mem  = self.lab_mem_proj(lab_vector)  * has_lab.unsqueeze(1)
        # Stack patient_repr + note + lab as 3 memory tokens
        return torch.stack([patient_repr, note_mem, lab_mem], dim=1)  # (B, 3, H)

    def _apply_sgm(
        self, hidden: torch.Tensor  # (B, T, H)
    ) -> torch.Tensor:
        """SGM exposure bias highway gate (L18)."""
        gate   = torch.sigmoid(self.sgm_gate(hidden))
        global_ = self.global_emb.unsqueeze(0).unsqueeze(0).expand_as(hidden)
        return gate * hidden + (1.0 - gate) * global_

    @staticmethod
    def _znorm_drugs(x: torch.Tensor) -> torch.Tensor:
        """F6: z-normalize along the final (drug) dimension.

        AR logits come from a CE softmax regime (typical std ≈ 8-15); static logits
        come from a BCE sigmoid regime (typical std ≈ 2-6). Adding them directly with
        a scalar gate (as Run 22 did) lets the larger-magnitude branch dominate regardless
        of gate value. Run 23 z-norms both along the drug axis before blending so the gate
        can trade genuine information content rather than absolute magnitude.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    def _apply_vita(
        self,
        ar_logits:     torch.Tensor,           # (B, T, num_drugs)
        static_logits: torch.Tensor | None,    # (B, num_drugs)
        hidden_states: torch.Tensor | None = None,  # (B, T, hidden_dim) — decoder output
    ) -> torch.Tensor:
        """VITA per-step blending (L19) with F6 scale normalization.

        gate = sigmoid(vita_gate(hidden))  → (B, T, 1), one gate per position
        logits = gate * z(AR_logits) + (1-gate) * z(static_logits)

        Z-normalization along the drug dim makes AR and static commensurate before the
        gate. Without it the blend collapses to whichever branch has larger raw std.
        """
        if static_logits is None:
            return ar_logits
        ar_z = self._znorm_drugs(ar_logits)
        static_z = self._znorm_drugs(static_logits).unsqueeze(1).expand_as(ar_z)  # (B, T, D)
        if hidden_states is not None:
            gate = torch.sigmoid(self.vita_gate(hidden_states))  # (B, T, 1)
        else:
            gate = torch.sigmoid(self.vita_lambda)
        return gate * ar_z + (1.0 - gate) * static_z

    # ─────────────────────────────────────────
    # Training forward (teacher forcing)
    # ─────────────────────────────────────────

    def _retrieval_bias_from_query(
        self,
        query:              torch.Tensor,          # (B, T, H) or (B, H)
        similar_reprs:      torch.Tensor | None,   # (B, k, H)
        similar_multihots:  torch.Tensor | None,   # (B, k, D)
        similar_weights:    torch.Tensor | None = None,  # (B, k) — optional cosine prior
    ) -> torch.Tensor | None:
        """F4: per-step retrieval bias. Returns (B, T, D) or (B, D) to add to logits.

        Matches HI-DR's copy_med idea but adapted for MIRROR's AR loop: each decoder
        position attends over the patient's top-k similar training visits (already
        whitened upstream by F2), attention scores go through a fixed att_tau softmax,
        and neighbour multi-hots are summed into a per-step bias. The bias is added
        to the AR logits BEFORE the VITA blend so retrieval can influence both the
        AR branch and — via the blend — the static branch's relative weight.
        """
        if similar_reprs is None or similar_multihots is None:
            return None
        squeeze_T = False
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, H)
            squeeze_T = True
        # (B, T, H) @ (B, H, k) → (B, T, k)
        sim = torch.bmm(query, similar_reprs.transpose(1, 2)) / self.retrieval_att_tau
        if similar_weights is not None:
            # log-prior biases the softmax toward higher-weight neighbours
            sim = sim + torch.log(similar_weights.clamp(min=1e-8)).unsqueeze(1)
        attn = torch.softmax(sim, dim=-1)  # (B, T, k)
        bias = torch.bmm(attn, similar_multihots)  # (B, T, D)
        if squeeze_T:
            bias = bias.squeeze(1)  # (B, D)
        return self.retrieval_ar_weight * bias

    def forward(
        self,
        patient_repr:  torch.Tensor,          # (B, H) — frozen base model output
        note_embed:    torch.Tensor,          # (B, 768)
        lab_vector:    torch.Tensor,          # (B, lab_dim)
        has_note:      torch.Tensor,          # (B,)
        has_lab:       torch.Tensor,          # (B,)
        med_sequence:  torch.Tensor,          # (B, T) — drug indices in rare-first order, -1 = pad
        static_logits: torch.Tensor | None = None,  # (B, num_drugs) — base predictor output
        # F4: per-step retrieval into AR. Leaving these None recovers Run-22 behaviour.
        similar_reprs:     torch.Tensor | None = None,  # (B, k, H) — whitened neighbours
        similar_multihots: torch.Tensor | None = None,  # (B, k, D)
        similar_weights:   torch.Tensor | None = None,  # (B, k) optional cosine prior
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        med_sequence[b, t] is the t-th rarest drug prescribed to patient b.
        Padded positions use value -1 (excluded from loss via the caller).

        Returns:
            logits: (B, T, num_drugs) — per-step drug scores (after VITA blending)
        """
        B, T = med_sequence.shape
        device = patient_repr.device

        # Truncate to max_seq_len: pos_embed has max_seq_len+2 entries (indices 0..max_seq_len+1).
        # Some patients have more drugs than max_seq_len; truncating is correct — the loss
        # is computed only on non-padded positions anyway, and rare drugs beyond max_seq_len
        # are the least informative (rare-first ordering means the important rare ones come first).
        if T > self.max_seq_len:
            med_sequence = med_sequence[:, :self.max_seq_len]
            T = self.max_seq_len

        memory = self._build_memory(patient_repr, note_embed, lab_vector, has_note, has_lab)

        # Build shifted-right input: [SOS, drug_0, drug_1, ..., drug_{T-2}]
        # SOS is represented by sos_embed; drugs use drug_embed(drug_idx + 1)
        sos    = self.sos_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, H)
        # For padded positions (-1), use token 0 (will be masked by loss anyway)
        safe_seq = med_sequence.clamp(min=0)  # -1 → 0 to avoid embedding index error
        drug_emb = self.drug_embed(safe_seq + 1)              # (B, T, H) — offset by 1
        # Inject DrugGNN memory (HI-DR / COGNet pattern):
        # x = drug_embed(token) + drug_memory[drug_idx]
        # Mask injection for padded positions (med_sequence == -1 → safe_seq == 0).
        # Without masking, padded steps would inject drug_0's GNN memory.
        # Since causal self-attention prevents padded positions from affecting earlier
        # positions, this is a cleanliness fix rather than a loss-correctness fix.
        if hasattr(self, "_drug_memory_cache"):
            mem = self._drug_memory_cache[safe_seq]           # (B, T, H)
            valid_mask = (med_sequence >= 0).unsqueeze(-1)    # (B, T, 1) — False for padding
            drug_emb = drug_emb + mem * valid_mask.float()
        # Shift right: input = [sos, drug_0..drug_{T-2}]
        tgt_emb = torch.cat([sos, self.dropout(drug_emb[:, :-1])], dim=1)  # (B, T, H)

        # Add positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        tgt_emb   = tgt_emb + self.pos_embed(positions)

        # Causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Transformer decode
        out = self.decoder_layer(tgt_emb, memory, tgt_mask=causal_mask)  # (B, T, H)

        # SGM highway
        out = self._apply_sgm(out)

        # Drug logits
        ar_logits = self.out_proj(out)  # (B, T, num_drugs)

        # F4: inject per-step retrieval bias BEFORE the VITA blend so retrieval can
        # shape the AR branch the gate then balances against static_logits.
        retrieval_bias = self._retrieval_bias_from_query(
            out, similar_reprs, similar_multihots, similar_weights
        )
        if retrieval_bias is not None:
            ar_logits = ar_logits + retrieval_bias

        # VITA per-step blending: pass hidden_states for per-position gate.
        # F6 z-normalization is applied inside _apply_vita.
        logits = self._apply_vita(ar_logits, static_logits, hidden_states=out)

        return logits

    # ─────────────────────────────────────────
    # Inference (greedy generation)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        patient_repr:  torch.Tensor,          # (B, H)
        note_embed:    torch.Tensor,          # (B, 768)
        lab_vector:    torch.Tensor,          # (B, lab_dim)
        has_note:      torch.Tensor,          # (B,)
        has_lab:       torch.Tensor,          # (B,)
        static_logits: torch.Tensor | None = None,   # (B, num_drugs)
        avg_med_count: float = 20.0,          # fallback sequence length for legacy "avg" mode
        length_mode: str = "base_threshold",  # F5: default flipped from "avg" → per-patient adaptive
        base_threshold: float = 0.5,
        min_steps: int = 5,                   # F5: clamp [5, 35]
        max_steps: int | None = 35,           # F5: clamp [5, 35]
        # F4: per-step retrieval into AR generate() — the signature Run 22 lacked.
        similar_reprs:     torch.Tensor | None = None,  # (B, k, H)
        similar_multihots: torch.Tensor | None = None,  # (B, k, D)
        similar_weights:   torch.Tensor | None = None,  # (B, k)
    ) -> torch.Tensor:
        """Greedy generation → multi-hot (B, num_drugs) binary predictions.

        F5 (Run 23): default length_mode flipped from "avg" (fixed ≈ 19 meds
        per patient, catastrophic on tails) to "base_threshold" (per-patient
        n_steps = count(sigmoid(static_logits) > threshold), clamped to
        [min_steps, max_steps]). AR now reorders an adaptive set rather than
        generating a fixed-length one.

        F4: if similar_reprs/similar_multihots are supplied, a per-step retrieval
        bias is added to the AR step logits before the VITA blend — the same
        mechanism used in forward().

        F6: ar_logits and static_logits are z-normalized along the drug dim
        before the VITA blend so scale mismatch cannot swamp the gate.
        """
        B      = patient_repr.size(0)
        device = patient_repr.device
        D      = self.num_drugs

        memory = self._build_memory(patient_repr, note_embed, lab_vector, has_note, has_lab)

        min_steps = max(1, int(min_steps))
        if max_steps is None:
            max_steps = self.max_seq_len
        max_steps = max(min_steps, min(int(max_steps), self.max_seq_len))

        if length_mode == "base_threshold" and static_logits is not None:
            base_probs = torch.sigmoid(static_logits)
            target_steps = (base_probs >= float(base_threshold)).sum(dim=1)
            target_steps = target_steps.clamp(min=min_steps, max=max_steps)
        else:
            fixed_steps = max(min_steps, int(round(avg_med_count)))
            fixed_steps = min(fixed_steps, max_steps)
            target_steps = torch.full((B,), fixed_steps, dtype=torch.long, device=device)

        n_steps = int(target_steps.max().item())

        # Accumulated token sequence: starts with just SOS
        # We build it step by step
        generated_tokens = []  # list of (B,) tensors of drug token indices (1..num_drugs+1)
        generated_mask   = torch.zeros(B, D, dtype=torch.bool, device=device)  # already picked

        for step in range(n_steps):
            active = target_steps > step
            if not bool(active.any()):
                break

            T_cur = step + 1  # current sequence length (SOS + step drugs)

            # Build current input sequence
            sos   = self.sos_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, H)
            if generated_tokens:
                prev_token_ids = torch.stack(generated_tokens, dim=1)  # (B, step) — 1-based
                prev_emb = self.drug_embed(prev_token_ids)              # (B, step, H)
                # Inject DrugGNN memory: x = drug_embed(token) + drug_memory[drug_idx]
                if hasattr(self, "_drug_memory_cache"):
                    prev_drug_ids = prev_token_ids - 1  # convert to 0-based drug indices
                    prev_emb = prev_emb + self._drug_memory_cache[prev_drug_ids]
                tgt_emb = torch.cat([sos, self.dropout(prev_emb)], dim=1)  # (B, T_cur, H)
            else:
                tgt_emb = sos  # (B, 1, H)

            positions = torch.arange(T_cur, device=device).unsqueeze(0)
            tgt_emb   = tgt_emb + self.pos_embed(positions)

            causal_mask = torch.triu(
                torch.ones(T_cur, T_cur, device=device, dtype=torch.bool), diagonal=1
            )

            out = self.decoder_layer(tgt_emb, memory, tgt_mask=causal_mask)  # (B, T_cur, H)
            out = self._apply_sgm(out)

            last_hidden = out[:, -1, :]                # (B, H) — last decoder position
            step_logits = self.out_proj(last_hidden)   # (B, D)

            # F4: inject per-step retrieval bias into the AR branch before the VITA blend.
            step_bias = self._retrieval_bias_from_query(
                last_hidden, similar_reprs, similar_multihots, similar_weights
            )
            if step_bias is not None:
                step_logits = step_logits + step_bias

            # VITA per-step gate — matches VITA/HI-DR/COGNet reference implementation.
            # F6: z-normalize both branches along the drug dim before blending so
            # AR's CE-softmax magnitudes don't swamp BCE-sigmoid static magnitudes.
            if static_logits is not None:
                gate = torch.sigmoid(self.vita_gate(last_hidden))  # (B, 1)
                ar_z = self._znorm_drugs(step_logits)
                static_z = self._znorm_drugs(static_logits)
                step_logits = gate * ar_z + (1.0 - gate) * static_z

            # Mask already-generated drugs
            step_logits = step_logits.masked_fill(generated_mask, float("-inf"))

            # Greedy pick
            next_drug = step_logits.argmax(dim=-1)  # (B,) drug indices 0..D-1

            # Record
            generated_tokens.append(next_drug + 1)        # +1 for token offset
            active_idx = active.nonzero(as_tuple=True)[0]
            if active_idx.numel() > 0:
                generated_mask[active_idx, next_drug[active_idx]] = True

        # Convert to multi-hot
        multihot = generated_mask.float()
        return multihot


# ─────────────────────────────────────────────────────────────────
# Loss function for AR training
# ─────────────────────────────────────────────────────────────────

def ar_sequence_loss(
    logits:       torch.Tensor,   # (B, T, num_drugs) — from AutoregressivePredictor.forward()
    med_sequence: torch.Tensor,   # (B, T) — target drug indices (rare-first), -1 = pad
) -> torch.Tensor:
    """Cross-entropy loss over valid (non-padded) AR steps.

    At step t, the model predicts the t-th drug in the rare-first sequence.
    Padded positions (med_sequence == -1) are excluded from the loss.

    Returns scalar loss.
    """
    B, T, D = logits.shape

    # Align target sequence length to decoder output length.
    # The dataloader pads med_sequence to the longest sequence in the batch,
    # which can exceed the decoder's max_seq_len when some patients have very
    # large prescription sets.  Truncate or pad with -1 (ignored by the mask).
    T_seq = med_sequence.size(1)
    if T_seq > T:
        med_sequence = med_sequence[:, :T]
    elif T_seq < T:
        pad = med_sequence.new_full((B, T - T_seq), -1)
        med_sequence = torch.cat([med_sequence, pad], dim=1)

    # Flatten over batch × time
    logits_flat = logits.reshape(B * T, D)
    targets_flat = med_sequence.reshape(B * T)

    # Mask: only compute loss on non-padded positions
    valid = targets_flat >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return F.cross_entropy(logits_flat[valid], targets_flat[valid])
