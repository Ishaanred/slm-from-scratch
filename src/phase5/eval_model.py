"""
lm-evaluation-harness adapter for this project's own GPT checkpoints.

This project's model isn't a HuggingFace AutoModelForCausalLM, and its
tokenizers (GPT-2 for Phase 1, a custom 32K BPE for Phase 3) aren't wired
into lm-eval's HF loader either. This is a thin translation layer only:
implements the harness's LM interface (loglikelihood, loglikelihood_rolling,
generate_until) against the existing GPT class and checkpoint files.
Infrastructure, not a re-architecture — see docs/phase5/sprint-plan.md.

Usage (from Python, see run_eval.py):
    lm = SLMEvalWrapper(
        checkpoint="checkpoints/phase3/150m_500m/final.pt",
        tokenizer="bpe32k", n_layer=12, n_head=12, n_embd=1024, ...
    )
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

PHASE3_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3")
sys.path.insert(0, PHASE3_SRC)
from model import GPT, GPTConfig  # noqa: E402


class GPT2Tok:
    """Wraps the GPT-2 tokenizer (Phase 1 checkpoints) with a unified interface."""

    def __init__(self):
        from transformers import GPT2TokenizerFast
        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(self, text):
        return self._tok.encode(text)

    def decode(self, ids):
        return self._tok.decode(ids)


class BPE32kTok:
    """Wraps this project's own 32K BPE tokenizer (Phase 3 checkpoints)."""

    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)


@register_model("slm_gpt")
class SLMEvalWrapper(LM):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        n_layer,
        n_head,
        n_embd,
        vocab_size,
        block_size=1024,
        tokenizer_path=None,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self._device = device
        self.block_size = block_size

        if tokenizer == "gpt2":
            self.tok = GPT2Tok()
        elif tokenizer == "bpe32k":
            self.tok = BPE32kTok(tokenizer_path)
        else:
            raise ValueError(f"unknown tokenizer {tokenizer!r}")

        config = GPTConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            vocab_size=vocab_size, block_size=block_size,
        )
        self.model = GPT(config).to(device)
        ckpt = self._load_checkpoint_ignoring_config_class(checkpoint, device)
        state_dict = ckpt["model"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _load_checkpoint_ignoring_config_class(checkpoint, device):
        """Checkpoints pickle the training run's TrainConfig object, which needs
        its original class importable to unpickle. We only need `model` (the
        state_dict) and don't care about the config object, so register a
        harmless stand-in class under every module name pickle might look for it
        under, rather than depend on the exact original train.py being importable."""
        import __main__
        for mod_name in ("__main__", "train"):
            mod = sys.modules.get(mod_name)
            if mod is None:
                import types
                mod = types.ModuleType(mod_name)
                sys.modules[mod_name] = mod
            if not hasattr(mod, "TrainConfig"):
                mod.TrainConfig = type("TrainConfig", (), {})
        return torch.load(checkpoint, map_location=device, weights_only=False)

    @torch.no_grad()
    def _logprobs_for_continuation(self, context, continuation):
        ctx_ids = self.tok.encode(context) if context else []
        cont_ids = self.tok.encode(continuation)
        if len(cont_ids) == 0:
            return 0.0, True

        ids = ctx_ids + cont_ids
        # keep the continuation intact, truncate context from the left if needed
        if len(ids) > self.block_size:
            overflow = len(ids) - self.block_size
            ctx_ids = ctx_ids[overflow:]
            ids = ctx_ids + cont_ids
        n_ctx = len(ctx_ids)

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx, targets=idx)  # targets=idx just forces full-sequence logits
        logits = logits[0]  # (T, vocab)

        # logits[i] predicts token i+1; continuation starts at position n_ctx
        pred_logits = logits[n_ctx - 1: n_ctx - 1 + len(cont_ids)]
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        target = torch.tensor(cont_ids, device=self.device)
        token_logprobs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        is_greedy = bool((pred_logits.argmax(dim=-1) == target).all().item())
        return token_logprobs.sum().item(), is_greedy

    def loglikelihood(self, requests, disable_tqdm=False):
        res = []
        for req in tqdm(requests, disable=disable_tqdm):
            context, continuation = req.args
            res.append(self._logprobs_for_continuation(context, continuation))
        return res

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        res = []
        for req in tqdm(requests, disable=disable_tqdm):
            (text,) = req.args
            ids = self.tok.encode(text)
            total = 0.0
            for start in range(0, len(ids), self.block_size):
                chunk = ids[start: start + self.block_size + 1]
                if len(chunk) < 2:
                    continue
                idx = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
                target = torch.tensor(chunk[1:], device=self.device)
                logits, _ = self.model(idx, targets=idx)
                log_probs = F.log_softmax(logits[0].float(), dim=-1)
                total += log_probs.gather(1, target.unsqueeze(1)).squeeze(1).sum().item()
            res.append(total)
        return res

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm=False):
        res = []
        for req in tqdm(requests, disable=disable_tqdm):
            context, gen_kwargs = req.args
            max_new = gen_kwargs.get("max_gen_toks", 64) if isinstance(gen_kwargs, dict) else 64
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            ids = self.tok.encode(context)[-self.block_size:]
            for _ in range(max_new):
                idx = torch.tensor([ids[-self.block_size:]], dtype=torch.long, device=self.device)
                logits, _ = self.model(idx)  # targets=None -> last-position logits only
                next_id = int(logits[0, -1].argmax().item())
                ids.append(next_id)
                partial = self.tok.decode(ids)
                if any(u and partial.endswith(u) for u in until):
                    break
            text = self.tok.decode(ids)
            res.append(text[len(context):] if text.startswith(context) else text)
        return res
