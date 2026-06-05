"""
src/generate.py — sample text from a trained checkpoint
Usage: python src/generate.py --prompt "The meaning of life is"
"""

import argparse
import torch
from transformers import GPT2TokenizerFast
from model import GPT, GPTConfig
from train import TrainConfig

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= 1024 else x[:, -1024:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model_config = GPTConfig(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
    )
    model = GPT(model_config).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    output = generate(model, tokenizer, args.prompt,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      device=device)
    print("\n" + "="*60)
    print(output)
    print("="*60)


if __name__ == "__main__":
    main()
