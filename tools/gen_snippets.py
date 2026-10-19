import json, sys, os
SRC = sys.argv[1]; OUT = sys.argv[2]

def lines_of(repo, path, cell=None):
    p = os.path.join(SRC, repo, path)
    if cell is None:
        return open(p).read().split("\n")
    nb = json.load(open(p))
    return "".join(nb["cells"][cell]["source"]).split("\n")

def find(lines, needle, start=0):
    for i in range(start, len(lines)):
        if (lines[i].strip() == needle[1:]) if needle.startswith("=") else (needle in lines[i]):
            return i
    raise SystemExit(f"not found: {needle!r}")

# each chunk: (from_text, to_text) inclusive, matched in order
SPECS = {
 "reigrad":   dict(repo="reigrad", path="engine.py", chunks=[("def backward", "v._backward()")]),
 "lora":      dict(repo="lora-llms", path="LoRA.ipynb", cell=8, chunks=[("class LoRA", "=return original_weights")]),
 "moe":       dict(repo="moe", path="moe_rev.py", chunks=[("router_logits = self.router", "top_k_probs = top_k_probs /"),
                                                           ("for i, expert in enumerate", "expert_mask = top_k_indices == i"),
                                                           ("final_output[token_indices] +=", "final_output[token_indices] +=")]),
 "toy-gpt":   dict(repo="toy-gpt", path="toygpt.py", chunks=[("wei = q @ k.transpose(-2,-1) * k.shape", "return out")]),
 "transformer": dict(repo="transformer-neural-machine-translation", path="model.py",
                     chunks=[("def attention(query, key, value, mask", "return (attention_scores @ value)")]),
 "autoencoder": dict(repo="autoencoder-pytorch", path="ae_playground.ipynb", cell=14, chunks=[("# Encoder", "nn.Sigmoid()", 1)]),
 "vae":       dict(repo="vae-pytorch", path="vae_pytorch.ipynb", cell=4, chunks=[("def reparameterize", "return mu + eps * std")],
                   dedent_each=True, extra=dict(cell=5, chunks=[("def vae_loss", "return BCE + KLD")])),
 "gan":       dict(repo="gan-pytorch", path="mnist-celeba-gan.ipynb", cell=None, chunks=[]),
 "diffusion": dict(repo="diffusion-pytorch", path="diffusion_pytorch.ipynb", cell=None, chunks=[]),
 "reifusion": dict(repo="reifusion", path="model/pipeline.py", chunks=[("for i, timestep in enumerate(timesteps)", "latents = sampler.step(")]),
 "frozenlake": dict(repo="frozenlake-rl", path="frozenlake_dqn.py", chunks=[("if terminated:", "reward + self.discount_factor_g", 1), ("target_q = target_dqn(", "target_q[action] = target")]),
 "flappy":    dict(repo="flappy-bird-rl", path="agent.py", chunks=[("with torch.no_grad():\n", None)]),
}

def cell_with(repo, path, needle):
    nb = json.load(open(os.path.join(SRC, repo, path)))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and needle in "".join(c["source"]):
            return i
    raise SystemExit(f"no cell with {needle!r}")

def extract(repo, path, cell, chunks):
    L = lines_of(repo, path, cell)
    out, pos = [], 0
    for ch in chunks:
        a, b, extra = (ch + (0,))[:3]
        i = find(L, a.rstrip("\n"), pos)
        j = find(L, b.rstrip("\n"), i) + extra
        out.append({"start": i + 1, "lines": L[i:j + 1]})
        pos = j + 1
    return out

result = {}
for key, s in SPECS.items():
    repo, path = s["repo"], s["path"]
    if key == "gan":
        c = cell_with(repo, path, "def train_gan")
        chunks = extract(repo, path, c, [("flipped_fake_labels = real_labels", "flipped_fake_labels = real_labels"),
            ("discr_pred_fake = model.discriminator_forward(fake_images.detach())", "discr_loss.backward()"),
            ("# get discriminator loss on fake images with flipped labels", "gener_loss.backward()")])
    elif key == "diffusion":
        c = cell_with(repo, path, "class DiffusionModel")
        chunks = extract(repo, path, c, [("betas_t = self.get_index_from_list(self.betas", "return mean + variance")])
    elif key == "flappy":
        chunks = []
        L = lines_of(repo, path)
        i = find(L, "target_q = rewards + (1-terminations)")
        chunks.append({"start": i, "lines": L[i-2:i+1]}); chunks[-1]["start"] = i - 1
        i = find(L, "current_q = policy_dqn(states).gather")
        chunks.append({"start": i, "lines": L[i-1:i+1]})
        i = find(L, "loss = self.loss_fn(current_q, target_q)")
        j = find(L, "self.optimizer.step()", i)
        chunks.append({"start": i, "lines": L[i-1:j+1]})
    else:
        chunks = extract(repo, path, s.get("cell"), s["chunks"])
        if "extra" in s:
            chunks += extract(repo, path, s["extra"]["cell"], s["extra"]["chunks"])
    # dedent all chunks by the common indent
    ind = min(len(l) - len(l.lstrip()) for ch in chunks for l in ch["lines"] if l.strip())
    for ch in chunks:
        if s.get("dedent_each"):
            ind = min(len(l) - len(l.lstrip()) for l in ch["lines"] if l.strip())
        ch["lines"] = [l[ind:] if l.strip() else "" for l in ch["lines"]]
        while ch["lines"] and not ch["lines"][-1].strip(): ch["lines"].pop()
    nb = path.endswith(".ipynb")
    result[key] = {"repo": repo, "file": path, "notebook": nb,
                   "chunks": [{"start": None if nb else c["start"], "code": "\n".join(c["lines"]).rstrip()} for c in chunks]}

with open(OUT, "w") as f:
    f.write("// Generated from the source repositories — verbatim excerpts.\nwindow.SNIPPETS = ")
    json.dump(result, f, indent=1, ensure_ascii=False)
    f.write(";\n")
for k, v in result.items():
    print("=====", k, v["file"])
    for c in v["chunks"]:
        print(f"[L{c['start']}]"); print(c["code"])
