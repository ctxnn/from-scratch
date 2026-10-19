// Generated from the source repositories — verbatim excerpts.
window.SNIPPETS = {
 "reigrad": {
  "repo": "reigrad",
  "file": "engine.py",
  "notebook": false,
  "chunks": [
   {
    "start": 79,
    "code": "def backward(self):\n    # write topological sort (it sorts the graph(we constructed the graph using _prev) left to right)\n    topo_order = []\n    visited = set()\n    def build_order(v):\n        if v not in visited:\n            visited.add(v)\n            for child in v._prev:\n                build_order(child)\n            topo_order.append(v)\n    build_order(self)\n\n    # calculating grad one by one and applying the chain rule to get their gradient\n    self.grad = 1\n    for v in reversed(topo_order):\n        v._backward()"
   }
  ]
 },
 "lora": {
  "repo": "lora-llms",
  "file": "LoRA.ipynb",
  "notebook": true,
  "chunks": [
   {
    "start": null,
    "code": "class LoRA(nn.Module):\n    def __init__(self, features_in, features_out, r=1, alpha=1, device='cpu'):\n        super(LoRA, self).__init__()\n\n        self.r = r\n        self.alpha = alpha\n        self.scale = self.alpha / self.r\n        self.lora_A = nn.Parameter(torch.zeros((r,features_out)).to(device))\n        self.lora_B = nn.Parameter(torch.zeros((features_in, r)).to(device))\n        nn.init.normal_(self.lora_A, mean=0, std=1)\n        self.enabled = True\n\n    def forward(self, original_weights):\n        if self.enabled:\n            # Return W + (B*A)*scale\n            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale\n        else:\n            return original_weights"
   }
  ]
 },
 "moe": {
  "repo": "moe",
  "file": "moe_rev.py",
  "notebook": false,
  "chunks": [
   {
    "start": 45,
    "code": "router_logits = self.router(x_flat)\nrouting_probs = F.softmax(router_logits, dim=-1)\n\n# Selecting the top_k experts for each input token.\ntop_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)\ntop_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)"
   },
   {
    "start": 58,
    "code": "for i, expert in enumerate(self.experts):\n    # Create a boolean mask for tokens assigned to the current expert (i)\n    expert_mask = top_k_indices == i"
   },
   {
    "start": 90,
    "code": "    final_output[token_indices] += expert_output * token_weights"
   }
  ]
 },
 "toy-gpt": {
  "repo": "toy-gpt",
  "file": "toygpt.py",
  "notebook": false,
  "chunks": [
   {
    "start": 88,
    "code": "wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)\nwei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)\nwei = F.softmax(wei, dim=-1) # (B, T, T)\n\n# Apply dropout for regularization\nwei = self.dropout(wei)\n# Compute weighted sum of values\nout = wei @ v\nreturn out"
   }
  ]
 },
 "transformer": {
  "repo": "transformer-neural-machine-translation",
  "file": "model.py",
  "notebook": false,
  "chunks": [
   {
    "start": 100,
    "code": "def attention(query, key, value, mask, dropout: nn.Dropout):\n    d_k = query.shape[-1]\n    # Just apply the formula from the paper\n    # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)\n    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        # Write a very low value (indicating -inf) to the positions where mask == 0\n        attention_scores.masked_fill_(mask == 0, -1e9)\n    attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax\n    if dropout is not None:\n        attention_scores = dropout(attention_scores)\n    # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)\n    # return attention scores which can be used for visualization\n    return (attention_scores @ value), attention_scores"
   }
  ]
 },
 "autoencoder": {
  "repo": "autoencoder-pytorch",
  "file": "ae_playground.ipynb",
  "notebook": true,
  "chunks": [
   {
    "start": null,
    "code": "# Encoder\nself.encoder = nn.Sequential(\n    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # [16, 14, 14]\n    nn.ReLU(True),\n    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [32, 7, 7]\n    nn.ReLU(True),\n    nn.Conv2d(32, LATENT_SPACE_DIM, kernel_size=7) # [LATENT_SPACE_DIM, 1, 1]\n)\n\n# Decoder\nself.decoder = nn.Sequential(\n    nn.ConvTranspose2d(LATENT_SPACE_DIM, 32, kernel_size=7), # [32, 7, 7]\n    nn.ReLU(True),\n    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [16, 14, 14]\n    nn.ReLU(True),\n    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # [1, 28, 28]\n    nn.Sigmoid() # Output pixel values between 0 and 1\n)"
   }
  ]
 },
 "vae": {
  "repo": "vae-pytorch",
  "file": "vae_pytorch.ipynb",
  "notebook": true,
  "chunks": [
   {
    "start": null,
    "code": "def reparameterize(self, mu, logvar):\n    std = torch.exp(0.5 * logvar)\n    eps = torch.randn_like(std)\n    return mu + eps * std"
   },
   {
    "start": null,
    "code": "def vae_loss(recon_x, x, mu, logvar):\n    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')\n    # KL divergence between learned distribution and standard normal\n    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())\n    return BCE + KLD"
   }
  ]
 },
 "gan": {
  "repo": "gan-pytorch",
  "file": "mnist-celeba-gan.ipynb",
  "notebook": true,
  "chunks": [
   {
    "start": null,
    "code": "flipped_fake_labels = real_labels # here, fake label = 1"
   },
   {
    "start": null,
    "code": "discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)\nfake_loss = loss_fn(discr_pred_fake, fake_labels)\n# fake_loss.backward()\n\n# combined loss\ndiscr_loss = 0.5*(real_loss + fake_loss)\ndiscr_loss.backward()"
   },
   {
    "start": null,
    "code": "# get discriminator loss on fake images with flipped labels\ndiscr_pred_fake = model.discriminator_forward(fake_images).view(-1)\ngener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)\ngener_loss.backward()"
   }
  ]
 },
 "diffusion": {
  "repo": "diffusion-pytorch",
  "file": "diffusion_pytorch.ipynb",
  "notebook": true,
  "chunks": [
   {
    "start": null,
    "code": "betas_t = self.get_index_from_list(self.betas, t, x.shape)\nsqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)\nsqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)\nmean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)\nposterior_variance_t = betas_t\n\nif t == 0:\n    return mean\nelse:\n    noise = torch.randn_like(x)\n    variance = torch.sqrt(posterior_variance_t) * noise\n    return mean + variance"
   }
  ]
 },
 "reifusion": {
  "repo": "reifusion",
  "file": "model/pipeline.py",
  "notebook": false,
  "chunks": [
   {
    "start": 113,
    "code": "for i, timestep in enumerate(timesteps):\n    # (1, 320)\n    time_embedding = get_time_embedding(timestep).to(device)\n\n    # (Batch_Size, 4, Latents_Height, Latents_Width)\n    model_input = latents\n\n    if do_cfg:\n        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)\n        model_input = model_input.repeat(2, 1, 1, 1)\n\n    # model_output is the predicted noise\n    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)\n    model_output = diffusion(model_input, context, time_embedding)\n\n    if do_cfg:\n        output_cond, output_uncond = model_output.chunk(2)\n        model_output = cfg_scale * (output_cond - output_uncond) + output_uncond\n\n    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)\n    latents = sampler.step(timestep, latents, model_output)"
   }
  ]
 },
 "frozenlake": {
  "repo": "frozenlake-rl",
  "file": "frozenlake_dqn.py",
  "notebook": false,
  "chunks": [
   {
    "start": 170,
    "code": "if terminated: \n    # For terminal states (goal/hole), target Q-value equals immediate reward\n    # No future rewards are possible after termination\n    target = torch.FloatTensor([reward])\nelse:\n    # For non-terminal states, calculate target Q-value using Bellman equation\n    # Q(s,a) = r + γ * max(Q(s',a')) where γ is discount factor\n    with torch.no_grad():  # Disable gradient calculation for target network\n        target = torch.FloatTensor(\n            reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()\n        )"
   },
   {
    "start": 188,
    "code": "target_q = target_dqn(self.state_to_dqn_input(state, num_states)) \ntarget_q[action] = target  # Update specific action's Q-value"
   }
  ]
 },
 "flappy": {
  "repo": "flappy-bird-rl",
  "file": "agent.py",
  "notebook": false,
  "chunks": [
   {
    "start": 289,
    "code": "with torch.no_grad():\n    # Calculate target Q values (expected returns)\n    target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]"
   },
   {
    "start": 298,
    "code": "# Calcuate Q values from current policy\ncurrent_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()"
   },
   {
    "start": 307,
    "code": "# Compute loss\nloss = self.loss_fn(current_q, target_q)\n\n# Optimize the model (backpropagation)\nself.optimizer.zero_grad()  # Clear gradients\nloss.backward()             # Compute gradients\nself.optimizer.step()       # Update network parameters i.e. weights and biases"
   }
  ]
 }
};
