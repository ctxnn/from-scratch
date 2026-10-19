// Interactive figures, one per project. Each widget renders into its root element.
// Everything is computed in the browser; images come from the project repos.
(function () {
  "use strict";

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (n, d = 3) => (Object.is(n, -0) ? 0 : n).toFixed(d);
  const int = (n) => Math.round(n).toLocaleString("en-US");

  /* ---------- tiny UI kit ---------- */
  function seg(name, opts, cur, label) {
    return `<div class="seg" role="group" aria-label="${esc(label || name)}" data-seg="${name}">${opts
      .map(([v, l]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(cur)}">${esc(l)}</button>`)
      .join("")}</div>`;
  }
  function onSeg(root, name, fn) {
    const g = root.querySelector(`[data-seg="${name}"]`);
    g.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      g.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      fn(b.dataset.v);
    });
  }
  function slider(name, label, min, max, step, val) {
    return `<label class="rng"><span>${esc(label)}</span><input type="range" data-r="${name}" min="${min}" max="${max}" step="${step}" value="${val}"><output data-o="${name}"></output></label>`;
  }
  const $ = (root, sel) => root.querySelector(sel);

  // Seeded PRNG + Gaussian, so demos are the same on every load.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const hash = (s) => [...s].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261);

  const ARROW_DEFS = `<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
    <marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="dg-arrow"/></marker>
    <marker id="ah-hl" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="dg-arrow hl"/></marker>
  </defs></svg>`;
  document.body.insertAdjacentHTML("afterbegin", ARROW_DEFS);

  /* Swappable image tabs (results straight from the repos). */
  function imageTabs(root, tabs, opts = {}) {
    const first = tabs[0];
    root.innerHTML = `
      <div class="fig-frame plate${opts.narrow ? " narrow" : ""}"><img src="${first.img}" width="${first.w}" height="${first.h}" alt="${esc(first.note)}" decoding="async"></div>
      <div class="wbar">${seg("tab", tabs.map((t, i) => [i, t.label]), 0, opts.label)}</div>
      <p class="wnote">${esc(first.note)}</p>`;
    const img = $(root, "img"), note = $(root, ".wnote");
    onSeg(root, "tab", (i) => {
      const t = tabs[+i];
      img.src = t.img; img.width = t.w; img.height = t.h; img.alt = t.note;
      note.textContent = t.note;
    });
    tabs.slice(1).forEach((t) => { const p = new Image(); p.src = t.img; });
  }

  const W = {};

  /* ---------- reifusion: walk the pipeline, see the tensor at each stage ---------- */
  W.pipeline = (root) => {
    const stages = [
      ["Prompt", "text", "pipeline.py", "The prompt, plus an empty prompt that classifier-free guidance compares against."],
      ["Tokenize", "(1, 77) × 2", "pipeline.py", "CLIP byte-pair tokens, padded to 77, for both prompts."],
      ["CLIP encoder", "(2, 77, 768)", "clip.py", "12 transformer layers turn tokens into the context the U-Net cross-attends to."],
      ["Latent noise", "(1, 4, 64, 64)", "pipeline.py", "Generation starts from Gaussian noise in latent space, 48× smaller than the 512×512 image."],
      ["U-Net × 50", "(2, 4, 64, 64) → (1, 4, 64, 64)", "diffusion.py · ddpm.py", "Each step predicts the noise for both prompts, mixes them at guidance scale 7.5, and the DDPM sampler removes it."],
      ["VAE decoder", "(1, 3, 512, 512)", "decoder.py", "The final latent is decoded back into pixels."],
    ];
    root.innerHTML = `
      <div class="pipe">
        <div class="pipe-img"><img src="assets/img/reifusion.webp" width="512" height="512" alt="Generated image of a black and white cat stretching on a wooden floor" decoding="async"></div>
        <ol class="pipe-steps">${stages.map(([k], i) => `<li><button type="button" data-i="${i}" aria-pressed="${i === 4}"><span class="pipe-n">${i + 1}</span>${esc(k)}</button></li>`).join("")}</ol>
        <div class="pipe-detail" aria-live="polite"></div>
      </div>`;
    const detail = $(root, ".pipe-detail");
    const show = (i) => {
      const [k, shape, file, text] = stages[i];
      detail.innerHTML = `<p class="pipe-shape">${esc(shape)}</p><p class="pipe-text">${esc(text)}</p><p class="wnote">model/${esc(file)}</p>`;
      root.querySelectorAll(".pipe-steps button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.i === i)));
    };
    $(root, ".pipe-steps").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(4);
  };

  /* ---------- DDPM: the forward process, computed live on an image ---------- */
  W.ddpm = (root) => {
    const T = 300, betas = [], abar = [];
    let prod = 1;
    for (let t = 0; t < T; t++) { const b = 1e-4 + (0.02 - 1e-4) * t / (T - 1); betas.push(b); prod *= 1 - b; abar.push(prod); }
    const pts = abar.map((a, t) => `${(t / (T - 1)) * 200},${60 - Math.sqrt(a) * 56}`).join(" ");
    root.innerHTML = `
      <div class="ddpm">
        <canvas width="128" height="128" aria-label="Image with Gaussian noise added at the chosen timestep"></canvas>
        <div class="ddpm-side">
          <p class="ddpm-t">t = <b data-o="t">0</b> / ${T}</p>
          <dl class="ddpm-read"><div><dt>signal √ᾱₜ</dt><dd data-o="s"></dd></div><div><dt>noise √(1−ᾱₜ)</dt><dd data-o="n"></dd></div></dl>
          <svg viewBox="-4 -4 208 72" class="ddpm-chart" aria-hidden="true">
            <polyline class="dg-line" points="${pts}"/>
            <line class="dg-line hl" data-o="mark" x1="0" x2="0" y1="0" y2="60"/>
            <text class="dg-tm" x="0" y="72" style="font-size:9px">√ᾱₜ over 300 steps</text>
          </svg>
        </div>
      </div>
      <div class="wbar">${slider("t", "timestep", 0, T - 1, 1, 0)}<button type="button" class="wbtn" data-act="play">Play ▸</button></div>
      <p class="wnote">x<sub>t</sub> = √ᾱ<sub>t</sub>·x<sub>0</sub> + √(1−ᾱ<sub>t</sub>)·ε with the notebook's linear β schedule (10⁻⁴ → 0.02).</p>
      <figure class="ddpm-nb"><img src="assets/img/diffusion.webp" width="1600" height="311" alt="Forward process from the notebook: an image becoming noise over five steps" loading="lazy" decoding="async"><figcaption>From the notebook: the same process on a training image.</figcaption></figure>`;
    const cv = $(root, "canvas"), ctx = cv.getContext("2d", { willReadFrequently: true });
    const input = $(root, '[data-r="t"]'), outT = $(root, '[data-o="t"]');
    const outS = $(root, '[data-o="s"]'), outN = $(root, '[data-o="n"]'), mark = $(root, '[data-o="mark"]');
    const N = 128 * 128, x0 = new Float32Array(N * 3), eps = new Float32Array(N * 3);
    const r = rng(7); for (let i = 0; i < eps.length; i++) eps[i] = gauss(r);
    let ready = false, frame = ctx.createImageData(128, 128);
    const draw = () => {
      const t = +input.value, a = Math.sqrt(abar[t]), b = Math.sqrt(1 - abar[t]);
      outT.textContent = t; outS.textContent = fmt(a); outN.textContent = fmt(b);
      const mx = (t / (T - 1)) * 200; mark.setAttribute("x1", mx); mark.setAttribute("x2", mx);
      if (!ready) return;
      const d = frame.data;
      for (let i = 0, j = 0; i < N; i++, j += 4) {
        for (let c = 0; c < 3; c++) {
          const v = a * x0[i * 3 + c] + b * eps[i * 3 + c];
          d[j + c] = Math.max(0, Math.min(255, (v + 1) * 127.5));
        }
        d[j + 3] = 255;
      }
      ctx.putImageData(frame, 0, 0);
    };
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, 128, 128);
      try {
        const src = ctx.getImageData(0, 0, 128, 128).data;
        for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) x0[i * 3 + c] = src[i * 4 + c] / 127.5 - 1;
        ready = true;
      } catch (e) { /* file:// pages can't read pixels; the slider still updates the numbers */ }
      draw();
    };
    img.src = "assets/img/ddpm-x0.png";
    input.addEventListener("input", draw);
    let timer = null;
    const btn = $(root, '[data-act="play"]');
    btn.addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; btn.textContent = "Play ▸"; return; }
      if (+input.value >= T - 1) input.value = 0;
      btn.textContent = "Pause ❚❚";
      timer = setInterval(() => {
        input.value = Math.min(T - 1, +input.value + 3); draw();
        if (+input.value >= T - 1) { clearInterval(timer); timer = null; btn.textContent = "Play ▸"; }
      }, 30);
    });
    draw();
  };

  /* ---------- Transformer: sinusoidal positional encoding, live ---------- */
  W.posenc = (root) => {
    const D = 512, P = 64, C = 128;
    const pe = (pos, j) => { const i = Math.floor(j / 2), ang = pos / Math.pow(10000, (2 * i) / D); return j % 2 ? Math.cos(ang) : Math.sin(ang); };
    // Column c shows the sin dims (even) on the left half and the matching cos dims (odd) on the right.
    const dimOf = (c) => (c < C / 2 ? 2 * c : 2 * (c - C / 2) + 1);
    root.innerHTML = `
      <div class="pe">
        <canvas width="${C * 4}" height="${P * 3}" aria-label="Heatmap of positional encodings: 64 positions; sine dimensions on the left, cosine on the right"></canvas>
        <svg viewBox="0 0 512 70" class="pe-line" aria-hidden="true">
          <line class="dg-line" x1="0" x2="512" y1="35" y2="35" style="stroke:var(--line)"/>
          <line class="dg-line" x1="256" x2="256" y1="2" y2="68" style="stroke:var(--line);stroke-dasharray:3 3"/>
          <polyline class="dg-line hl" points=""/>
        </svg>
      </div>
      <div class="wbar">${slider("p", "position", 0, P - 1, 1, 12)}<span class="wread" aria-live="polite"></span></div>
      <p class="wnote">Rows are positions 0–63. Left half: sin on the first 64 even dimensions; right half: cos on the matching odd ones (of 512). Hover to read a value; the line traces the selected row.</p>`;
    const cv = $(root, "canvas"), ctx = cv.getContext("2d");
    const input = $(root, '[data-r="p"]'), out = $(root, '[data-o="p"]'), read = $(root, ".wread"), line = $(root, "polyline");
    const lerp = (a, b, t) => a + (b - a) * t;
    const stops = [[18, 44, 120], [120, 160, 235], [236, 242, 253]];
    const color = (v) => {
      const t = (v + 1) / 2, k = t < 0.5 ? 0 : 1, u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
      return `rgb(${stops[k].map((c, i) => Math.round(lerp(c, stops[k + 1][i], u))).join(",")})`;
    };
    const base = document.createElement("canvas"); base.width = cv.width; base.height = cv.height;
    const bctx = base.getContext("2d");
    for (let p = 0; p < P; p++) for (let c = 0; c < C; c++) { bctx.fillStyle = color(pe(p, dimOf(c))); bctx.fillRect(c * 4, p * 3, 4, 3); }
    const draw = () => {
      const p = +input.value; out.textContent = p;
      ctx.drawImage(base, 0, 0);
      ctx.strokeStyle = "#ffcf3f"; ctx.lineWidth = 1.5; ctx.strokeRect(0.75, p * 3 - 0.5, cv.width - 1.5, 4);
      line.setAttribute("points", Array.from({ length: C }, (_, c) => `${c * 4 + 2},${35 - pe(p, dimOf(c)) * 30}`).join(" "));
    };
    cv.addEventListener("mousemove", (e) => {
      const r = cv.getBoundingClientRect();
      const j = dimOf(Math.min(C - 1, Math.floor(((e.clientX - r.left) / r.width) * C)));
      const p = Math.min(P - 1, Math.floor(((e.clientY - r.top) / r.height) * P));
      read.textContent = `pos ${p}, dim ${j}: ${j % 2 ? "cos" : "sin"} → ${fmt(pe(p, j))}`;
    });
    cv.addEventListener("mouseleave", () => (read.textContent = ""));
    cv.addEventListener("click", (e) => {
      const r = cv.getBoundingClientRect();
      input.value = Math.min(P - 1, Math.floor(((e.clientY - r.top) / r.height) * P)); draw();
    });
    input.addEventListener("input", draw);
    draw();
  };

  /* ---------- MoE: route a token, change k ---------- */
  W.moe = (root) => {
    const tokens = ["the", "cat", "sat", "on", "a", "warm", "mat"];
    const N = 8;
    const probsFor = (tok) => {
      const r = rng(hash(tok)); const z = Array.from({ length: N }, () => gauss(r) * 1.3);
      const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    };
    let tok = "cat", k = 2;
    root.innerHTML = `
      <div class="fig-frame diagram"><div data-svg></div></div>
      <div class="wbar"><div class="chips" role="group" aria-label="Token">${tokens.map((t) => `<button type="button" data-t="${t}" aria-pressed="${t === tok}">${t}</button>`).join("")}</div>${seg("k", [[1, "k = 1"], [2, "k = 2"], [3, "k = 3"]], k, "Experts per token")}</div>
      <p class="wnote" aria-live="polite"></p>`;
    const host = $(root, "[data-svg]"), note = $(root, ".wnote");
    const render = () => {
      const p = probsFor(tok), order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      const on = order.slice(0, k).map((x) => x[1]), mass = on.reduce((s, i) => s + p[i], 0), pmax = order[0][0];
      const y0 = 8, h = 24, g = 6, cy = (i) => y0 + i * (h + g) + h / 2;
      let rows = "", curves = "";
      p.forEach((v, i) => {
        const y = y0 + i * (h + g), isOn = on.includes(i);
        rows += `<path class="dg-line" d="M172 125 C190 125 192 ${cy(i)} 206 ${cy(i)}" style="stroke-width:0.8"/>
          <rect class="dg-bar${isOn ? " on" : ""}" x="210" y="${y + 6}" width="${Math.max(2, (v / pmax) * 96)}" height="${h - 12}" rx="1"/>
          <text class="${isOn ? "dg-ta" : "dg-tm"}" x="314" y="${cy(i) + 4}">${v.toFixed(2)}</text>
          <rect class="dg-box${isOn ? " on" : " soft"}" x="350" y="${y}" width="56" height="${h}" rx="2"/>
          <text class="${isOn ? "dg-ta" : "dg-tm"}" x="378" y="${cy(i) + 4}" text-anchor="middle">E${i + 1}</text>`;
        if (isOn) {
          curves += `<path class="dg-line hl" d="M406 ${cy(i)} C450 ${cy(i)} 455 125 486 125" marker-end="url(#ah-hl)"/>
            <text class="dg-ta" x="412" y="${cy(i) - 6}">× ${(v / mass).toFixed(2)}</text>`;
        }
      });
      host.innerHTML = `<svg viewBox="0 0 590 250" role="img" aria-label="Router probabilities for the token ${esc(tok)}; the top ${k} experts are used">
        <rect class="dg-box" x="0" y="103" width="60" height="44" rx="3"/>
        <text class="dg-op" x="30" y="130" text-anchor="middle">“${esc(tok)}”</text>
        <path class="dg-line" d="M60 125 H86" marker-end="url(#ah)"/>
        <rect class="dg-box" x="88" y="103" width="84" height="44" rx="3"/>
        <text class="dg-t" x="130" y="130" text-anchor="middle">router</text>
        <text class="dg-tm" x="130" y="168" text-anchor="middle">softmax · top-${k}</text>
        ${rows}${curves}
        <circle class="dg-box" cx="504" cy="125" r="16"/>
        <text class="dg-op" x="504" y="131" text-anchor="middle" style="font-style:normal">Σ</text>
        <path class="dg-line" d="M520 125 H550" marker-end="url(#ah)"/>
        <text class="dg-op" x="566" y="131" text-anchor="middle">y</text></svg>`;
      note.textContent = `“${tok}” runs ${on.map((i) => "E" + (i + 1)).join(", ")}: ${k} of ${N} experts, ${Math.round((k / N) * 100)}% of the expert compute. Weights are renormalised over the chosen experts.`;
    };
    $(root, ".chips").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      tok = b.dataset.t; root.querySelectorAll(".chips button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); render();
    });
    onSeg(root, "k", (v) => { k = +v; render(); });
    render();
  };

  /* ---------- toy-gpt: what each character is allowed to see ---------- */
  W.causal = (root, project) => {
    const text = "Which is between I am";
    const chars = [...text];
    const T = chars.length;
    let q = 11;
    const sample = project.figure.sample || [];
    root.innerHTML = `
      <div class="sample">${sample.map((l) => `<p>${esc(l)}</p>`).join("")}</div>
      <p class="wnote">Unedited output from the trained model (toygpt_gita_output.txt).</p>
      <div class="causal">
        <p class="causal-h">Causal mask · hover or tap a character</p>
        <div class="causal-row" role="group" aria-label="Characters">${chars.map((c, i) => `<button type="button" data-i="${i}">${c === " " ? "␣" : esc(c)}</button>`).join("")}</div>
        <div class="causal-grid" aria-hidden="true" style="grid-template-columns: repeat(${T}, 1fr)">${Array.from({ length: T * T }, (_, n) => `<i data-r="${Math.floor(n / T)}" data-c="${n % T}"></i>`).join("")}</div>
        <p class="wnote" aria-live="polite"></p>
      </div>`;
    const btns = [...root.querySelectorAll(".causal-row button")], cells = [...root.querySelectorAll(".causal-grid i")], note = $(root, ".causal .wnote");
    const show = (i) => {
      q = i;
      btns.forEach((b, j) => { b.className = j === i ? "q" : j < i ? "k" : "x"; });
      cells.forEach((c) => {
        const r = +c.dataset.r, col = +c.dataset.c;
        c.className = col > r ? "" : r === i ? "row" : "tri";
      });
      const seen = text.slice(0, i + 1).replace(/ /g, "␣");
      note.textContent = `Predicting what follows “${chars[i] === " " ? "␣" : chars[i]}” (position ${i}), attention can only use the ${i + 1} characters “${seen}”. Future positions are masked to −∞ before the softmax.`;
    };
    $(root, ".causal-row").addEventListener("mouseover", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    $(root, ".causal-row").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(q);
  };

  /* ---------- Flappy / CartPole training curves ---------- */
  W.dqnRuns = (root) => imageTabs(root, [
    { label: "Flappy Bird", img: "assets/img/flappy.webp", w: 1200, h: 400, note: "Flappy Bird: mean reward (left) and ε decay (right). Best episode reward 37.6 after about 57,000 episodes." },
    { label: "CartPole", img: "assets/img/cartpole.webp", w: 1200, h: 400, note: "CartPole: mean reward (left) and ε decay (right). Hit the 100,000-reward stopping point at episode 4,041." },
  ], { label: "Environment" });

  /* ---------- LoRA: change the rank, watch the cost ---------- */
  W.lora = (root) => {
    const layers = [[1000, 784], [2000, 1000], [10, 2000]];
    const FROZEN = 2807010;
    const ranks = [1, 2, 4, 8, 16, 32];
    let r = 1;
    root.innerHTML = `
      <div class="fig-frame diagram"><div data-svg></div></div>
      <div class="wbar">${seg("r", ranks.map((x) => [x, "r = " + x]), r, "Rank")}</div>
      <p class="wnote" aria-live="polite"></p>`;
    const host = $(root, "[data-svg]"), note = $(root, ".wnote");
    let hatch = "";
    for (let i = -170; i < 170; i += 12) hatch += `<line class="dg-hatch" x1="${Math.max(0, i)}" y1="${20 + Math.max(0, -i)}" x2="${Math.min(170, i + 170)}" y2="${20 + Math.min(170, 170 - i)}"/>`;
    const render = () => {
      const w = 8 + 6 * Math.log2(r) + 4;
      const per = (layers[0][0] + layers[0][1]) * r;
      const total = layers.reduce((s, [d, k]) => s + (d + k) * r, 0);
      host.innerHTML = `<svg viewBox="0 0 580 250" role="img" aria-label="Frozen W0 plus trainable B times A at rank ${r}">
        <defs><clipPath id="wclip"><rect x="0" y="20" width="170" height="170"/></clipPath></defs>
        <rect class="dg-frozen" x="0" y="20" width="170" height="170"/>
        <g clip-path="url(#wclip)">${hatch}</g>
        <rect x="58" y="88" width="54" height="34" style="fill:var(--panel)"/>
        <text class="dg-op" x="85" y="111" text-anchor="middle" style="font-size:22px">W<tspan dy="5" style="font-size:14px">0</tspan></text>
        <text class="dg-tm" x="0" y="212">frozen · 1000 × 784</text>
        <text class="dg-tm" x="0" y="228">784,000 weights</text>
        <text class="dg-op" x="196" y="112" text-anchor="middle" style="font-size:24px;font-style:normal">+</text>
        <text class="dg-op" x="228" y="112" text-anchor="middle" style="font-size:15px">α/r</text>
        <rect class="dg-box on" x="252" y="20" width="${w}" height="170"/>
        <text class="dg-op" x="${252 + w / 2}" y="14" text-anchor="middle">B</text>
        <text class="dg-ta" x="${252 + w / 2}" y="212" text-anchor="middle">1000 × ${r}</text>
        <rect class="dg-box on" x="${266 + w}" y="20" width="170" height="${w}"/>
        <text class="dg-op" x="${351 + w}" y="14" text-anchor="middle">A</text>
        <text class="dg-ta" x="${442 + w}" y="${24 + w / 2 + 4}">${r} × 784</text>
        <text class="dg-tm" x="${270 + w}" y="${86 + w}">trainable here  (d + k)·r</text>
        <text class="dg-ta" x="${270 + w}" y="${108 + w}" style="font-size:15px">${int(per)}</text>
        <text class="dg-tm" x="${270 + w}" y="${132 + w}">${((per / 784000) * 100).toFixed(2)}% of this layer</text>
      </svg>`;
      note.textContent = `Rank ${r} across all three layers: ${int(total)} trainable parameters, ${((total / FROZEN) * 100).toFixed(2)}% of the 2.8M frozen ones.${r === 1 ? " This is the rank the notebook trains." : ""}`;
    };
    onSeg(root, "r", (v) => { r = +v; render(); });
    render();
  };

  /* ---------- GAN: first attempt → final ---------- */
  W.ganRuns = (root) => imageTabs(root, [
    { label: "Final", img: "assets/img/gan.webp", w: 856, h: 876, note: "After switching from DCGAN to the fully-connected GAN: recognisable digits from noise." },
    { label: "First attempt", img: "assets/img/gan-first.webp", w: 768, h: 778, note: "The first DCGAN run at epoch 15: the generator collapses onto a few blob-like shapes." },
    { label: "Losses", img: "assets/img/gan-loss.webp", w: 752, h: 558, note: "Generator and discriminator losses from the first run." },
  ], { narrow: true, label: "Run" });

  /* ---------- VAE ---------- */
  W.vaeRuns = (root) => imageTabs(root, [
    { label: "Reconstructions", img: "assets/img/vae.webp", w: 1260, h: 363, note: "Top: test digits. Bottom: the VAE's reconstructions." },
    { label: "Latent space", img: "assets/img/vae-latent.webp", w: 745, h: 590, note: "Test-set latent means projected to 2-D with PCA: the KL term pulls everything into one Gaussian blob." },
  ], { label: "View" });

  /* ---------- reigrad: change the inputs, gradients follow ---------- */
  W.autograd = (root) => {
    root.innerHTML = `
      <div class="fig-frame diagram"><div data-svg></div></div>
      <div class="wbar">${slider("x", "x", -3, 3, 0.1, 2)}${slider("y", "y", -3, 3, 0.1, 3)}</div>
      <p class="wnote">z = x·y + tanh(x). Gradients are what backward() computes: ∂z/∂x = y + (1 − tanh²x), ∂z/∂y = x.</p>`;
    const host = $(root, "[data-svg]");
    const vbox = (x, y, label, val, grad) => `<g>
      <rect class="dg-box" x="${x}" y="${y}" width="112" height="54" rx="3"/>
      <text class="dg-op" x="${x + 12}" y="${y + 22}">${label}</text>
      <text class="dg-tm" x="${x + 100}" y="${y + 21}" text-anchor="end" style="fill:var(--ink)">${val}</text>
      <text class="dg-tm" x="${x + 12}" y="${y + 42}">grad</text>
      <text class="dg-ta" x="${x + 100}" y="${y + 42}" text-anchor="end">${grad}</text></g>`;
    const op = (x, y, s) => `<circle class="dg-box" cx="${x}" cy="${y}" r="16"/><text class="dg-op" x="${x}" y="${y + (s.length > 1 ? 4 : 5)}" text-anchor="middle"${s.length > 1 ? ' style="font-size:11px"' : ""}>${s}</text>`;
    const ix = $(root, '[data-r="x"]'), iy = $(root, '[data-r="y"]');
    const render = () => {
      const x = +ix.value, y = +iy.value, t = Math.tanh(x), m = x * y, z = m + t;
      $(root, '[data-o="x"]').textContent = fmt(x, 1); $(root, '[data-o="y"]').textContent = fmt(y, 1);
      host.innerHTML = `<svg viewBox="0 0 590 250" role="img" aria-label="Computation graph with values and gradients">
        <path class="dg-line" d="M112 47 H170" marker-end="url(#ah)"/>
        <path class="dg-line" d="M202 47 H240" marker-end="url(#ah)"/>
        <path class="dg-line" d="M112 60 C140 60 150 175 172 190" marker-end="url(#ah)"/>
        <path class="dg-line" d="M112 197 H170" marker-end="url(#ah)"/>
        <path class="dg-line" d="M202 197 H240" marker-end="url(#ah)"/>
        <path class="dg-line" d="M352 47 C390 47 395 100 412 113" marker-end="url(#ah)"/>
        <path class="dg-line" d="M352 197 C390 197 395 144 412 131" marker-end="url(#ah)"/>
        <path class="dg-line" d="M444 122 H470" marker-end="url(#ah)"/>
        ${op(186, 47, "tanh")}${op(186, 197, "×")}${op(428, 122, "+")}
        ${vbox(0, 20, "x", fmt(x), fmt(y + (1 - t * t)))}
        ${vbox(0, 170, "y", fmt(y), fmt(x))}
        ${vbox(240, 20, "tanh x", fmt(t), "1.000")}
        ${vbox(240, 170, "x·y", fmt(m), "1.000")}
        ${vbox(472, 95, "z", fmt(z), "1.000")}</svg>`;
    };
    ix.addEventListener("input", render); iy.addEventListener("input", render);
    render();
  };

  /* ---------- Autoencoder: bottleneck size ---------- */
  W.aeRuns = (root) => imageTabs(root, [
    { label: "latent 32", img: "assets/img/autoencoder-32.webp", w: 1600, h: 321, note: "Top: input digits. Bottom: reconstructions through a 32-dimensional latent." },
    { label: "latent 8", img: "assets/img/autoencoder-8.webp", w: 1600, h: 325, note: "Top: input digits. Bottom: reconstructions through an 8-dimensional latent: blurrier, but still legible." },
  ], { label: "Latent size" });

  /* ---------- FrozenLake: Q-learning in the browser ---------- */
  W.lake = (root) => {
    const MAP = "SFFFFHFHFFFHHFFG";
    const N = 4, A = 4, MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // left, down, right, up
    const ARROWS = ["←", "↓", "→", "↑"];
    const GAMMA = 0.9, ALPHA = 0.5;
    let Q, episodes, wins, r, agentAt = null, running = false;
    const reset = () => { Q = Array.from({ length: 16 }, () => [0, 0, 0, 0]); episodes = 0; wins = []; r = rng(42); agentAt = null; };
    const step = (s, a) => {
      const row = Math.floor(s / N), col = s % N;
      const nr = Math.min(N - 1, Math.max(0, row + MOVES[a][0])), nc = Math.min(N - 1, Math.max(0, col + MOVES[a][1]));
      const ns = nr * N + nc, cell = MAP[ns];
      return [ns, cell === "G" ? 1 : 0, cell === "G" || cell === "H"];
    };
    const best = (s) => { const q = Q[s], m = Math.max(...q); const idx = q.map((v, i) => (v === m ? i : -1)).filter((i) => i >= 0); return idx[Math.floor(r() * idx.length)]; };
    const train = (n) => {
      for (let e = 0; e < n; e++) {
        const eps = Math.max(0.05, 1 - episodes / 400);
        let s = 0, done = false, won = 0;
        for (let t = 0; t < 100 && !done; t++) {
          const a = r() < eps ? Math.floor(r() * A) : best(s);
          const [ns, rew, d] = step(s, a);
          const target = rew + (d ? 0 : GAMMA * Math.max(...Q[ns]));
          Q[s][a] += ALPHA * (target - Q[s][a]);
          s = ns; done = d; won = rew;
        }
        episodes++; wins.push(won); if (wins.length > 100) wins.shift();
      }
    };
    root.innerHTML = `
      <div class="lake-wrap"><div class="lake" role="img" aria-label="FrozenLake 4 by 4 grid with the learned policy"></div></div>
      <div class="wbar"><button type="button" class="wbtn" data-act="train">Train 100 episodes</button><button type="button" class="wbtn" data-act="run">Run policy</button><button type="button" class="wbtn ghost" data-act="reset">Reset</button><span class="wread" aria-live="polite"></span></div>
      <p class="wnote">Tabular Q-learning running in your browser with γ = 0.9, the same Bellman target the repo's DQN learns with a network. Arrows show the greedy action; shading is max Q.</p>`;
    const lake = $(root, ".lake"), read = $(root, ".wread");
    const draw = () => {
      lake.innerHTML = [...MAP].map((c, s) => {
        const m = Math.max(...Q[s]), a = Q[s].indexOf(m);
        const label = c === "H" ? "hole" : c === "G" ? "goal" : m > 0 ? ARROWS[a] : c === "S" ? "start" : "·";
        const style = c === "F" || c === "S" ? `style="--q:${Math.min(1, m).toFixed(3)}"` : "";
        return `<div class="cell ${c}" ${style}>${agentAt === s ? '<span class="agent"></span>' : ""}<span class="lbl${label.length > 1 ? " txt" : ""}">${label}</span></div>`;
      }).join("");
      const rate = wins.length ? Math.round((wins.reduce((x, y) => x + y, 0) / wins.length) * 100) : 0;
      read.textContent = `${episodes} episodes · ${rate}% of the last ${wins.length || 100} reached the goal`;
    };
    root.addEventListener("click", (e) => {
      const b = e.target.closest("[data-act]"); if (!b || running) return;
      if (b.dataset.act === "train") { train(100); agentAt = null; draw(); }
      if (b.dataset.act === "reset") { reset(); draw(); }
      if (b.dataset.act === "run") {
        running = true; let s = 0, n = 0; agentAt = s; draw();
        const tick = setInterval(() => {
          const q = Q[s], a = q.indexOf(Math.max(...q));
          const [ns, , d] = step(s, a); s = ns; agentAt = s; n++; draw();
          if (d || n >= 12) { clearInterval(tick); running = false; read.textContent += MAP[s] === "G" ? " · reached the goal" : MAP[s] === "H" ? " · fell in" : " · wandered"; }
        }, 260);
      }
    });
    reset(); draw();
  };

  window.WIDGETS = W;
})();
