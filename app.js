(function () {
  "use strict";

  const SITE = window.SITE;
  const PROJECTS = window.PROJECTS;
  const SNIPPETS = window.SNIPPETS || {};
  const GH = SITE.github;

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pad = (n) => String(n).padStart(2, "0");

  /* ---------- chrome ---------- */
  document.getElementById("back-link").href = SITE.portfolio;
  document.getElementById("back-link").textContent = SITE.name;
  document.getElementById("gh-link").href = GH;
  document.getElementById("foot-gh").href = GH;
  document.getElementById("foot-name").textContent = SITE.name;

  /* ---------- python highlighter ---------- */
  const KW = new Set(("False None True and as assert break class continue def del elif else except finally for from global if " +
    "import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
  const TOKEN = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|([A-Za-z_][A-Za-z0-9_]*)/g;

  function highlight(line) {
    let out = "", last = 0, m, prev = "";
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(line))) {
      out += esc(line.slice(last, m.index));
      const [t, com, str, num, id] = m;
      if (com) out += `<span class="tk-com">${esc(com)}</span>`;
      else if (str) out += `<span class="tk-str">${esc(str)}</span>`;
      else if (num) out += `<span class="tk-num">${esc(num)}</span>`;
      else if (KW.has(id)) out += `<span class="tk-kw">${id}</span>`;
      else if (id === "self") out += `<span class="tk-self">self</span>`;
      else if (prev === "def" || prev === "class") out += `<span class="tk-fn">${id}</span>`;
      else out += esc(id);
      if (id) prev = id;
      last = m.index + t.length;
    }
    return out + esc(line.slice(last));
  }

  function codeBlock(p) {
    const s = SNIPPETS[p.id];
    if (!s) return "";
    const chunks = s.chunks;
    let body = "";
    chunks.forEach((c, ci) => {
      if (ci > 0) body += `<span class="gap" aria-hidden="true">⋯</span>`;
      c.code.split("\n").forEach((ln, i) => {
        const n = c.start ? c.start + i : "";
        body += `<span class="ln" data-n="${n}">${highlight(ln) || " "}</span>`;
      });
    });
    let range = "", href = `${GH}/${s.repo}/blob/main/${s.file}`;
    if (!s.notebook) {
      const first = chunks[0].start;
      const lastC = chunks[chunks.length - 1];
      const end = lastC.start + lastC.code.split("\n").length - 1;
      range = ` · L${first}–${end}`;
      href += `#L${first}-L${end}`;
    } else {
      range = " · notebook";
    }
    return `
      <div class="code${s.notebook ? " nb" : ""}">
        <div class="code-head"><span class="file">${esc(s.file)}${range}</span><a href="${href}" target="_blank" rel="noopener">view source ↗</a></div>
        <pre><code>${body}</code></pre>
      </div>`;
  }

  /* ---------- figure: an interactive widget (widgets.js) ---------- */
  function figure(p) {
    const f = p.figure;
    if (!f) return "";
    return `<figure class="wfig"><div class="widget" data-widget="${f.widget}" data-id="${p.id}"></div>${f.caption ? `<figcaption>${esc(f.caption)}</figcaption>` : ""}</figure>`;
  }

  /* ---------- render ---------- */
  const indexEl = document.getElementById("index-list");
  const projectsEl = document.getElementById("projects");

  indexEl.innerHTML = PROJECTS.map((p, i) => `
    <li><a href="#${p.id}">
      <span class="ix-num">${pad(i + 1)}</span>
      <span class="ix-name">${esc(p.name)}</span>
      <span class="ix-tag">${esc(p.tagline)}</span>
      <span class="ix-year">${p.year}</span>
    </a></li>`).join("");

  projectsEl.innerHTML = PROJECTS.map((p, i) => `
    <section class="proj" id="${p.id}" aria-labelledby="${p.id}-title">
      <div class="wrap">
        <header class="proj-head">
          <span class="proj-num">${pad(i + 1)}</span>
          <div class="proj-title">
            <h2 id="${p.id}-title">${esc(p.name)}</h2>
            <p class="tagline">${esc(p.tagline)}</p>
          </div>
          <div class="proj-links">
            <a class="primary" href="${GH}/${p.repo}" target="_blank" rel="noopener">Code ↗</a>
            ${p.paper ? `<a href="${p.paper.url}" target="_blank" rel="noopener">Paper ↗</a>` : ""}
          </div>
        </header>
        <div class="proj-body">
          <div class="text">
            <p class="summary">${esc(p.summary)}</p>
            <dl class="stats">${p.stats.map(([v, l]) => `<div><dt>${esc(l)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
            <div class="eq" data-tex="${esc(p.equation)}"></div>
            <p class="built-label">What's implemented</p>
            <ul class="built">${p.built.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
          </div>
          <div class="visual">
            ${figure(p)}
            ${codeBlock(p)}
          </div>
        </div>
      </div>
    </section>`).join("");

  /* ---------- widgets ---------- */
  const byId = Object.fromEntries(PROJECTS.map((p) => [p.id, p]));
  document.querySelectorAll("[data-widget]").forEach((el) => {
    const fn = (window.WIDGETS || {})[el.dataset.widget];
    try {
      if (!fn) throw new Error("missing widget " + el.dataset.widget);
      fn(el, byId[el.dataset.id]);
    } catch (e) {
      console.error(e);
      el.innerHTML = `<p class="wnote">This demo couldn't load. The code and results are in the repository.</p>`;
    }
  });

  /* ---------- equations ---------- */
  document.querySelectorAll(".eq").forEach((el) => {
    const t = el.getAttribute("data-tex");
    try {
      window.katex.render(t, el, { displayMode: true, throwOnError: false });
    } catch (e) {
      el.textContent = t;
      el.classList.add("raw");
    }
  });

  /* ---------- deep links: rendered after load, so scroll once content exists ---------- */
  if (location.hash) {
    const t = document.getElementById(location.hash.slice(1));
    if (t) requestAnimationFrame(() => t.scrollIntoView());
  }
})();
