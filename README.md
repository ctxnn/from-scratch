# from scratch

A one-page showcase of my from-scratch ML implementations. It's a static site with no build step: plain HTML, CSS and JS.

```
index.html      page shell
styles.css      all styling (light + dark via prefers-color-scheme)
projects.js     the content: one entry per project
snippets.js     verbatim code excerpts, generated from the repos
widgets.js      one interactive figure per project (all computed in the browser)
app.js          renders projects, mounts widgets, code highlighting, KaTeX
assets/img/     figures taken from each repo
tools/gen_snippets.py   regenerates snippets.js from local clones
```

## Preview locally

```bash
python3 -m http.server 4391
```

Then open http://localhost:4391.

## Deploy on GitHub Pages

1. Create a repo (for example `ctxnn/from-scratch`) and push this folder to `main`.
2. In the repo, open **Settings → Pages**. Set **Source: Deploy from a branch** and choose `main` / `/ (root)`.
3. The site goes live at `https://ctxnn.github.io/from-scratch/`.

### Custom subdomain (`fromscratch.<yourdomain>.com`)

1. At your DNS provider, add a **CNAME** record with host `fromscratch` and target `ctxnn.github.io`.
2. In **Settings → Pages → Custom domain**, enter `fromscratch.<yourdomain>.com` and save. GitHub commits a `CNAME` file to the repo.
3. Once the certificate is issued (minutes to about an hour), tick **Enforce HTTPS**.
4. Recommended: verify the domain under **GitHub → Settings → Pages → Verified domains**. This stops anyone else from claiming the subdomain.

## Adding a project

1. Add an entry to `PROJECTS` in `projects.js`. The `id` must match the key used in `snippets.js`.
2. Put its figures in `assets/img/` (WebP, at most about 1600px wide). Give it a widget in `widgets.js` and set `figure: { widget: "name" }`. For a plain image switcher, copy one of the `imageTabs` widgets such as `vaeRuns`.
3. Add an excerpt spec to `SPECS` in `tools/gen_snippets.py`, clone your repos into a folder, then run:

   ```bash
   python3 tools/gen_snippets.py /path/to/clones snippets.js
   ```

4. Bump the `?v=` query on the asset links in `index.html` so browsers fetch the new files.

Also set `SITE.portfolio` in `projects.js` to your main portfolio URL; it's the "Chirag Taneja" link in the top-right corner. The accent colour is the `--accent` variable at the top of `styles.css`.
