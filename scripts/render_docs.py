"""Render RESEARCH.md → RESEARCH.html with syntax highlighting and sidebar nav."""
import re
import markdown
from markdown.extensions.toc import slugify_unicode
from pathlib import Path

SRC = Path("RESEARCH.md")
DST = Path("RESEARCH.html")

md_text = SRC.read_text()

md = markdown.Markdown(
    extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list", "md_in_html"],
    extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": False},
        "toc": {"permalink": True, "toc_depth": "2-3", "slugify": slugify_unicode},
    },
)
body = md.convert(md_text)
toc  = md.toc  # generated sidebar HTML

HTML = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BassEmulatorVST — Research</title>
<style>
/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:       #0f1117;
  --surface:  #1a1d27;
  --border:   #2e3147;
  --accent:   #7c6af7;
  --accent2:  #4fc3f7;
  --text:     #e2e4f0;
  --muted:    #7a7f9a;
  --code-bg:  #13151f;
  --tag-bg:   #1e2235;
  --nav-w:    280px;
}}
html {{ scroll-behavior: smooth; }}
body {{
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  font-size: 15px;
}}

/* ── Layout ── */
.layout {{ display: flex; min-height: 100vh; }}

nav {{
  width: var(--nav-w);
  min-width: var(--nav-w);
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 32px 20px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}}
nav .logo {{
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 6px;
}}
nav .subtitle {{
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 28px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
}}
nav ul {{ list-style: none; }}
nav li {{ margin-bottom: 2px; }}
nav a {{
  display: block;
  padding: 5px 10px;
  border-radius: 6px;
  color: var(--muted);
  text-decoration: none;
  font-size: 13px;
  transition: all .15s;
}}
nav a:hover {{ background: var(--tag-bg); color: var(--text); }}
nav ul ul a {{ padding-left: 22px; font-size: 12px; }}
nav ul ul ul a {{ padding-left: 34px; }}

main {{
  flex: 1;
  max-width: 860px;
  padding: 56px 56px 100px;
  min-width: 0;
}}

/* ── Hero ── */
.hero {{
  background: linear-gradient(135deg, #1a1d27 0%, #141828 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 40px 44px;
  margin-bottom: 52px;
  position: relative;
  overflow: hidden;
}}
.hero::before {{
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 80% 50%, rgba(124,106,247,.08) 0%, transparent 70%);
  pointer-events: none;
}}
.hero h1 {{
  font-size: 26px;
  font-weight: 700;
  background: linear-gradient(135deg, #fff 0%, var(--accent) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 10px;
}}
.hero .meta {{
  color: var(--muted);
  font-size: 13px;
  display: flex;
  gap: 24px;
}}
.hero .meta span::before {{ content: '◆ '; color: var(--accent); font-size: 10px; }}

/* ── Typography ── */
h2 {{
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  margin: 56px 0 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}}
h2::before {{
  content: '';
  display: block;
  width: 3px;
  height: 20px;
  background: var(--accent);
  border-radius: 2px;
}}
h3 {{
  font-size: 16px;
  font-weight: 600;
  color: var(--accent2);
  margin: 32px 0 12px;
}}
h4 {{
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  margin: 24px 0 8px;
  text-transform: uppercase;
  letter-spacing: .06em;
}}
p {{ margin-bottom: 14px; }}

/* ── Details / Summary (collapsible) ── */
details {{
  background: var(--tag-bg);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
  margin: 16px 0 24px;
  padding: 0;
}}
details[open] summary {{ border-bottom: 1px solid var(--border); }}
summary {{
  padding: 12px 18px;
  cursor: pointer;
  font-weight: 600;
  font-size: 13.5px;
  color: var(--accent2);
  list-style: none;
  display: flex;
  align-items: center;
  gap: 10px;
  user-select: none;
}}
summary::-webkit-details-marker {{ display: none; }}
summary::before {{
  content: '▶';
  font-size: 10px;
  color: var(--accent);
  transition: transform .2s;
  flex-shrink: 0;
}}
details[open] summary::before {{ transform: rotate(90deg); }}
details > *:not(summary) {{
  padding: 16px 20px;
}}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
strong {{ color: #fff; font-weight: 600; }}
em {{ color: var(--accent2); font-style: normal; font-weight: 500; }}

/* ── Code ── */
code {{
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 7px;
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 13px;
  color: #a6b0cf;
}}
pre {{
  background: var(--code-bg) !important;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 24px;
  overflow-x: auto;
  margin: 16px 0 24px;
  font-size: 13px;
  line-height: 1.6;
}}
pre code {{
  background: none;
  border: none;
  padding: 0;
  font-size: inherit;
  color: #a6b0cf;
}}

/* Pygments dark theme overrides */
.highlight .hll {{ background-color: #1e2235 }}
.highlight  {{ background: transparent }}
.highlight .c  {{ color: #6a6f8a; font-style: italic }}
.highlight .k  {{ color: #c792ea }}
.highlight .n  {{ color: #a6b0cf }}
.highlight .o  {{ color: #89ddff }}
.highlight .s  {{ color: #c3e88d }}
.highlight .m  {{ color: #f78c6c }}
.highlight .nb {{ color: #82aaff }}
.highlight .nf {{ color: #82aaff }}
.highlight .nn {{ color: var(--accent2) }}
.highlight .nt {{ color: #f07178 }}
.highlight .na {{ color: #ffcb6b }}
.highlight .mi {{ color: #f78c6c }}
.highlight .kn {{ color: #c792ea }}
.highlight .kd {{ color: #c792ea }}
.highlight .kr {{ color: #c792ea }}
.highlight .kt {{ color: var(--accent2) }}
.highlight .cm {{ color: #6a6f8a; font-style: italic }}
.highlight .cp {{ color: #6a6f8a }}
.highlight .c1 {{ color: #6a6f8a; font-style: italic }}
.highlight .cs {{ color: #6a6f8a; font-style: italic }}

/* ── Tables ── */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0 28px;
  font-size: 13.5px;
}}
thead tr {{
  background: var(--tag-bg);
  border-bottom: 2px solid var(--accent);
}}
th {{
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  color: var(--accent2);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .06em;
}}
td {{
  padding: 9px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
tr:hover td {{ background: rgba(124,106,247,.05); }}
td code {{ font-size: 12px; }}

/* ── Approach map ── */
.approach-map pre {{
  border-left: 3px solid var(--accent) !important;
  border-radius: 0 8px 8px 0 !important;
  background: var(--tag-bg) !important;
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--text);
}}
.approach-map pre code {{ color: var(--text); }}

/* ── Blockquote ── */
blockquote {{
  border-left: 3px solid var(--accent);
  background: var(--tag-bg);
  margin: 16px 0;
  padding: 14px 20px;
  border-radius: 0 8px 8px 0;
  font-style: italic;
  color: #b0b8d0;
}}

/* ── Lists ── */
ul, ol {{
  padding-left: 24px;
  margin-bottom: 14px;
}}
li {{ margin-bottom: 5px; }}
li > ul {{ margin-top: 5px; margin-bottom: 5px; }}

/* ── Horizontal rule ── */
hr {{ border: none; border-top: 1px solid var(--border); margin: 40px 0; }}

/* ── Star badge ── */
.toc a[href*="pesto"], .toc a[href*="swift"] {{ color: var(--accent); }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}

/* ── Responsive ── */
@media (max-width: 900px) {{
  nav {{ display: none; }}
  main {{ padding: 32px 24px 60px; }}
}}
</style>
</head>
<body>
<div class="layout">

<nav>
  <div class="logo">BassEmulatorVST</div>
  <div class="subtitle">Research Document · May 2026</div>
  {toc}
</nav>

<main>
  <div class="hero">
    <h1>Эмуляция баса из гитарного сигнала</h1>
    <div class="meta">
      <span>WebSearch · May 2026</span>
      <span>13 разделов</span>
      <span>Источники верифицированы</span>
    </div>
  </div>

  {body}
</main>

</div>
</body>
</html>
"""

DST.write_text(HTML)
print(f"✓ {DST} ({DST.stat().st_size // 1024} KB)")
