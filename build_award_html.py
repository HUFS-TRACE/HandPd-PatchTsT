# -*- coding: utf-8 -*-
"""paper/award.md -> award.html (이미지 base64 인라인, 발행용 셸 포함)"""
import base64, io, os, re, sys
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "paper", "award.md")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "award.html")

md = io.open(SRC, encoding="utf-8").read()

title = re.search(r"^# (.+)$", md, re.M).group(1).strip()
subtitle = re.search(r"^### (.+)$", md, re.M).group(1).strip()
md = re.sub(r"\A.*?^---\s*$", "", md, count=1, flags=re.S | re.M)


def inline_img(m):
    alt, rel = m.group(1), m.group(2)
    path = os.path.normpath(os.path.join(ROOT, "paper", rel))
    if not os.path.exists(path):
        return m.group(0)
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return '<img src="data:image/png;base64,%s" alt="%s">' % (b64, alt)


md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", inline_img, md)

body = markdown.markdown(md, extensions=["tables", "sane_lists", "attr_list"])
body = body.replace("<table>", '<div class="tw"><table>')
body = body.replace("</table>", "</table></div>")

secs = re.findall(r"<h2>(\d+)\. ([^<]+)</h2>", body)
body = re.sub(
    r"<h2>(\d+)\. ([^<]+)</h2>",
    lambda m: '<h2 id="s%s"><span class="secno">%s</span>%s</h2>'
    % (m.group(1), m.group(1), m.group(2)),
    body,
)
toc = "\n".join(
    '<a href="#s%s"><span>%s</span>%s</a>' % (n, n, t.split(" — ")[0]) for n, t in secs
)

STATS = [("38", "MFLOPs"), ("7.8", "ms · CPU 1코어"),
         (".851", "창 ROC-AUC"), ("92×", "연산량 절감")]
stats = "\n".join('<div class="stat"><b>%s</b><span>%s</span></div>' % (v, l)
                  for v, l in STATS)

HTML = u"""<title>파킨슨 필기 스크리닝 제출본</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@500;700&family=Noto+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f4f6f7; --surface:#ffffff; --ink:#16242f; --ink-soft:#4d616f;
  --rule:#dde4e9; --rule-soft:#eaeff2; --accent:#1b4965; --accent-soft:#e6eef4;
  --serif:"Noto Serif KR",Georgia,serif;
  --sans:"Noto Sans KR",-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0e1418; --surface:#151d23; --ink:#dbe5ea; --ink-soft:#93a6b2;
  --rule:#26323b; --rule-soft:#1d262d; --accent:#7fb2cf; --accent-soft:#17262f;
}}
:root[data-theme="dark"]{
  --paper:#0e1418; --surface:#151d23; --ink:#dbe5ea; --ink-soft:#93a6b2;
  --rule:#26323b; --rule-soft:#1d262d; --accent:#7fb2cf; --accent-soft:#17262f;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.78;margin:0;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 96px;
  display:grid;grid-template-columns:minmax(0,1fr)}
@media(min-width:1040px){.wrap{grid-template-columns:186px minmax(0,1fr);gap:52px}}

header.masthead{grid-column:1/-1;padding:64px 0 34px;border-bottom:2px solid var(--ink)}
.kicker{font-family:var(--mono);font-size:11.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-soft);margin:0 0 18px}
h1{font-family:var(--serif);font-weight:700;font-size:clamp(30px,4.6vw,50px);
  line-height:1.24;letter-spacing:-.02em;margin:0;text-wrap:balance;max-width:19ch}
.sub{font-family:var(--serif);font-weight:500;font-size:clamp(16px,2vw,21px);
  color:var(--ink-soft);margin:16px 0 0;text-wrap:balance;max-width:38ch}
.stats{display:flex;flex-wrap:wrap;gap:14px;margin-top:34px}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:12px 18px;min-width:118px}
.stat b{display:block;font-family:var(--mono);font-size:23px;font-weight:500;
  color:var(--accent);letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.stat span{display:block;font-size:11.5px;color:var(--ink-soft);margin-top:3px}

nav.toc{display:none}
@media(min-width:1040px){nav.toc{display:block;position:sticky;top:0;align-self:start;
  padding-top:52px;max-height:100vh;overflow-y:auto}}
nav.toc a{display:flex;gap:9px;text-decoration:none;color:var(--ink-soft);
  font-size:12.5px;line-height:1.45;padding:6px 0;border-bottom:1px solid var(--rule-soft)}
nav.toc a span{font-family:var(--mono);color:var(--accent);flex:none;width:13px}
nav.toc a:hover{color:var(--ink)}
nav.toc a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

main{padding-top:44px;min-width:0}
main>p,main>ol,main>hr,main>h3{max-width:68ch}
h2{font-family:var(--serif);font-size:clamp(21px,2.6vw,27px);font-weight:700;
  line-height:1.35;margin:64px 0 20px;padding-top:26px;border-top:1px solid var(--rule);
  display:flex;gap:14px;text-wrap:balance;max-width:68ch}
h2 .secno{font-family:var(--mono);font-size:14px;font-weight:500;color:var(--accent);
  flex:none;padding-top:9px}
h2:first-of-type{margin-top:0;border-top:0;padding-top:0}
h3{font-family:var(--serif);font-size:18px;font-weight:700;margin:38px 0 12px}
p{margin:0 0 17px}
em{font-family:var(--serif)}
code{font-family:var(--mono);font-size:.87em;background:var(--rule-soft);
  padding:1px 5px;border-radius:2px}
ol{padding-left:22px;margin:0 0 18px}
ol li{margin-bottom:11px;padding-left:4px}
ol li::marker{font-family:var(--mono);color:var(--accent);font-size:.9em}
hr{border:0;height:1px;background:var(--rule);margin:44px 0}

blockquote{margin:26px 0;padding:17px 22px;background:var(--accent-soft);
  border-left:3px solid var(--accent);border-radius:0 3px 3px 0;max-width:68ch}
blockquote p{margin:0 0 10px;font-size:15.3px}
blockquote p:last-child{margin:0}

.tw{overflow-x:auto;margin:22px 0 26px;max-width:100%;
  border:1px solid var(--rule);border-radius:3px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14.2px;
  font-variant-numeric:tabular-nums;line-height:1.5}
th,td{padding:10px 15px;text-align:left;border-bottom:1px solid var(--rule-soft);
  vertical-align:top;white-space:nowrap}
th{font-size:11.5px;font-weight:700;letter-spacing:.05em;color:var(--ink-soft);
  background:var(--rule-soft);border-bottom:1px solid var(--rule);white-space:normal}
td:first-child,th:first-child{white-space:normal;min-width:150px}
tr:last-child td{border-bottom:0}
td strong{color:var(--accent)}

img{display:block;width:100%;height:auto;margin:30px 0 12px;
  border:1px solid var(--rule);border-radius:3px;background:#fff}
.figure{max-width:none}
.caption{font-size:13.4px;color:var(--ink-soft);line-height:1.62;
  margin:0 0 34px;padding-left:14px;border-left:2px solid var(--rule);max-width:64ch}
.caption strong{color:var(--ink)}

#refs p{font-size:13.4px;line-height:1.62;color:var(--ink-soft);
  padding-left:30px;text-indent:-30px;margin-bottom:11px;max-width:68ch}
footer{grid-column:1/-1;margin-top:60px;padding-top:22px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11px;letter-spacing:.05em;color:var(--ink-soft)}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
  <header class="masthead">
    <p class="kicker">학술제 제출본 &middot; 2026-09-09 &middot; NewHandPD 61명 &middot; PatchTST</p>
    <h1>__TITLE__</h1>
    <p class="sub">__SUB__</p>
    <div class="stats">__STATS__</div>
  </header>
  <nav class="toc" aria-label="목차">__TOC__</nav>
  <main>__BODY__</main>
  <footer>이우림 &middot; 임시연 &middot; 원종윤 &middot; 박민규 &nbsp;/&nbsp; 근거 파일 results/ &middot; 가이드 09_진행기록.md</footer>
</div>
<script>
document.querySelectorAll('main > p').forEach(function (p) {
  if (p.querySelector(':scope > img')) { p.className = 'figure'; return; }
  var s = p.querySelector(':scope > strong:first-child');
  if (s && /^(그림|표)\s*\d+\./.test(s.textContent)) p.className = 'caption';
});
var hs = document.querySelectorAll('main h2');
var last = hs[hs.length - 1];
if (last && /참고문헌/.test(last.textContent)) {
  var box = document.createElement('div');
  box.id = 'refs';
  last.after(box);
  var n = box.nextElementSibling;
  while (n) { var t = n.nextElementSibling; box.appendChild(n); n = t; }
}
</script>
"""
HTML = (HTML.replace("__TITLE__", title).replace("__SUB__", subtitle)
        .replace("__STATS__", stats).replace("__TOC__", toc).replace("__BODY__", body))
io.open(OUT, "w", encoding="utf-8").write(HTML)
print("wrote", OUT, "%.1f KB" % (len(HTML) / 1024.0))
