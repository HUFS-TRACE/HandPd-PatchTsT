# -*- coding: utf-8 -*-
"""poster_src.html -> poster.html (그림 base64 인라인 + 화면용 축소 셸)

인쇄는 A1 세로(594×841mm) 원본 크기, 화면에서는 뷰포트에 맞춰 축소해서 보인다.
"""
import base64
import io
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "poster_src.html")
OUT = os.path.join(ROOT, "poster.html")

s = io.open(SRC, encoding="utf-8").read()


def inline(m):
    rel = m.group(1)
    path = os.path.normpath(os.path.join(ROOT, rel))
    if not os.path.exists(path):
        print("  !! 없음:", rel)
        return m.group(0)
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return 'src="data:image/png;base64,%s"' % b64


s = re.sub(r'src="(results/[^"]+)"', inline, s)

# 화면용 축소 셸 — 인쇄에는 영향을 주지 않는다
SHELL_CSS = """
<style>
@media screen{
  body{background:#8a97a0;overflow-x:hidden}
  .stage{width:100%;overflow:hidden}
  .stage .sheet{transform-origin:top center;margin:0 auto}
  .hint{font-family:var(--mono);font-size:12px;letter-spacing:.06em;color:#e8eef2;
    text-align:center;padding:14px 16px 0;line-height:1.6}
  .hint b{color:#fff;font-weight:600}
}
@media print{ .hint{display:none} .stage{overflow:visible} }
</style>
"""

SHELL_JS = """
<script>
(function () {
  var sheet = document.querySelector('.sheet');
  var stage = document.querySelector('.stage');
  if (!sheet || !stage) return;
  function fit() {
    sheet.style.transform = 'none';
    var w = sheet.offsetWidth, h = sheet.offsetHeight;
    var avail = stage.clientWidth - 24;
    var s = Math.min(1, avail / w);
    sheet.style.transform = 'scale(' + s + ')';
    stage.style.height = (h * s + 24) + 'px';
  }
  fit();
  window.addEventListener('resize', fit);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
  window.addEventListener('load', fit);
})();
</script>
"""

s = s.replace("</style>", "</style>" + SHELL_CSS, 1)
s = s.replace('<div class="sheet">',
              '<p class="hint">인쇄용 <b>A1 세로 594 × 841 mm</b> · '
              '브라우저 인쇄에서 배율 100%, 여백 없음, 배경 그래픽 켜기</p>'
              '<div class="stage"><div class="sheet">', 1)
s = s.rstrip()
assert s.endswith("</div>")
s = s + "\n</div>\n" + SHELL_JS

io.open(OUT, "w", encoding="utf-8").write(s)
print("wrote", OUT, "%.1f KB" % (len(s) / 1024.0))
