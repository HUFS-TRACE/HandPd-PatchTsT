"""
Shared PatchTST — 필기·음성 두 모달이 인코더를 공유하는 분류 모델.

왜 공유 인코더인가
    두 데이터셋에 공통 피험자가 0명이라 late fusion은 "좋아졌다"를 잴 수 없다
    (한 사람의 P필기·P음성이 동시에 있어야 하는데 그런 사람이 없다).
    대신 학습 시점에 가중치를 공유하면, 평가는 여전히 모달 하나씩 하면서
    "모달 B에서 배운 것이 모달 A의 성능을 올리는가"를 A/B로 잴 수 있다.
    짝 데이터가 필요한 순간이 아예 없다.

구조
    필기 [B,  6, 2000] → 필기 stem ┐
                                   ├→ ★공유 인코더★ → 모달별 head → 로짓
    음성 [B, 64,  126] → 음성 stem ┘

    stem   = RevIN · 패칭 · 선형 임베딩 · 위치 인코딩      (모달 전용)
    인코더 = TransformerEncoder N층                        (공유)
    head   = 패치 평균풀링 → 채널 concat → Linear          (모달 전용)

    임베딩을 지나면 둘 다 "d_model 차원 토큰의 수열"이라, 인코더는 입력이
    펜 압력인지 멜 밴드인지 알 방법이 없다. 어텐션은 토큰끼리 내적하고 FFN은
    토큰마다 d_model 차원을 가공할 뿐이며, 수열 길이가 달라도(39 vs 9)
    상관없다. 인코더는 태생적으로 모달을 가리지 않는다.

    단위 문제도 stem의 RevIN이 해결한다 — 두 모달 모두 윈도우 자체 통계로
    정규화되어 헤르츠도 필압도 아닌 무단위 값이 된다.

model.py의 PatchTSTClassifier와 백본 구성(RevIN · sinusoidal PE · pre-LN ·
GELU · channel-independent 패칭)을 그대로 맞췄다. 따라서 같은 d_model·층 수로
돌린 단독 모델과 비교하면 달라지는 것이 "공유 여부" 하나뿐이다.

⚠️ 비교 시 주의
    필기 최적은 d64/L6, 음성 최적은 d128/L2로 서로 다르다. 인코더를 공유하려면
    하나로 통일해야 하는데, 그러면 한쪽은 자기 최적을 못 쓴다. 성능이 떨어졌을 때
    "공유 탓"인지 "설정 탓"인지 섞이지 않으려면, 비교 대상 단독 모델도
    **같은 설정으로 다시 돌려** 기준선을 만들어야 한다.
        ✅ 공유(d64 L6) vs 단독(d64 L6)
        ❌ 공유(d64 L6) vs 단독(d128 L2)
"""
import torch.nn as nn

from model import PositionalEncoding, RevIN


class ModalityStem(nn.Module):
    """모달 전용 입구 — (B, C, T) → (B*C, num_patches, d_model).

    채널을 배치 축에 접어 넣는 channel-independent 방식이라, 채널 수가 다른
    모달(필기 6 · 음성 64)이 같은 인코더를 통과할 수 있다. 인코더가 보는 것은
    "토큰 수열"뿐이고 그 수열이 몇 개 묶여 왔는지는 무관하다.
    """

    def __init__(self, num_channels, seq_len, patch_len, stride, d_model, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.patch_len    = patch_len
        self.stride       = stride
        self.num_patches  = (seq_len - patch_len) // stride + 1

        if self.num_patches < 1:
            raise ValueError(
                f"seq_len({seq_len})이 너무 짧습니다. "
                f"patch_len({patch_len})·stride({stride})를 확인하세요."
            )

        self.revin       = RevIN(num_channels)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc     = PositionalEncoding(d_model, max_len=self.num_patches)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        b, c, _ = x.shape
        x = self.revin(x)                                     # (B, C, T)
        x = x.unfold(dimension=-1,
                     size=self.patch_len,
                     step=self.stride)                        # (B, C, P, patch_len)
        x = x.reshape(b * c, self.num_patches, self.patch_len)
        x = self.patch_embed(x)                               # (B*C, P, d_model)
        x = self.pos_enc(x)
        return self.dropout(x)


class ModalityHead(nn.Module):
    """모달 전용 출구 — (B, C, d_model) → 로짓.

    채널 수가 모달마다 다르므로 입력 차원(C*d_model)도 다르다. head는 어차피
    모달별로 따로 둘 수밖에 없다. 구조는 model.py의 head와 동일하다.
    """

    def __init__(self, num_channels, d_model, num_classes, head_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                                     # (B, C*d_model)
            nn.Dropout(head_dropout),
            nn.Linear(num_channels * d_model, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class SharedPatchTST(nn.Module):
    """모달별 stem·head + 공유 인코더.

    specs: {모달이름: dict(num_channels=, seq_len=, patch_len=, stride=)}
        예) {"hw":    dict(num_channels=6,  seq_len=2000, patch_len=100, stride=50),
             "voice": dict(num_channels=64, seq_len=126,  patch_len=24,  stride=12)}

    forward(x, modality)로 한 번에 한 모달씩 흘린다. 두 모달을 한 텐서에
    담지 않는 이유는 애초에 담을 수 없기 때문이다 — 채널 수도 길이도 다르고,
    같은 사람의 두 모달이 0명이라 채널 축으로 붙일 수도 없다.
    """

    def __init__(
        self,
        specs,
        num_classes=2,
        d_model=64,
        n_heads=4,
        n_layers=6,
        d_ff=128,
        dropout=0.2,
        head_dropout=0.2,
    ):
        super().__init__()
        if not specs:
            raise ValueError("specs가 비어 있습니다. 최소 한 모달이 필요합니다.")

        self.modalities = list(specs)
        self.d_model    = d_model
        self.n_layers   = n_layers

        self.stems = nn.ModuleDict({
            name: ModalityStem(
                num_channels=s["num_channels"], seq_len=s["seq_len"],
                patch_len=s["patch_len"], stride=s["stride"],
                d_model=d_model, dropout=dropout)
            for name, s in specs.items()
        })

        # ★ 공유 구간 — 모달을 가리지 않는 유일한 부분
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-LN (model.py와 동일)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.heads = nn.ModuleDict({
            name: ModalityHead(
                num_channels=s["num_channels"], d_model=d_model,
                num_classes=num_classes, head_dropout=head_dropout)
            for name, s in specs.items()
        })

    def forward(self, x, modality):
        if modality not in self.stems:
            raise KeyError(f"모르는 모달: {modality} (가능: {self.modalities})")
        b, c, _ = x.shape
        z = self.stems[modality](x)          # (B*C, P, d_model)
        z = self.encoder(z)                  # ★ 공유
        z = z.mean(dim=1).reshape(b, c, -1)  # 패치 평균 → (B, C, d_model)
        return self.heads[modality](z)

    def param_report(self):
        """공유/전용 파라미터 수. "얼마나 공유하는가"를 숫자로 보고하기 위한 것.

        공유 비율이 낮으면 "인코더를 공유했다"는 주장 자체가 약해진다.
        """
        shared = sum(p.numel() for p in self.encoder.parameters())
        own = {
            m: sum(p.numel() for p in self.stems[m].parameters())
               + sum(p.numel() for p in self.heads[m].parameters())
            for m in self.modalities
        }
        total = shared + sum(own.values())
        return dict(shared=shared, own=own, total=total,
                    shared_ratio=shared / total if total else 0.0)

    def describe(self):
        r = self.param_report()
        lines = [
            f"SharedPatchTST  d_model={self.d_model} · {self.n_layers}층 "
            f"· 모달 {len(self.modalities)}개",
            f"  공유 인코더        {r['shared']:>9,}  ({r['shared_ratio']*100:.1f}%)",
        ]
        for m in self.modalities:
            stem = self.stems[m]
            lines.append(
                f"  {m:<8} 전용      {r['own'][m]:>9,}   "
                f"(채널 {stem.num_channels} · 패치 {stem.patch_len}/{stem.stride}"
                f" → 토큰 {stem.num_patches}개)"
            )
        lines.append(f"  {'합계':<8}          {r['total']:>9,}")
        return "\n".join(lines)
