import math

import torch
import torch.nn as nn


class RevIN(nn.Module):

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # 처음엔 아무 변화 없도록 초기화 (weight=1, bias=0)
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True)          # 시간 축 평균
        std  = (x.var(dim=-1, keepdim=True,
                      unbiased=False) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1)]


class PatchTSTClassifier(nn.Module):


    def __init__(
        self,
        seq_len,
        num_channels,
        num_classes,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
        head_dropout=0.2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patch_len    = patch_len
        self.stride       = stride
        self.num_patches  = (seq_len - patch_len) // stride + 1

        if self.num_patches < 1:
            raise ValueError(
                f"seq_len({seq_len})이 너무 짧아요. "
                f"patch_len({patch_len})과 stride({stride})를 확인하세요."
            )

        # 정규화
        self.revin = RevIN(num_channels)

        # 패치 임베딩: patch_len → d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # 위치 인코딩
        self.pos_enc = PositionalEncoding(d_model,
                                          max_len=self.num_patches)

        # 임베딩 후 dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN (학습 안정성 향상)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 분류 Head
        self.head = nn.Sequential(
            nn.Flatten(),                                      # (B, C*d_model)
            nn.Dropout(head_dropout),
            nn.Linear(num_channels * d_model, num_classes),   # → (B, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape

        # 1. 채널별 정규화
        x = self.revin(x)                              # (B, C, T)

        # 2. 패치 분할: 시간 축을 patch_len 크기로 stride만큼 이동
        x = x.unfold(dimension=-1,
                     size=self.patch_len,
                     step=self.stride)                 # (B, C, num_patches, patch_len)

        # 3. Channel-Independent: 채널과 배치를 합쳐서 독립 처리
        x = x.reshape(b * c,
                       self.num_patches,
                       self.patch_len)                 # (B*C, num_patches, patch_len)

        # 4. 패치 임베딩
        x = self.patch_embed(x)                        # (B*C, num_patches, d_model)

        # 5. 위치 인코딩
        x = self.pos_enc(x)                            # (B*C, num_patches, d_model)

        # 6. Dropout
        x = self.dropout(x)

        # 7. Transformer Encoder
        x = self.encoder(x)                            # (B*C, num_patches, d_model)

        # 8. 패치 평균 풀링
        x = x.mean(dim=1)                              # (B*C, d_model)

        # 9. 채널 차원 복원
        x = x.reshape(b, c, -1)                        # (B, C, d_model)

        # 10. 분류
        return self.head(x)                            # (B, num_classes)
