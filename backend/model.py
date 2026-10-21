"""Small byte-level causal transformer. Random initialization; no downloaded weights."""
import torch
from torch import nn

CONFIG = dict(vocab_size=256, width=256, layers=6, heads=4, context=256)

class NodeTiny(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CONFIG.copy()
        c = self.config
        self.tokens = nn.Embedding(c['vocab_size'], c['width'])
        self.positions = nn.Embedding(c['context'], c['width'])
        layer = nn.TransformerEncoderLayer(c['width'], c['heads'], c['width'] * 4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, c['layers'], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c['width'])
        self.head = nn.Linear(c['width'], c['vocab_size'], bias=False)
        self.head.weight = self.tokens.weight
        # TransformerEncoder clones the starting layer: initialize independently.
        self.apply(self._init)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        length = ids.shape[1]
        x = self.tokens(ids) + self.positions(torch.arange(length, device=ids.device))
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1)
        return self.head(self.norm(self.blocks(x, mask=mask)))
