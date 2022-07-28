import torch
from x_transformers import TransformerWrapper, Decoder
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='evaluate_policy', filename_suffix="_tzm")
model = TransformerWrapper(
    num_tokens=20000,
    max_seq_len=1024,
    max_mem_len=2048,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=16,
        gate_residual=True
    )
)

x = torch.randint(0, 256, (1, 1024))
print(x)

y = model(x)
writer.add_graph(model, (x,))
print(y)
print(1)
