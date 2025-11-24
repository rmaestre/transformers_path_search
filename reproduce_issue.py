
import torch
from src.utils_final import MultiheadGQA

def test_dropout_in_eval():
    d_in = 32
    d_out = 32
    context_length = 10
    dropout = 0.5
    num_heads = 4
    kv_heads = 2

    model = MultiheadGQA(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        kv_heads=kv_heads
    )

    model.eval() # Set to evaluation mode

    x = torch.randn(1, 5, d_in)

    # Run forward pass twice with the same input
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    if torch.allclose(out1, out2):
        print("Output is deterministic in eval mode. Dropout is NOT active.")
    else:
        print("Output is NOT deterministic in eval mode. Dropout IS active.")

if __name__ == "__main__":
    test_dropout_in_eval()
