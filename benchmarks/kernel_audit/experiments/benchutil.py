import torch


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for i in range(iters):
        flush.zero_()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return ts[len(ts) // 2] * 1e3
