import torch, os, json

CKPT = "chess-engine-classic-vs-ml/ml/endgame.ord.pt"
HEADER = "chess-engine-classic-vs-ml/src/net/endgame_net_weights.h"   # the generated C++ header

sd = torch.load(CKPT, map_location="cpu")
# count only the actual network parameters (exclude buffers if you want pure weights;
# include feat_mean/feat_std since they ship in the header and are part of the footprint)
total_params = 0
detail = {}
for k, v in sd.items():
    n = v.numel() if hasattr(v, "numel") else 1
    detail[k] = n
    total_params += n

payload_bytes = total_params * 4   # float32
print("parameter breakdown:")
for k, n in detail.items():
    print(f"  {k:<20} {n:>6}")
print(f"\ntotal parameters     : {total_params:,}")
print(f"payload (float32)    : {payload_bytes:,} bytes  ({payload_bytes/1024:.1f} KiB)")
print(f"payload (float16)    : {total_params*2:,} bytes  ({total_params*2/1024:.1f} KiB)  # if you quantized")

if os.path.exists(HEADER):
    hsize = os.path.getsize(HEADER)
    print(f"generated header .h  : {hsize:,} bytes  ({hsize/1024:.1f} KiB)  # ASCII text, NOT the runtime cost")