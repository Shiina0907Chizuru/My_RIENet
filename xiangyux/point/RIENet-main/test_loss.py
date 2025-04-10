import torch
import torch.nn as nn

P = torch.tensor([[[1.0, 6.0, 3.0],
                   [4.0, 7.0, 3.0],
                   [0.5, 8.0, 1.0]]])
print("P",P.shape)
mins, _ = torch.min(P, 1)
print("mins",mins)
mins, _ = torch.min(P, 2)
print("mins",mins)