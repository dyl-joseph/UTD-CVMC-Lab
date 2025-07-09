import torch.nn as nn
from representational_network import embedNet, posEnc
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = embedNet().to(device)
        self.pos = posEnc().to(device)
        self.mha = nn.MultiheadAttention()
        self.fc = nn.Linear()
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
    def forward(self,x):
        identity = x
        x = self.embed(x)
        x += self.pos(x)
        query,key = self.mha(x)
        ans = 0
        ans += identity
        ans = self.norm1(ans)

        identity = ans
        ans = self.fc(ans)
        ans+=identity
        ans = self.norm2(ans)

        return ans


class ActRec(nn.Module): # for 1 frame
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention()
        self.fc = nn.Linear()
        self.softmax = nn.Softmax()
    def forward(self,encoder):
        logits = self.mha(encoder)
        logits = self.fc(logits)
        soft_logits = self.softmax(logits) # not needed if cross entropy loss ? 

        return soft_logits

