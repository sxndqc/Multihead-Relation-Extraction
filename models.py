import torch
import numpy as np
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, input_size, output_size, num_heads, dr):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model_size = input_size

        self.depth = int(output_size / self.num_heads) # this is hidden size

        # self.dropq = torch.nn.Dropout(dr)
        # self.dropk = torch.nn.Dropout(dr)

        self.Wh = torch.nn.Linear(input_size, output_size, bias=False)  # all heads are packed into one matrix sequentially
        self.Wt = torch.nn.Linear(input_size, output_size, bias=False)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # BS * SL * NH * H  NH is number of heads which is also num_classes
        return x.permute([0, 2, 1, 3])  # BS * NH * SL * H

    def forward(self, tail, head):  # BS * SL * HS
        batch_size = head.shape[0]

        head = self.Wh(head)  # BS * SL * OUT
        tail = self.Wt(tail)  # BS * SL * OUT

        head = self.split_into_heads(head, batch_size)  # BS * NH * SL * H
        tail = self.split_into_heads(tail, batch_size)  # BS * NH * SL * H

        attn_score = torch.matmul(head, tail.permute(0, 1, 3, 2)) # BS * NH * SL * H times BS * NH * H *SL
        attn_score = attn_score / np.sqrt(head.shape[-1])

        return attn_score

def loss_func(score, labels):

    # entity_mask = torch.zeros(score.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_matrix = torch.zeros(score.shape).to(device)

    batches = torch.LongTensor(range(score.shape[0]))
    classes = labels[:, 0]
    e1s = labels[:, 1]
    e2s = labels[:, 2]

    label_matrix[batches, classes, e1s, e2s] = 1.
    
    output_mask = label_matrix.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
    output_mask = (output_mask > 0).float()

    entity_sum = (output_mask != 0).sum(dim=(2, 3)).float()  # BS, NL

    loss = ((F.binary_cross_entropy_with_logits(score, label_matrix, reduction="none") * output_mask).sum(dim=(2, 3)) / entity_sum).sum()

    return loss

def loss_func_softmax(score, labels):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_matrix = torch.zeros(score.shape).to(device)

    batches = torch.LongTensor(range(score.shape[0]))
    classes = labels[:, 0]
    e1s = labels[:, 1]
    e2s = labels[:, 2]

    label_matrix[batches, classes, e1s, e2s] = 1.
    output_mask = label_matrix.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1).float()

    predict_score = (score*output_mask).sum(dim=(2,3))

    assert (label_matrix.sum(dim=(2,3)).argmax(dim=-1) == classes).all()

    return F.cross_entropy(predict_score, classes)


