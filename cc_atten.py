import torch
EPSILON = 1e-10


'''atten strategy'''
def attention_fusion_weight(tensor1, tensor2, kernel_size):
    f_channel = channel_fusion(tensor1, tensor2)
    f_spatial = spatial_fusion(tensor1, tensor2, kernel_size)
    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f

# channel atten
def channel_fusion(tensor1, tensor2):
    # calculate channel attention
    attention_map1 = channel_attention(tensor1)
    attention_map2 = channel_attention(tensor2)
    # get weight map
    attention_p1_w1 = attention_map1 / (attention_map1 + attention_map2 + EPSILON)
    attention_p2_w2 = attention_map2 / (attention_map1 + attention_map2 + EPSILON)

    tensor_f = attention_p1_w1 * tensor1 + attention_p2_w2 * tensor2
    return tensor_f

def channel_attention(tensor):
    B, C, H, W = tensor.size()
    
    query_H = tensor.view(B*C, H, W)# [B*C,H,W]
    query_W = tensor.view(B*C, -1, H)# [B*C,W,H]
    
    key_H = tensor.view(B*C,H,W).permute(0, 2, 1) # [B*C,W,H]
    key_W = tensor.view(B*C,H,W) # [B*C,H,W]

    value_H = tensor.view(B*C,W,H) # [B*C,W,H]
    value_W = tensor.view(B*C,H,W) # [B*C,H,W]

    energy_H = (torch.bmm(query_H, key_H)).view(B*C,H*H) # [B*C,H*H]
    energy_W = torch.bmm(query_W, key_W).view(B*C,W*W)# [B*C,W*W]
    
    energy = torch.cat([energy_H, energy_W], -1)# [B*C,H*H+W*W]
    energy_max = torch.max(energy)
    energy_min = torch.min(energy)
    energy =  (energy - energy_min) / (energy_max - energy_min)

    cc_softmax = torch.nn.Softmax(dim=-1)
    concate = cc_softmax(energy)# [B*C,H*H+W*W]
    
    att_H = concate[:,0:H*H].contiguous().view(-1,H,H)# [B*C,H,H]
    att_W = concate[:,H*H:H*H+W*W].contiguous().view(-1,W,W)# [B*C,W,W]
    
    out_H = torch.bmm(value_H, att_H).view(B,C,W,H).permute(0,1,3,2)
    out_W = torch.bmm(value_W, att_W).view(B,C,H,W)

    # gamma = nn.Parameter(torch.zeros(1))
    gamma = torch.tensor(0.5)
    return gamma*(out_H + out_W) + tensor

# spatial atten
def spatial_fusion(tensor1, tensor2, kernel_size):
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, kernel_size)
    spatial2 = spatial_attention(tensor2, kernel_size)
    # get weight map
    spatial_w1 = spatial1 / (spatial1 + spatial2 + EPSILON)
    spatial_w2 = spatial2 / (spatial1 + spatial2 + EPSILON)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2
    return tensor_f

def spatial_attention(tensor, kernel_size=[8,1]):
    B, C, H, W = tensor.size()
    avg_pooling_H = torch.nn.AvgPool2d((kernel_size[0],kernel_size[1]), stride=(kernel_size[0],kernel_size[1]))
    avg_pooling_W = torch.nn.AvgPool2d((kernel_size[1],kernel_size[0]), stride=(kernel_size[1],kernel_size[0]))
    
    query = tensor
    query_H = query.permute(0, 3, 1, 2).contiguous().view(B, -1, H).permute(0, 2, 1) # [B,H,C*W]
    query_W = query.permute(0, 2 ,1, 3).contiguous().view(B, -1, W).permute(0, 2, 1) # [B,W,C*H]
    
    key_H = avg_pooling_H(tensor) # [B,C,H/8,W]
    key_W = avg_pooling_W(tensor) # [B,C,W/8,H]
    key_size_H = key_H.size()[2]
    key_size_W = key_W.size()[3]
    key_H = key_H.permute(0, 3, 1, 2).contiguous().view(B,-1,key_size_H) # [B,C*W,H/8]
    key_W = key_W.permute(0, 2, 1, 3).contiguous().view(B,-1,key_size_W) # [B,C*W,W/8]

    value_H = avg_pooling_H(tensor) # [B,C,H/8,W]
    value_W = avg_pooling_W(tensor) # [B,C,W/8,H]
    value_size_H = value_H.size()[2]
    value_size_W = value_W.size()[3]
    value_H = value_H.permute(0,3,1,2).contiguous().view(B,-1,value_size_H) # [B,C*W,H/8]
    value_W = value_W.permute(0,2,1,3).contiguous().view(B,-1,value_size_W) # [B,C*W,W/8]

    energy_H = torch.bmm(query_H, key_H).view(B,H*key_size_H)  # [B, H*H/8]
    energy_W = torch.bmm(query_W, key_W).view(B,W*key_size_W)  # [B, W*W/8]
    
    energy = torch.cat([energy_H, energy_W], -1)
    energy_max = torch.max(energy)
    energy_min = torch.min(energy)
    energy =  (energy - energy_min) / (energy_max - energy_min)

    cc_softmax = torch.nn.Softmax(dim=-1)
    concate = cc_softmax(energy)# [B,H*H/8 + W*W/8]
    
    att_H = concate[:,0:H*key_size_H].view(B,H,key_size_H) # [B, H*H/8]
    att_W = concate[:,H*key_size_H:H*key_size_H+W*key_size_W].view(B,W,key_size_W) # [B, W*W/8]
    
    out_H = torch.bmm(value_H, att_H.permute(0, 2, 1)).view(B,W,-1,H).permute(0,2,3,1)
    out_W = torch.bmm(value_W, att_W.permute(0, 2, 1)).view(B,H,-1,W).permute(0,2,1,3)

    # gamma = nn.Parameter(torch.zeros(1))
    gamma = torch.tensor(0.5)

    return gamma*(out_H + out_W) + tensor

'''add srtategy'''
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2) / 2
