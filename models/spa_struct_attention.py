import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import misc
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('models')
from inception import InceptionC, InceptionD, InceptionE
__all__ = ['TimeNet','SpaNet']

model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

#mid_channels = 768#1280#2048#1280
def inception_v3(pretrained=False, model_root=None, **kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        misc.load_state_dict(model, model_urls['inception_v3_google'], model_root)
        return model
    return Inception3(**kwargs)

def TimeNet(pretrained=False, model_root=None, **kwargs):
    return MultiLevelAttention(**kwargs)

def SpaNet(pretrained=False, model_root=None, model_type='str', **kwargs):
  if model_type is 'str':
    return SpaStrNet(**kwargs)
  elif model_type is 'old':
    return OldAttNet(**kwargs)
  else :
    raise ValueError
#    return GloLocNet(**kwargs)
#    return TopAttention2(**kwargs)

class SpaStrNet(nn.Module):
    def __init__(self, glo_channels, loc_channels, out_channels, num_classes=101, aMid=0.001, transform_input=False, drop_rate=0.5, out_type='glo', n=5):
        super(SpaStrNet, self).__init__()
        self.strnet=StrAttention(glo_channels, loc_channels, 1)
        self.fc = nn.Sequential(OrderedDict([('fc', nn.Linear(glo_channels+loc_channels, num_classes))]))
        self.n = n
        self.comLoss = comLoss(glo_channels, loc_channels, num_classes)
    def forward(self, x_input):
        x = x_input[0]
        x_list = torch.split(x,1,dim=0)
        x = torch.cat([a.squeeze(dim=0) for a in x_list],dim=0)
        obj, stu = self.strnet(x)
        obj = torch.split(obj,self.n,dim=0) if self.training else torch.split(obj,25,dim=0)
        obj = torch.stack(obj,dim=0)
        obj = obj.transpose(1,2).squeeze(-1).mean(-1)
        stu = torch.split(stu,self.n,dim=0) if self.training else torch.split(stu,25,dim=0)
        stu = torch.stack(stu,dim=0)
        stu = stu.transpose(1,2).squeeze(-1).mean(-1)
        com_loss = self.comLoss(obj, stu, x_input[-1])
        x = torch.cat([obj,stu], dim=1)
        x = self.fc(x)
        output = x
        if self.training:
            return output, com_loss
        else:
            return output

class comLoss(nn.Module):
    def __init__(self, glo_dim, loc_dim, num_classes, drop_rate=0.5, **kwargs):
        super(comLoss, self).__init__()
        self.drop_rate = drop_rate
        self.fc_glo = nn.Sequential(OrderedDict([('glo_fc', nn.Linear(glo_dim, num_classes))]))
        self.fc_loc = nn.Sequential(OrderedDict([('loc_fc', nn.Linear(loc_dim, num_classes))]))
    def forward(self, glo, loc, one_hot):
        glo = F.dropout(glo, p=self.drop_rate, training=self.training)
        x_glo = self.fc_glo(glo)
        score_glo = F.softmax(x_glo, dim=-1)
        x_loc = self.fc_loc(loc)
        score_loc = F.softmax(x_loc, dim=-1)
        if  self.training:
            loss_w = 1-torch.mul(score_glo, one_hot).sum(-1)
            loss = torch.mul(-one_hot,torch.log(score_loc)).sum(-1)#self.cc(x_loc, label)
            loss_com = torch.mul(loss_w, loss).mean()
        else:
            loss_com = 0
        return loss_com

class OldAttNet(nn.Module):
    def __init__(self, glo_channels, loc_channels, out_channels, num_classes=101, aMid=0.001, transform_input=False, drop_rate=0.5, out_type='glo', n=5):
        super(OldAttNet, self).__init__()
        self.Attention = SpaAttention(glo_channels, loc_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(OrderedDict([('fc', nn.Linear(glo_channels, num_classes))]))
        self.n = n
    def forward(self, x_input):
        x = x_input[0]
        x_list = torch.split(x,1,dim=0)
        x = torch.cat([a.squeeze(dim=0) for a in x_list],dim=0)
        obj, _ = self.Attention(x)
        obj = torch.split(obj,self.n,dim=0) if self.training else torch.split(obj,25,dim=0)
        obj = torch.stack(obj,dim=0)
        obj = obj.transpose(1,2).squeeze(-1).mean(-1)
        x = self.fc(obj)
        output = x
        if self.training:
            return output
        else:
            return output

class GloLocNet(nn.Module):
    def __init__(self, glo_channels, loc_channels, out_channels, num_classes=101, aMid=0.001, transform_input=False, drop_rate=0.5, out_type='glo', n=5):
        super(GloLocNet, self).__init__()
        self.Attention = SpaAttention(glo_channels, glo_channels, 1, kernel_size=1)
        self.LocAttention = SpaAttention(loc_channels, loc_channels, 1, kernel_size=1)
        self.resize = torch.nn.Upsample(size=(17,17))
        self.pool = nn.AvgPool2d(8)
        self.fc_glo = nn.Sequential(OrderedDict([('glo_dc', nn.Linear(glo_channels, num_classes))]))
        self.fc_loc = nn.Sequential(OrderedDict([('loc_fc', nn.Linear(loc_channels, num_classes))]))
        self.fc_cat = nn.Sequential(OrderedDict([('cat_fc', nn.Linear(loc_channels+glo_channels, num_classes))]))
        self.drop_rate = drop_rate
        self.cc = nn.CrossEntropyLoss(reduce=False)
        self.type = out_type
        self.Mixed = InceptionE(1280)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.n = n
    def forward(self, x):
        # global attention
        glo = x[0]
        glo_list = torch.split(glo,1,dim=0)
        glo_x = torch.cat([a.squeeze(dim=0) for a in glo_list],dim=0)
        #glo_x = self.Mixed_7a(loc_x)
        #glo_x = self.Mixed_7b(glo_x)
        glo,glo_w = self.GloAttention(glo_x)
        glo = torch.split(glo,self.n,dim=0) if self.training else torch.split(glo,25,dim=0)
        glo = torch.stack(glo,dim=0)
        glo = glo.transpose(1,2).squeeze(-1).mean(-1)
        # local attention
        loc = x[1]
        loc_list = torch.split(loc,1,dim=0)
        loc_x = torch.cat([a.squeeze(dim=0) for a in loc_list],dim=0)
        loc,loc_w = self.LocAttention(loc_x)
        loc = torch.split(loc,self.n,dim=0) if self.training else torch.split(loc,25,dim=0)
        loc = torch.stack(loc,dim=0)
        loc = loc.transpose(1,2).squeeze(-1).mean(-1)
        # top
        #top = x[2]
        #top = top.squeeze().transpose(1,2).mean(-1)
        # inception modules
        #glo = self.Mixed(glo)
        # comput gui loss
        #   The loss consider that the local attention map should focus on the regions which global attention focus on.
        #   simple to minimize sum(P(~Mglobal)*P(Mlocal))
        #   need to renice
        

        # attention guided loss
        glo_w = glo_w.view(glo_x.size(0), glo_x.size(2), glo_x.size(3))
        glo_w = F.upsample(glo_w.unsqueeze(1), size=(17,17), mode='bilinear').squeeze(1).view(glo_w.size(0),-1)
        glo_w_max = torch.max(glo_w, dim=-1, keepdim=True)[0]
        glo_w_max = glo_w_max.repeat(1,glo_w.size(1))
        glo_w_ = torch.div(glo_w_max - glo_w, glo_w_max)
        loc_w = torch.mul(loc_w, glo_w)
        loc_w_max = torch.max(loc_w, dim=-1, keepdim=True)[0]
        loc_w_max = loc_w_max.repeat(1,loc_w.size(1))
        loc_w = torch.div(loc_w, loc_w_max)
        loss_gui = torch.mul(glo_w_, loc_w).mean().mean()
        
        # comput complemented loss for global and local deep feat
        one_hot = x[-1]
        #glo = F.dropout(glo, p=self.drop_rate, training=self.training)
        #glo = torch.cat([glo, top],dim=1)
        x_glo = self.fc_glo(glo)
        #x_loc = self.fc_loc(loc)
        score_glo = F.softmax(x_glo, dim=-1)
        #score_loc = F.softmax(x_loc, dim=-1)
        #if 1 or self.training:
        #    loss_w = 1-torch.mul(score_glo, one_hot).sum(-1)
        #    loss = torch.mul(-one_hot,torch.log(score_loc)).sum(-1)#self.cc(x_loc, label)
        #    loss_com = torch.mul(loss_w, loss).mean()
        #else:
        #if self.type == 'all':
        #    score = score_glo + 0.2 * score_loc
        #elif self.type == 'glo':
        #    score = score_glo
        #elif self.type == 'loc':
        #    score = score_loc
#        score = score_glo #+ 0.2 * score_loc#torch.mul(loss_w.unsqueeze(1).repeat(1,x_glo.size(1)), score_loc)
        score = score_glo #+ 0.1*score_loc#torch.mul(loss_w.unsqueeze(1).repeat(1,x_glo.size(1)), score_loc)
        # concat
        '''
        x = torch.cat([glo,loc],dim=1)
        # classification
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc_cat(x)
        '''
        loss = 0
        output = torch.cat([glo,loc],dim=1)
        output = self.fc_cat(output)
        if self.type == 'loc':
          output = x_loc
        if self.training:
            return output, score, 0.1*loss_gui
        else:
            return output, score, loss


class TopAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_classes=101, aMid=0.001, transform_input=False, drop_rate=0.5):
        super(TopAttention, self).__init__()
        self.TempNet = TemporalAttention(in_channels, mid_channels, 1,kernel_size=1)
        self.gloAttention = SpaAttention(in_channels, mid_channels, 1,kernel_size=1)
        self.pool = nn.AvgPool2d(8)
        self.group1 = nn.Sequential(
            OrderedDict([('fc', nn.Linear(in_channels, num_classes))])
        )
        self.drop_rate = drop_rate
    def forward(self, x):
        # spatial attention
        x_list = torch.split(x,1,dim=0)
        x = torch.cat([a.squeeze(dim=0) for a in x_list],dim=0)
        x,_ = self.gloAttention(x)
        #x = x.mean(-1).mean(-1)
        x = x.view(x.size(0), -1)
        if self.training:x = torch.split(x,5,dim=0)
        else:x = torch.split(x,25,dim=0)
        x = torch.stack(x,dim=0)
        #x = self.TempNet(x.transpose(1,2)).squeeze(-1)
        x = x.transpose(1,2).squeeze(-1).mean(-1)
        # classification
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.group1(x)
        return x
class TopAttention2(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=101, aMid=0.001, transform_input=False, drop_rate=0.5):
        super(TopAttention2, self).__init__()
        self.gloAttention = SpaAttention(in_channels, 1,kernel_size=1)
        self.locAttention = SpaAttention(in_channels, 1,kernel_size=1)
        self.resize = torch.nn.Upsample(size=(17,17))
        self.group1 = nn.Sequential(
            OrderedDict([('fc', nn.Linear(in_channels, num_classes))])
        )
        self.drop_rate = drop_rate
    def forward(self, x):
        x_glo,w = self.gloAttention(x[0])
        w = self.resize(w)
        x_loc,_ = self.locAttention(x[1],w)
        x = torch.cat([x_glo, x_loc],dim=1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.group1(x)
        return x

class StrAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, withW=False, **kwargs):
        super(StrAttention, self).__init__()
        self.conv1 = Conv2d_tanh(in_channels, mid_channels, kernel_size=1, **kwargs)
        self.conv2 = Conv2d_softmax(mid_channels, 1, kernel_size=1, **kwargs)
        self.str1_1 = Conv2d_relu(in_channels, mid_channels, kernel_size=3, padding=1, **kwargs)
        self.str1_2 = Conv2d_relu(in_channels, mid_channels, kernel_size=5, padding=2, **kwargs)
        self.str2_1 = Conv2d_relu(2*mid_channels, mid_channels, kernel_size=3, padding=1, **kwargs)
    def forward(self, input_x):
        x = self.conv1(input_x)
        AttentionWs = self.conv2(x)
#  structure information part
        a1 = self.str1_1(input_x)
        a2 = self.str1_2(input_x)
        str1 = torch.cat([a1, a2], dim=1)
        str2 = self.str2_1(str1)
#        x = torch.cat([x, str2], dim=1)
        obj = torch.bmm( input_x.view(input_x.size(0),input_x.size(1),-1) , AttentionWs.unsqueeze(-1) ).squeeze(-1)
        stu = torch.bmm( str2.view(str2.size(0),str2.size(1),-1) , AttentionWs.unsqueeze(-1) ).squeeze(-1)
        return obj, stu
class SpaAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, withW=False, **kwargs):
        super(SpaAttention, self).__init__()
        self.conv1 = Conv2d_tanh(in_channels, mid_channels, **kwargs)
        self.conv2 = Conv2d_softmax(mid_channels, 1, **kwargs)
    def forward(self, x):
        x1 = self.conv1(x)
        AttentionWs = self.conv2(x1)
        outputs = torch.bmm( x.view(x.size(0),x.size(1),-1) , AttentionWs.unsqueeze(-1) )
        return outputs.squeeze(-1), AttentionWs
class SpaAttention_withW(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, withW=False, **kwargs):
        super(SpaAttention_withW, self).__init__()
        self.conv1 = Conv2d_tanh(in_channels, mid_channels, **kwargs)
        self.conv2 = Conv2d_softmax(mid_channels, 1, **kwargs)
        self.resize = torch.nn.Upsample(size=(17,17))
    def forward(self, x, w):
        w = self.resize(w)
        AttentionWs = self.conv2(self.conv1(x))
        AttentionWs = AttentionWs.mul(w)
        outputs = torch.bmm( x.view(x.size(0),x.size(1),-1) , AttentionWs.unsqueeze(-1) )
        return outputs, AttentionWs
class MultiLevelAttention(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, num_classes=101, aMid=0.001, transform_input=False, drop_rate=0.5, out_type='time'):
        super(MultiLevelAttention, self).__init__()
        self.ShortTermNet = TemporalAttention_withCenLoss(in_channels, mid_channels, 1, kernel_size=1)
        self.MidTermNet = TemporalAttention(in_channels, out_channels, 1, kernel_size=1, withW=True)
        self.LongTermNet = LongTermAttention(in_channels, mid_channels, out_channels, kernel_size=1)
        self.TimeNet = TemporalAttention(in_channels, 1,kernel_size=1)
        self.group1 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(in_channels, num_classes))
            ])
        )
        self.drop_rate = drop_rate
        self.loss = torch.zeros([1,1])
        self.aMid = aMid
        self.type = out_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        # Short Term Net
        xShort, wShort, lossShort = self.ShortTermNet(x[0])#torch.Size([32, 2048, 5]) torch.Size([32, 5, 1])
        # Mid Term Net
        xMid, wMid = self.MidTermNet(x[1])
        w_diff = (wMid-wShort).squeeze(-1)
        lossMid = w_diff.mul(w_diff).mean(-1).mean(-1)
        # Long Term Net
        xLong, lossLong = self.LongTermNet(x[1])
        # Time Net
        x = torch.cat([xShort, xMid, xLong],-1)
        x = self.TimeNet(x)
        # Other outputs
        #x = x.sum(-1).squeeze(-1)
        if   self.type=='short': x = xShort
        elif self.type=='mid' : x = xMid
        elif self.type=='long': x = xLong
        elif self.type=='short+mid':x = (xShort+xMid)/2
        elif self.type=='short+mid+long': x = (xShort+xMid+xLong)/3
        # Classification with fc
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.group1(x)
        #return x, None, None,None
        return x, lossShort, lossMid, lossLong

class LongTermAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(LongTermAttention, self).__init__()
        self.hidden_dim = out_channels
        self.batch_size = 1#batch_size
        self.rnn = nn.LSTM(in_channels, self.hidden_dim, batch_first=True)
#        self.rnn = nn.GRU(in_channels, self.hidden_dim, batch_first=True)
        self.Attention = TemporalAttention(self.hidden_dim, out_channels,withW=True,**kwargs)
        self.loss = torch.zeros([1,1])
        self.Long_a = None
    def forward(self, x):
        if self.Long_a is None:
          a = np.arange(0,1,1/x.size(2))
          a = a.max()-a
          self.Long_a = (a/a.sum()).astype('float32')[:,np.newaxis]
        emb = x.transpose(2,1)#.transpose(1,0)
        out, self.hidden_out = self.rnn(emb)
        out, w = self.Attention(out.transpose(1,2))
        if self.training:
          a = Variable(torch.from_numpy(self.Long_a).cuda(),requires_grad=False)
          loss = torch.mm(w.squeeze(-1), a).squeeze(-1)#torch.from_numpy(a/a.sum()).unsqueeze(-1))
        else:
          loss = self.loss
        #out = out.transpose(1,2).sum(-1)
        #loss = self.loss
        return out, loss.mean(-1)

class TemporalAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels=1, withW=False, **kwargs):
        super(TemporalAttention, self).__init__()
        self.conv1 = Conv1d_tanh(in_channels, mid_channels, **kwargs)
        self.conv2 = Conv1d_softmax(mid_channels, 1, **kwargs)
        self.withW = withW
    def forward(self, x):
        AttentionWs = self.conv2(self.conv1(x)).transpose(2,1)
        outputs = torch.bmm( x , AttentionWs )
        if self.withW:
            return outputs, AttentionWs
        else:
            return outputs


class TemporalAttention_withCenLoss(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(TemporalAttention_withCenLoss, self).__init__()
        self.conv1 = Conv1d_tanh(in_channels, mid_channels, **kwargs)
        self.conv2 = Conv1d_softmax(mid_channels, 1, **kwargs)
        self.loss = torch.zeros([1,1])
    def forward(self, x):
        AttentionWs = self.conv2(self.conv1(x)).transpose(2,1)
        outputs = torch.bmm( x , AttentionWs )
        y = outputs.repeat(1,1,x.size(-1))
        z = (x-y)
        z = z.mul(z)
        z = torch.bmm(z, AttentionWs)
        loss = z.squeeze(-1).mean(dim=-1) if self.training else self.loss
        return outputs, AttentionWs, loss.mean(-1)

class Conv1d_tanh(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv1d_tanh, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv1d(in_channels, out_channels, bias=True, **kwargs)),
#                ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )
    def forward(self, x):
        x = self.group1(x)
        return F.tanh(x)

class Conv1d_sigmoid(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv1d_sigmoid, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)),
                ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )
    def forward(self, x):
        x = self.group1(x)
        return F.sigmoid(x)

class Conv1d_softmax(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv1d_softmax, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv1d(in_channels, out_channels, bias=False, **kwargs))#,                ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )
    def forward(self, x):
        x = self.group1(x)
        return F.softmax(x,dim=-1)#.squeeze(-2), dim=1)


class Conv2d_relu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2d_relu, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)),
                ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return F.relu(x)
    
class Conv2d_tanh(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2d_tanh, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)),
          #      ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return F.tanh(x)
class Conv2d_softmax(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2d_softmax, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)),
         #       ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )
    def forward(self, x):
        x = self.group1(x)
        w = x.view(x.size(0),-1)
        w = F.softmax(w, dim=-1)#.view(x.size(0),x.size(2),x.size(3))
        return w
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.group1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)),
                ('bn', nn.BatchNorm2d(out_channels, eps=0.001))
            ])
        )
    def forward(self, x):
        x = self.group1(x)
        return F.relu(x, inplace=True)
