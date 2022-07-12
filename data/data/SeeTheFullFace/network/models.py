import torch
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        recurrent = self.dropout(recurrent)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class Res50RNNNet(nn.Module):
    def __init__(self, num_classes, nh=256):
        super(Res50RNNNet, self).__init__()
        self.nh = nh
        self.num_classes = num_classes

        self.ResNet = models.resnet50(num_classes=1000, pretrained=True)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, num_classes, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))
    
    def forward(self, input):

        conv = self.ResNet(input)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(conv)
        output = output.permute(2, 0, 1)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)

        return output



class Res101RNNNet(nn.Module):
    def __init__(self, num_classes, nh=256):
        super(Res101RNNNet, self).__init__()
        self.nh = nh
        self.num_classes = num_classes

        self.ResNet = models.resnet101(num_classes=1000, pretrained=True)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, num_classes, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))
    
    def forward(self, input):

        conv = self.ResNet(input)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(conv)
        output = output.permute(2, 0, 1)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)

        return output



class Res152RNNNet(nn.Module):
    def __init__(self, num_classes, nh=256):
        super(Res152RNNNet, self).__init__()
        self.nh = nh
        self.num_classes = num_classes

        self.ResNet = models.resnet152(num_classes=1000, pretrained=True)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, num_classes, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))
    
    def forward(self, input):

        conv = self.ResNet(input)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(conv)
        output = output.permute(2, 0, 1)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)
        return output

# --------------------------Unet --------------------------

class UnetBlock(nn.Module):
    def __init__(self, input_chan, output_chan, features_chan=None, flag="down"):
        super(UnetBlock, self).__init__()
        self.features_chan = features_chan
        self.type = flag
        if self.type == "down":
            self.conv1 = nn.Conv2d(input_chan, output_chan, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(output_chan)
            self.conv2 = nn.Conv2d(output_chan, output_chan, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.Conv2d(output_chan, output_chan, kernel_size=4, padding=1, stride=2)
            self.relu3 = nn.ReLU(True)
            self.bn3 = nn.BatchNorm2d(output_chan)
        elif self.type =="up":
            self.conv1 = nn.Conv2d(input_chan, int(input_chan/2), kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(int(input_chan/2))
            self.conv2 = nn.Conv2d(int(input_chan/2), output_chan, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.ConvTranspose2d(output_chan, output_chan, kernel_size=4, stride=2, padding=1)
            self.relu3 = nn.ReLU(True)
            self.bn3 = nn.BatchNorm2d(output_chan)
        elif self.type =="last":
            self.conv1 = nn.Conv2d(input_chan, output_chan, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(output_chan)
            self.conv2 = nn.Conv2d(output_chan, output_chan, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            if self.features_chan is not None:
                self.conv3 = nn.Conv2d(output_chan, features_chan, kernel_size=3, padding=1)
                self.relu3 = nn.Tanh()
        elif self.type == "bottom":
            self.conv1 = nn.Conv2d(input_chan, int(input_chan*2), kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(int(input_chan*2))
            self.conv2 = nn.Conv2d(int(input_chan*2), output_chan, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.ConvTranspose2d(output_chan, output_chan, kernel_size=4, stride=2, padding=1)
            self.relu3 = nn.ReLU(True)
            self.bn3 = nn.BatchNorm2d(output_chan)

    def forward(self, x):
        if self.type == "down":
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.bn2(out)
            skip = out
            out = self.conv3(out)
            out = self.relu3(out)
            out = self.bn3(out)
            return out, skip

        elif self.type == "up" or self.type == "bottom":
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.relu3(out)
            out = self.bn3(out)
            return out

        elif self.type == "last":
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.bn2(out)
            if self.features_chan is not None:
                out = self.conv3(out)
                out = self.relu3(out)
            return out


class Unet(nn.Module):
    def __init__(self, feature_chan):
        super(Unet, self).__init__()
        # Down path
        self.l1st_down = UnetBlock(input_chan=3, output_chan=112, flag="down")
        self.l2nd_down = UnetBlock(input_chan=112, output_chan=224, flag="down")
        self.l3rd_down = UnetBlock(input_chan=224, output_chan=448, flag="down")
        # self.l4th_down = UnetBlock(input_chan=256, output_chan=512, flag="down")
        # up path
        self.l1st_up = UnetBlock(input_chan=448, output_chan=448, flag="bottom")
        self.l2nd_up = UnetBlock(input_chan=896, output_chan=224, flag="up")
        self.l3rd_up = UnetBlock(input_chan=448, output_chan=112, flag="up")
        # self.l4th_up = UnetBlock(input_chan=256, output_chan=64, flag="up")
        # last layer
        self.last = UnetBlock(input_chan=224, output_chan=112, features_chan=feature_chan, flag="last")

    def forward(self, x):
        out, skip1 = self.l1st_down(x)
        # skip1 = F.interpolate(skip1, size=512)
        out, skip2 = self.l2nd_down(out)
        # skip2 = F.interpolate(skip2, size=256)
        out, skip3 = self.l3rd_down(out)
        # skip3 = F.interpolate(skip3, size=128)
        # out, skip4 = self.l4th_down(out)
        out = self.l1st_up(out)
        out = torch.cat((out, skip3),1)
        out = self.l2nd_up(out)
        out = torch.cat((out, skip2), 1)
        out = self.l3rd_up(out)
        out = torch.cat((out, skip1), 1)
        # out = self.l4th_up(out)
        # out = torch.cat((out, skip1), 1)
        out = self.last(out)
        return out

class Unet_Res50_RNN(nn.Module):
    def __init__(self, num_classes, nh=256):
        super(Unet_Res50_RNN, self).__init__()
        self.nh = nh
        self.num_classes = num_classes

        self.Unet = Unet(feature_chan=64)

        self.ResNet = models.resnet50(num_classes=1000, pretrained=True)
        self.ResNet.conv1 =  nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, num_classes, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))

    def forward(self, input):
        conv = self.Unet(input)
        conv = self.ResNet(conv)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        output = output.permute(2, 0, 1)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)

        return output

class Unet_Res101_RNN(nn.Module):
    def __init__(self, num_classes, nh=256):
        super(Unet_Res101_RNN, self).__init__()
        self.nh = nh
        self.num_classes = num_classes

        self.Unet = Unet(feature_chan=64)

        self.ResNet = models.resnet50(num_classes=1000, pretrained=True)
        self.ResNet.conv1 =  nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, num_classes, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))

    def forward(self, input):
        conv = self.Unet(input)
        conv = self.ResNet(conv)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        output = output.permute(2, 0, 1)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)

        return output

class Unet_Res152_RNN(nn.Module):
    def __init__(self, num_classes, nh=256):
        super(Unet_Res152_RNN, self).__init__()
        self.nh = nh
        self.num_classes = num_classes

        self.Unet = Unet(feature_chan=64)

        self.ResNet = models.resnet50(num_classes=1000, pretrained=True)
        self.ResNet.conv1 =  nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, num_classes, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))

    def forward(self, input):
        conv = self.Unet(input)
        conv = self.ResNet(conv)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        output = output.permute(2, 0, 1)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)
        return output

class Res50RNNNetExpectArcLoss(nn.Module):
    def __init__(self, out_chan=2048, nh=1024):
        super(Res50RNNNetExpectArcLoss, self).__init__()
        self.nh = nh
        self.out_chan = out_chan

        self.ResNet = models.resnet50(num_classes=1000, pretrained=True)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, out_chan, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))
    
    def forward(self, input):
        conv = self.ResNet(input)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(conv)
        # print(output.shape)
        output = output.permute(2, 0, 1)
        # print(output.shape)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)
        return output

class Res101RNNNetExpectArcLoss(nn.Module):
    def __init__(self, out_chan=2048, nh=1024):
        super(Res101RNNNetExpectArcLoss, self).__init__()
        self.nh = nh
        self.out_chan = out_chan

        self.ResNet = models.resnet101(num_classes=1000, pretrained=True)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, out_chan, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))
    
    def forward(self, input):
        conv = self.ResNet(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        # print(output.shape)
        output = output.permute(2, 0, 1)
        # print(output.shape)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)
        return output

class Res152RNNNetExpectArcLoss(nn.Module):
    def __init__(self, out_chan=2048, nh=1024):
        super(Res152RNNNetExpectArcLoss, self).__init__()
        self.nh = nh
        self.out_chan = out_chan

        self.ResNet = models.resnet152(num_classes=1000, pretrained=True)
        # print(self.ResNet)
        self.ResNet = nn.Sequential(*(list(self.ResNet.children())[:-2]))
        self.ResNet.mp = nn.MaxPool2d((7,1), stride=(1,1)) #2048x1x7
        # print(self.ResNet)
        self.rnn = nn.Sequential(
                    BidirectionalLSTM(2048, nh, nh, 0),
                    BidirectionalLSTM(nh, nh, out_chan, 0)
                    )
        self.to_catigories = nn.Sequential(nn.MaxPool2d((7,1), stride=(1,1)))
    
    def forward(self, input):
        conv = self.ResNet(input)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        # print(output.shape)
        output = output.permute(2, 0, 1)
        # print(output.shape)
        output = self.to_catigories(output)
        output = output.permute(2,1,0)
        output = output.squeeze(1)
        return output
