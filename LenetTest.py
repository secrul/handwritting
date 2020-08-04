import torch.nn.parallel
from torch.autograd import Variable
import torch
import model
from mydataset import MyDataset,ToTensor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader



batchsize = 1
epochsize = 500
learningRate = 0.001
print_step = 10


trainPath = r'data/testDigits'

train_dataset = MyDataset(trainPath,transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=True)

modelPath = 'trainedModel/epoch_relu_55.pth'
net = torch.load(modelPath)
acc = 0
s = 0
for iteration, batch in enumerate(train_loader):
    data = Variable(batch['data'].cuda(),requires_grad = True)
    label = Variable(batch['label'].cuda(),requires_grad = False).squeeze(0)
    pred= net(data)
    maxx = torch.max(pred)
    s += 1
    if pred[0][label[0]] == maxx:
        acc += 1
print(acc, s)