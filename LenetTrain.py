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
print_step = 50





trainPath = r'data/trainingDigits'

train_dataset = MyDataset(trainPath,transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=True)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
net = model.LeNet().cuda()
criteron = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def train(epoch):
    net.train()
    epoch_loss = 0
    scheduler.step()
    for iteration, batch in enumerate(train_loader):
        data = Variable(batch['data'].cuda(),requires_grad = True)
        label = Variable(batch['label'].cuda(),requires_grad = False).squeeze(0)
        pred= net(data)
        optimizer.zero_grad()
        loss = criteron(pred, label)
        epoch_loss += loss.data
        loss.backward(retain_graph=True)
        optimizer.step()
        print("++++++++++++++++",iteration)
        if iteration % print_step == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration + 1, len(train_loader),
                                                                    epoch_loss / print_step))

            checkpoint(epoch)
            epoch_loss = 0

def checkpoint(epoch):
    model_out_path = "./trainedModel/epoch_relu_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, epochsize + 1):
    train(epoch)
    checkpoint(epoch)
