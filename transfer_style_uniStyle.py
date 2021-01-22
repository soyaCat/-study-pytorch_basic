import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

import time

content_layer_num = 1
image_size = 256
totalEpoch = 250
content_dir = "./transfer_data/images_for_transfer_style/content/Neckarfront.jpg"
style_dir = "./transfer_data/images_for_transfer_style/style/StarryNight.jpg"



class Resnet_model(nn.Module):
    def __init__(self):
        super(Resnet_model, self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    
    def forward(self,x):
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5

class image_process():
    # Since trained ResNet model is trained by ImageNet model, normalize accordingly.
    def image_preprocess(self, img_dir):
        img = Image.open(img_dir)
        transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
                                    ])
        img = transform(img).view((-1,3,image_size,image_size))
        return img

    # Add the subtracted values ​​to re-image.
    # also img has value 0~1
    def image_postprocess(self, tensor):
        transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1])
        img = transform(tensor.clone())
        img = img.clamp(0,1)
        img = torch.transpose(img,0,1)
        img = torch.transpose(img,1,2)
        return img

class Style_transfer():
    def __init__(self, device, generated):
        self.device = device
        self.model = Resnet_model().to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.LBFGS([generated])
        self.content_layer_num = content_layer_num

    def GramMatrix(self, input):
        b,c,h,w = input.size()
        F = input.view(b,c,h*w)
        Gram = torch.bmm(F, F.transpose(1,2))
        return Gram

    def GramMSELoss(self, input, target):
        out = self.loss_func(self.GramMatrix(input), target)
        return out

    def closure(self):
        self.optimizer.zero_grad()
        out = self.model.forward(self.generated)
        # The style loss is calculated according to each target value and saved as a list.
        style_loss = []
        for i in range(len(self.style_target_GramMatrix)):
            style_loss.append(self.GramMSELoss(out[i], self.style_target_GramMatrix[i]).to(self.device) * style_weight[i])
        content_loss = self.loss_func(out[self.content_layer_num], self.content_target)

        #calculate the total loss as a weight of Style:Content = 1000:1
        total_loss = 1000 * sum(style_loss) + torch.sum(content_loss)
        total_loss.backward()
        
        self.total_loss = total_loss
        return total_loss

    def train_model(self, generated, style_target_GramMatrix, content_target):
        self.generated = generated
        self.style_target_GramMatrix = style_target_GramMatrix
        self.content_target = content_target
        self.optimizer.step(self.closure)

if __name__ == "__main__":
    resnet = torchvision.models.resnet50(pretrained=True)
    for name,module in resnet.named_children():
        print(name)
        '''conv1
        bn1
        relu
        maxpool
        layer1
        layer2
        layer3
        layer4
        avgpool
        fc'''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_process = image_process()
    # Define content images, style images, and generated image to be trained
    content = image_process.image_preprocess(content_dir).to(device)
    style = image_process.image_preprocess(style_dir).to(device)
    generated = content.clone().requires_grad_().to(device)
    Style_transfer = Style_transfer(device, generated)


    # Set the target value and also define the weigh accoding to the size of the matrix
    style_target_GramMatrix = []
    for result in Style_transfer.model.forward(style):
        style_target_GramMatrix.append(Style_transfer.GramMatrix(result).to(device))

    content_target = Style_transfer.model.forward(content)[content_layer_num]

    style_weight = [1/n**2 for n in [64,64,256,512,1024,2048]]

    start = time.time()
    for epoch in range(totalEpoch):
        Style_transfer.train_model(generated, style_target_GramMatrix, content_target)

        if epoch % 10 == 0:
            print(Style_transfer.total_loss)

    print("\n")
    print("print total_learning_time:", time.time() - start)
    gen_img = image_process.image_postprocess(generated[0].cpu()).data.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(gen_img)
    plt.show()



'''result
tensor(84.3030, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.7112, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.2178, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.1422, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.1151, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.1013, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0929, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0873, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0832, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0800, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0776, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0757, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0741, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0727, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0715, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0705, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0696, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0688, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0681, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0675, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0669, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0664, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0659, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0655, device='cuda:0', grad_fn=<AddBackward0>)
tensor(0.0651, device='cuda:0', grad_fn=<AddBackward0>)


print total_learning_time: 592.2310276031494
'''