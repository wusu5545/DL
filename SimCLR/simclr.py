import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torchvision.models.resnet import resnet50


class SimCLR(nn.Module):
    def __init__(self,feature_dim = 128):
        super(SimCLR, self).__init__()

        # base model f()
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)

        # projection head
        self.g = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        # h
        feature = torch.flatten(x, 1)
        # z
        output = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(output, dim=-1)


def info_nec_loss(features, temperature, device):

    # calculate cosine similarity
    cos_sim = F.cosine_similarity(features[:,None,:], features[None,:,:], dim=-1)
    # mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # find positive example -> batch_size // 2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # info nec loss
    cos_sim = cos_sim / temperature
    nll = (-cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)).mean()

    return nll

def train(model, data_loader, train_optimizer, transform, epoch, epochs, batch_size=32, temperature=0.5, device = 'cuda'):
    model.train()
    total_loss = 0
    total_num = 0
    train_bar = tqdm(data_loader)
    for x_i, x_j, _ in train_bar:
        x_i, x_j = x_i.to(device), x_j.to(device)
        out_left, out_right, loss = None, None, None
        # encode all image
        (_, out_left),(_, out_right) = model (x_i), model(x_j)
        out = torch.cat([out_left, out_right], dim = 0)
        loss = info_nec_loss(out, temperature, device)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num

def test(model, memory_loader,test_loader, epoch, epochs, classes, batch_size=32, temperature=0.5, k = 200, device='cuda'):
    model.eval()
    total_top1, total_top5 = 0.0, 0.0
    total_num = 0
    feature_list = []

    with torch.no_grad():
        # build feature list
        for images, labels in tqdm(memory_loader, desc='Feature extracting'):
            feature, _ = model(images.to(device))
            feature_list.append(feature)

        # [D,N]
        feature_list = torch.cat(feature_list, dim = 0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_loader.dataset.targets, device = feature_list.device)

        # predict labels by weighted knn search
        test_bar = tqdm(test_loader)
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            feature, _ = model(images)

            total_num += images.size(0)
            # compute cos similarity between each feature vector and feature_list [B, N]
            sim_matrix = torch.mm(feature, feature_list)

            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k, dim = -1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(images.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(images.size(0) * k, classes, device=device)
            # [B * k, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weight score -> [B, C]
            pred_score = torch.sum(one_hot_label.view(images.size(0), -1, classes) * sim_weight.unsqueeze(dim = -1), dim = 1)

            pred_labels = pred_score.argsort(dim=-1, descending = True)

            total_top1 += torch.sum((pred_labels[:, :1] == labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cuda'):
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100