import torch as tr
from torch.utils.data import Dataset, DataLoader
import Utils
import CNNmodels.Model as Model
import torchaudio
import matplotlib.pyplot as plt
import librosa
from prefetch_generator import BackgroundGenerator
import time

# seperate the dataset into training set and test set
music_data = Utils.MusicDataLoader("Utils/processed_data.npy")
trainset_size = int(music_data.__len__()*0.85)
testset_size = music_data.__len__()-trainset_size
train_set, test_set = tr.utils.data.random_split(music_data,[trainset_size, testset_size])
train_loader = DataLoader(train_set, batch_size=80, num_workers=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, num_workers=11, pin_memory=True)

epochs = 60
lr = 0.003
print_iters = 100
model = Model.CnnModel(num_class=8).cuda()
criterion = tr.nn.BCEWithLogitsLoss()
optimizer = tr.optim.Adam(model.parameters(), lr=lr)

for e in range(epochs):
    sum_loss = 0.0
    print('------------------------->epoch: ',e)
    for _, item in enumerate(train_loader):
        data, label = item
        y_ = model(data.cuda())
        loss = criterion(y_, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _%print_iters==(print_iters-1):
            print("------> training loss:", sum_loss.item()/print_iters)
            sum_loss = 0.0

tr.save(model.state_dict(), "CNNmodels/cnnModel2.pth")
model.eval()
