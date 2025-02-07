import os
import glob
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import csv
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_models(data_path, batchsize, model_path, save_model_dir, seed=42):
    # シードを固定
    set_seed(seed)

    # data_pathsのひとつ下の階層にあるフォルダを取得
    data_paths = glob.glob(f"{data_path}/*")
    # ソートすることで、データの順番を固定する
    data_paths = sorted(data_paths)
    l = list(data_paths)
    # 今日の日付を取得
    ll = datetime.datetime.now().strftime('%Y%m%d')
    for i in range(len(l)):
        print(str(i + 1) + "個目のモデル作成を始めました！")
        print(l[i], model_path[i])
        dir_path = l[i]
        study_data = glob.glob(f"{dir_path}/*")
        print(f"good {len(study_data)}")

        # データセット関数の定義
        class Custom_Dataset(Dataset):
            def __init__(self, img_list):
                self.img_list = img_list
                self.preprocess = T.Compose([T.ToTensor()])

            def __getitem__(self, idx):
                img = Image.open(self.img_list[idx])
                img = self.preprocess(img)
                return img

            def __len__(self):
                return len(self.img_list)

        # データを学習用・評価用に8:2へ分割
        train_list = study_data[:int(len(study_data) * 0.8)]
        val_list = study_data[int(len(study_data) * 0.8):]

        train_dataset = Custom_Dataset(train_list)
        val_dataset = Custom_Dataset(val_list)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)


        class DeepAutoencoder(nn.Module):
            def __init__(self):
                super(DeepAutoencoder, self).__init__()
                self.Encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 256 -> 128
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 128 -> 64
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 64 -> 32
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 32 -> 16
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # 16 -> 8
                )
                self.Decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16 -> 32
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32 -> 64
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 64 -> 128
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  # 128 -> 256
                    nn.ReLU(),
                    nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
                )

            def forward(self, x):
                x = self.Encoder(x)
                x = self.Decoder(x)
                return x

        # 全体学習回数    
        epoch_num = 1000
        # GPUを用いて学習する
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 1番少ない誤差の値を保存する
        best_loss = None
        # modelをGPU用に用意
        model = DeepAutoencoder().to(device)
        model.load_state_dict(torch.load(model_path[i]))
        # 連続学習回数にも制限を用意
        limit_epoch = 30

        # モデルの最適化手法
        optimizer = optim.Adam(model.parameters())
        # 誤差関数
        criterion = nn.MSELoss()

        counter = 0
        # 損失を記録するリスト
        train_losses = []
        val_losses = []

        # 保存先のディレクトリを作成
        os.makedirs(save_model_dir, exist_ok=True)

        for e in range(epoch_num):
            total_loss = 0
            # モデルを学習用に設定
            model.train()
            with tqdm(train_loader) as pbar:
                for itr, data in enumerate(pbar):
                    optimizer.zero_grad()
                    data = data.to(device)
                    output = model(data)
                    loss = criterion(output, data)
                    total_loss += loss.detach().item()
                    pbar.set_description(f"[train] Epoch {e+1:03}/{epoch_num:03} Itr {itr+1:02}/{len(pbar):02} Loss {total_loss/(itr+1)} Counter {counter}")
                    loss.backward()
                    optimizer.step()
            
            train_losses.append(total_loss / len(train_loader))

            total_loss = 0
            model.eval()
            with tqdm(val_loader) as pbar:
                for itr, data in enumerate(pbar):
                    data = data.to(device)
                    with torch.no_grad():
                        output = model(data)
                    loss = criterion(output, data)
                    total_loss += loss.detach().item()
                    pbar.set_description(f"[ val ] Epoch {e+1:03}/{epoch_num:03} Itr {itr+1:02}/{len(pbar):02} Loss {total_loss/(itr+1)}")
            
            val_losses.append(total_loss / len(val_loader))
            if best_loss is None or best_loss > total_loss/(itr+1):
                if best_loss is not None:
                    print(f"update best_loss {best_loss:.6f} to {total_loss/(itr+1):.6f}")
                best_loss = total_loss/(itr+1)
                save_model_path = f"{save_model_dir}/{len(study_data)}_fineAEdeepmodel_{ll}_{os.path.basename(l[i])}.pth"
                torch.save(model.state_dict(), save_model_path)
                counter = 0
            else:
                counter += 1
                if limit_epoch <= counter:
                    break

        # 学習の収束具合をプロット
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # 縦軸を対数表記に設定
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(f'imgs/result_img/loss_plot_{ll}_fine{len(study_data)}_{os.path.basename(l[i])}.png')
        plt.close()

        # 損失の値の推移をCSVファイルに書き込む
        csv_path = f'imgs/result_img/loss_plot_{ll}_fine{len(study_data)}_{os.path.basename(l[i])}.csv'
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
            for epoch in range(epoch_num):
                train_loss = train_losses[epoch] if epoch < len(train_losses) else ''
                val_loss = val_losses[epoch] if epoch < len(val_losses) else ''
                writer.writerow([epoch + 1, train_loss, val_loss])

        print(str(i + 1) + "個目のモデル作成が終了しました！")

# 最後の数字を抽出する関数
def extract_last_number(path):
    match = re.search(r'_(\d+)$', path)
    return match.group(1) if match else None

if __name__ == "__main__":
    #適宜変更
    data_pathss = list(glob.glob("imgs/update_train_imgs_remove_mix/*"))
    modelpaths = glob.glob("models/old_models6048/*")
    modelpaths = sorted(modelpaths)
    save_model_dir = "models/fine_model_paths"
    batchs = {100: 8, 1000: 32, 3000: 64, 5000: 128, 6048: 256}
    for data_paths in data_pathss:
        last_number = extract_last_number(data_paths)
        batch_size = batchs[int(last_number)]
        data_paths = list(glob.glob(f"{data_paths}/*"))
        for data_path in data_paths:
            make_models(data_path, batch_size, modelpaths, save_model_dir)