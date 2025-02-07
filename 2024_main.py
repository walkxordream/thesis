import glob
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
import torch.nn as nn

def findcontours(img):
    # グレースケールに変換する。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2値化する
    ret, bin_img = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    # 輪郭を抽出する。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 最大面積の輪郭を保存する
    max_area = 0
    for i, cnt in enumerate(contours):
        # 面積
        area = cv2.contourArea(cnt)
        max_area = max(max_area, area)
    
    return max_area

def make_models(model_paths):
        
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
    
    models = []
    for model_path in model_paths:
        model = DeepAutoencoder().cuda()
        model.load_state_dict(torch.load(model_path))
        models.append(model)
    return models

def AE(IMG, models):
    min_mse = float('inf')
    best_area = 0
    preprocess = T.Compose([T.ToTensor()])

    for model in models:
        model.eval()
        img = preprocess(IMG).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img)[0]
        output = output.cpu().numpy().transpose(1, 2, 0)
        output = np.uint8(np.maximum(np.minimum(output * 255, 255), 0))
        origin = np.uint8(img[0].cpu().numpy().transpose(1, 2, 0) * 255)
        diff = np.uint8(np.abs(output.astype(np.float32) - origin.astype(np.float32)))
        mse = np.mean((diff.astype(np.float32)) ** 2)
        area = findcontours(diff)
        
        if mse < min_mse:
            min_mse = mse
            best_area = area
    return best_area

def split(files):
    # 画像分割サイズ
    crop_size = 224
    # 分割後の画像を格納するリスト
    cropped_images = []

    for file_index, file in enumerate(files):
        img = cv2.imread(file)
        h, w = img.shape[:2]

        position = 1
        for y in range(0, h, crop_size):
            for x in range(0, w, crop_size):
                if x + crop_size > w or y + crop_size > h:
                    continue
                cropped_img = img[y:y + crop_size, x:x + crop_size]
                cropped_images.append([cropped_img, [file_index + 1, position]])
                position += 1

    return cropped_images
#出力は、[[画像, [画像番号, 画像の位置]]...]となる

def autoencoder(IMAGES, models):
    # 物体ありと判断する面積のしきい値
    area_threshold = 1200
    # AEに通した結果、居た画像を保存しておくリスト
    AE_YES_img = []
    # AEに通した結果、居た画像の位置を保存しておくリスト
    AE_YES_position = []
    # 画像は1枚ずつ
    for image_position in IMAGES:
        # 呼び出した関数からは面積と輪郭画像が出力されてくる
        area = AE(image_position[0], models)

        # 面積がしきい値より大きければ、暫定鳥ありとして格納
        if area > area_threshold:
            AE_YES_img.append(image_position[0])
            AE_YES_position.append(image_position[1])

    return AE_YES_img, AE_YES_position


def get_image_filenames(directory):
    # 指定されたディレクトリ内の画像ファイル名を取得
    return set(f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

def count_common_images(dir1, dir2):
    # 2つのディレクトリ内の共通の画像ファイル名をカウント
    images_dir1 = get_image_filenames(dir1)
    images_dir2 = get_image_filenames(dir2)
    
    common_images = images_dir1.intersection(images_dir2)
    return len(common_images)

def calculate_f2_score(tp, fp, fn):
    # PrecisionとRecallを計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F2スコアを計算
    if precision + recall == 0:
        return 0
    f2_score = (5 * precision * recall) / (4 * precision + recall)
    return f2_score

# メインの処理を行う関数
def main():
    model_paths = ["models/fine_model_paths/6048_fineAEdeepmodel_20250115_rdark.pth","models/fine_model_paths/6048_fineAEdeepmodel_20250115_rlight.pth","models/fine_model_paths/6048_fineAEdeepmodel_20250115_rwhite.pth"]
    models = make_models(model_paths)
    files = list(glob.glob("imgs/test_img/*.JPG"))
    images = split(files)
    img_list, posision_list = autoencoder(images, models)

    # 画像を保存するディレクトリを作成
    save_dir = "imgs/AE_2024_rresult"
    os.makedirs(save_dir, exist_ok=True)

    # 画像を保存
    for idx, img in enumerate(img_list):
        save_path = os.path.join(save_dir, f"detected_{posision_list[idx][0]}_{posision_list[idx][1]}.jpg")
        cv2.imwrite(save_path, img)

    return img_list, posision_list

if __name__ == "__main__":
    img_list, posision_list = main()
    
    # 共通の画像ファイルの枚数をカウント
    directory1 = "imgs/AE_2024_rresult"
    directory2 = "imgs/test_objects"
    tp = count_common_images(directory1, directory2)
    print(f"共通の画像ファイルの枚数 (TP): {tp}")

    # 正解の物体数
    total_objects = 50

    # 推論結果の画像の枚数 (FP)
    fp = len(get_image_filenames(directory1)) - tp

    # 分割後の画像の枚数 (全体の画像数)
    split_directory = "imgs/test_img"
    total_images = len(split(glob.glob(os.path.join(split_directory, "*.JPG"))))

    # False Negatives (FN) を計算
    fn = total_objects - tp

    # True Negatives (TN) を計算
    tn = total_images - fp - fn 

    # F2スコアを計算
    f2_score = calculate_f2_score(tp, fp, fn)
    print(f"推論結果の画像の枚数 (FP): {fp}")
    print(f"全体の画像の枚数 (Total): {total_images}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print(f"F2スコア: {f2_score}")


