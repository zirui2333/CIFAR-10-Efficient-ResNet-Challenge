# ==========================================
# 0. 自动依赖检查与安装 (Auto-Install Missing Libraries)
# ==========================================
import sys
import subprocess

def install_and_import(pip_package_name, import_name):
    try:
        __import__(import_name)
    except ImportError:
        print(f"⚠️ 未检测到库 '{import_name}'，正在自动安装 '{pip_package_name}'...")
        # 尝试使用当前环境的 pip 进行安装
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_package_name])
            print(f"✅ 成功安装 '{pip_package_name}'！")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装 '{pip_package_name}' 失败，可能是因为权限不足。请尝试手动运行: pip install {pip_package_name} --user")
            sys.exit(1)

# 定义需要检查的库 (pip安装名, import导入名)
required_packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("Pillow", "PIL"),
    ("scikit-learn", "sklearn"),
    ("torch", "torch"),
    ("torchvision", "torchvision")
]

for pip_name, mod_name in required_packages:
    install_and_import(pip_name, mod_name)

print("✅ 所有依赖库检查完毕，环境就绪！\n")


import os
import sys
import time
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ==========================================
# 0. HPC 路径与基础设置
# ==========================================
OUTPUT_DIR = "./output/project1_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = "Exp7_BasicBlock_200Ep_Warmup_Smooth"

EXPERIMENT_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "experiments.csv")
PER_EPOCH_LOG_FILE = os.path.join(OUTPUT_DIR, "epoch_log.csv")

# 固定的 LATEST 检查点路径，用于断点续训（不带 RUN_ID 以便重连时能找到）
LATEST_CHECKPOINT = os.path.join(OUTPUT_DIR, f"{EXP_NAME}_latest.pth")

# ==========================================
# 1. 设备与加速判断
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# ==========================================
# 2. 纯净化数据增强与 Dataset 定义
# ==========================================
DATA_DIR = "./cifar_data"
ZIP_FILE = "data.zip" 
CRITICAL_FILE = os.path.join(DATA_DIR, 'train_labels.csv') 

if not os.path.exists(CRITICAL_FILE):
    if os.path.exists(ZIP_FILE):
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

class CIFARDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 1] 
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, self.data.iloc[idx, 0]
        else:
            return image, self.data.iloc[idx, 2]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TRAIN_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_images')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

full_train_df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(
    full_train_df, test_size=0.1, random_state=42, stratify=full_train_df.iloc[:, 2]
)

temp_train_csv = f'temp_train_{RUN_ID}.csv'
temp_val_csv = f'temp_val_{RUN_ID}.csv'
train_df.to_csv(temp_train_csv, index=False)
val_df.to_csv(temp_val_csv, index=False)

train_dataset = CIFARDataset(temp_train_csv, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = CIFARDataset(temp_val_csv, TRAIN_IMG_DIR, transform=val_transform)
test_dataset = CIFARDataset(TEST_CSV, TEST_IMG_DIR, transform=val_transform, is_test=True)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 3. 架构重构：回归 BasicBlock (宽且浅)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet4M_Basic(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet4M_Basic, self).__init__()
        self.in_planes = 48 
        
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 48, 2, stride=1) 
        self.layer2 = self._make_layer(BasicBlock, 96, 2, stride=2) 
        self.layer3 = self._make_layer(BasicBlock, 192, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2) 
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out) 
        out = self.fc(out)
        return out

# ==========================================
# 4. CutMix 平滑降级 (alpha=0.5)
# ==========================================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=0.5): 
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam

# ==========================================
# 5. 模型实例化与调度器 (包含断点续训加载逻辑)
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = ResNet4M_Basic().to(device)
param_count = count_parameters(model)
print(f"Device: {device} | Params: {param_count:,}")
assert param_count <= 5000000, "Error: Model exceeds the 5M parameter limit!"

EPOCHS = 200
BASE_LR = 0.1
WARMUP_EPOCHS = 5
CUTMIX_OFF_EPOCH = 180

optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)
criterion_standard = nn.CrossEntropyLoss()
criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(EPOCHS - WARMUP_EPOCHS))

best_val_acc = 0.0
best_epoch = 0
start_epoch = 1

# --- 断点续训加载逻辑 ---
if os.path.exists(LATEST_CHECKPOINT):
    print(f"\n⚠️ 发现断点检查点，正在从 {LATEST_CHECKPOINT} 恢复训练...")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_cosine.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    print(f"✅ 成功恢复！将直接从 Epoch {start_epoch} 开始训练，历史最高 Val Acc: {best_val_acc:.2f}%\n")

start_time_total = time.time()
print(f"Starting training for {EXP_NAME} (Run ID: {RUN_ID})...")

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_start_time = time.time()
    
    if epoch <= WARMUP_EPOCHS:
        lr = BASE_LR * (epoch / WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    current_lr = optimizer.param_groups[0]['lr']
    use_cutmix = (epoch <= CUTMIX_OFF_EPOCH)
    criterion = criterion_standard if use_cutmix else criterion_smooth
    
    # --- 训练阶段 ---
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=0.5)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if use_cutmix:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if use_cutmix:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    if epoch > WARMUP_EPOCHS:
        scheduler_cosine.step()
        
    train_loss = train_loss / total
    train_acc = 100. * correct / total
    
    # --- 验证阶段 ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion_standard(outputs, targets) 
            
            _, predicted = outputs.max(1)
            val_loss += loss.item() * inputs.size(0)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
                
    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    gen_gap = val_loss - train_loss
    epoch_time = time.time() - epoch_start_time
    
    print(f"Epoch {epoch:03d}/{EPOCHS} | LR: {current_lr:.4f} | "
          f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
          f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%) | "
          f"Gap: {gen_gap:.4f} | Time: {epoch_time:.1f}s")

    epoch_record = pd.DataFrame([{
        'Exp_Name': EXP_NAME, 
        'Epoch': epoch, 
        'Train_Loss': round(train_loss, 4), 
        'Val_Loss': round(val_loss, 4), 
        'Val_Acc': round(val_acc, 2),
        'Epoch_Time': round(epoch_time, 2)
    }])
    epoch_record.to_csv(PER_EPOCH_LOG_FILE, mode='a', header=not os.path.exists(PER_EPOCH_LOG_FILE), index=False)
    
    # --- 保存当前最新进度 (断点续训) ---
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_cosine.state_dict(),
        'best_val_acc': best_val_acc
    }, LATEST_CHECKPOINT)

    # --- 保存历史最佳模型 ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        model_filename = f"{EXP_NAME}_best.pth"  # 去掉了 RUN_ID 防止多次重启产生多个 best 文件
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, model_filename))

if os.path.exists(temp_train_csv): os.remove(temp_train_csv)
if os.path.exists(temp_val_csv): os.remove(temp_val_csv)

# ==========================================
# 6. 生成 Submission (带 TTA 测试时增强)
# ==========================================
total_time = time.time() - start_time_total
avg_epoch_time = total_time / (EPOCHS - start_epoch + 1)

summary_record = pd.DataFrame([{
    'Exp_Name': EXP_NAME,
    'Param_Count': param_count,
    'Total_Time': round(total_time, 2),
    'Avg_Epoch_Time': round(avg_epoch_time, 2),
    'Best_Val_Acc': best_val_acc
}])
summary_record.to_csv(EXPERIMENT_SUMMARY_FILE, mode='a', header=not os.path.exists(EXPERIMENT_SUMMARY_FILE), index=False)

print(f"\nTraining Complete! Best Val Acc: {best_val_acc:.2f}% at Epoch {best_epoch}")
print("Generating submission with TTA...")

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, model_filename)))
model.eval()

submission_ids = []
submission_labels = []
all_avg_probs = [] 

with torch.no_grad():
    for inputs, img_ids in test_loader:
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        
        inputs_flip = torch.flip(inputs, dims=[3]) 
        outputs_flip = model(inputs_flip)
        probs_flip = torch.softmax(outputs_flip, dim=1)
        
        avg_probs = (probs + probs_flip) / 2.0
        _, predicted = avg_probs.max(1)
        
        submission_ids.extend(img_ids.tolist())
        submission_labels.extend(predicted.tolist())
        all_avg_probs.append(avg_probs.cpu().numpy())

all_avg_probs = np.concatenate(all_avg_probs, axis=0) 
prob_filename = f"probs_{EXP_NAME}_{RUN_ID}.npy"
np.save(os.path.join(OUTPUT_DIR, prob_filename), all_avg_probs)
print(f"✅ 已保存概率矩阵: {prob_filename}")

submission_df = pd.DataFrame({'id': submission_ids, 'label': submission_labels})
submission_filename = f"submission_{EXP_NAME}_{RUN_ID}.csv"
submission_df.to_csv(os.path.join(OUTPUT_DIR, submission_filename), index=False)
print(f"Generated: {submission_filename}")