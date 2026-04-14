# PIRN 服务器部署教程

本文档给出从 0 到 1 的部署流程，包括：

1. 代码拉取与环境准备
2. 前端上线（Nginx）
3. 训练任务运行（CLI）
4. 常驻服务（systemd）
5. 常见问题排查

---

## 1) 服务器环境准备

推荐系统：Ubuntu 22.04+

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip nginx
```

如果服务器有 NVIDIA GPU：

```bash
nvidia-smi
```

确认驱动正常后再安装对应 CUDA 版本的 PyTorch。

---

## 2) 拉取代码并安装

```bash
git clone https://github.com/lqj311/PIRN.git
cd PIRN
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> 如果需要 GPU 版 PyTorch，请按你服务器 CUDA 版本安装官方 wheel，再执行 `pip install -e .`。

---

## 3) 准备数据（特征级）

目录格式：

```text
data_root/
  train/normal/*.pt
  test/normal/*.pt
  test/anomaly/*.pt
```

每个 `*.pt` 为字典，至少包含 RGB 与 SN 特征（见 `README.md`）。

可先快速生成玩具数据验证流程：

```bash
python examples/make_toy_feature_dataset.py --out toy_features --dim 768 --tokens 196
```

---

## 4) 启动训练

```bash
python -m pirn_paper.train \
  --data-root toy_features \
  --output-dir runs/toy \
  --epochs 5 \
  --batch-size 4 \
  --device cpu
```

产物：

- `runs/toy/config.json`
- `runs/toy/metrics.csv`
- `runs/toy/best.pt`

评估：

```bash
python -m pirn_paper.eval \
  --data-root toy_features \
  --checkpoint runs/toy/best.pt \
  --batch-size 4 \
  --device cpu
```

---

## 5) 前端上线（Nginx 方式）

前端是纯静态页面，直接由 Nginx 托管。

### 5.1 拷贝静态文件

```bash
sudo mkdir -p /var/www/pirn
sudo cp -r frontend/* /var/www/pirn/
```

### 5.2 新建 Nginx 配置

```bash
sudo tee /etc/nginx/sites-available/pirn <<'EOF'
server {
    listen 80;
    server_name _;

    root /var/www/pirn;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF
```

启用并重载：

```bash
sudo ln -sf /etc/nginx/sites-available/pirn /etc/nginx/sites-enabled/pirn
sudo nginx -t
sudo systemctl restart nginx
```

浏览器访问：`http://<你的服务器IP>/`

---

## 6) 可选：前端本地 Python 服务

如果不想装 Nginx：

```bash
python frontend/server.py
```

访问：`http://<服务器IP>:8080`

---

## 7) 训练任务 systemd（可选）

创建服务文件：

```bash
sudo tee /etc/systemd/system/pirn-train.service <<'EOF'
[Unit]
Description=PIRN Training Job
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PIRN
Environment=PATH=/home/ubuntu/PIRN/.venv/bin
ExecStart=/home/ubuntu/PIRN/.venv/bin/python -m pirn_paper.train --data-root /data/pirn_features --output-dir /home/ubuntu/PIRN/runs/prod --epochs 100 --batch-size 8 --device cuda
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable pirn-train
sudo systemctl start pirn-train
sudo systemctl status pirn-train
```

日志查看：

```bash
journalctl -u pirn-train -f
```

---

## 8) 常见问题

1. `ModuleNotFoundError: torch`
- 安装 PyTorch 后再执行训练命令。

2. `No samples found`
- 检查数据目录是否是 `train/normal`, `test/normal`, `test/anomaly`。

3. CUDA 不可用
- 用 `--device cpu` 验证流程；GPU 环境检查驱动/CUDA/PyTorch 版本匹配。

4. Nginx 403/404
- 检查 `/var/www/pirn` 权限与 `root` 路径，重新 `nginx -t`。

