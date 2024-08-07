# 一、Model
音频 + 视频 双流网络

# 二、训练
## 2.1、Train:
```python
torchrun --standalone --nproc_per_node=3 train.py --gpu 0 1 2
```
## 2.2、Loss:
```python
1、交叉熵损失
2、二元交叉熵损失
3、交叉熵loss+二元交叉熵损失
```
# 三、Data
## 3.1、视频转音频
```python
python video2wav.py
```
## 3.2、生成csv文件
```python
python read2csv.py
```
## 3.3、数据格式
```python
--｜DeepDakeData
--｜--｜trainset
--｜--｜--｜0a0a2e5239ce88e393d50e915e9b523c.mp4
--｜--｜--｜...
--｜--｜valset
--｜--｜--｜0a0c7194d40d33c56b1ab6259188ec97.mp4
--｜--｜--｜...
--｜--｜wav
--｜--｜--｜trainset
--｜--｜--｜--｜0a0a2e5239ce88e393d50e915e9b523c.wav
--｜--｜--｜--｜...
--｜--｜--｜valset
--｜--｜--｜--｜0a0c7194d40d33c56b1ab6259188ec97.wav
--｜--｜--｜--｜...
--｜--｜train_list.csv
--｜--｜val_list.csv
```
