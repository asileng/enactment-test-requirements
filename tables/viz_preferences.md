# 可视化偏好设置

## 禁止项（Do Not）
- ❌ 不要画跨动词的彩色环线（cross-verb colored lines）
- ❌ 不要区分中英文颜色（统一灰色）
- ❌ 不要连接到中心（保持开放式折线）

## 要求项（Requirements）
- ✅ 每个动词一个扇区，共12个扇区
- ✅ 每个扇区内5个点（p1-p5），共60个点
- ✅ 每个点只连接到角度更大的下一个点（开放式折线）
- ✅ 人类数据统一使用灰色
- ✅ 每个扇区使用不同的线条样式区分

## 参数顺序
p1=FORCE → p2=HAND → p3=ARM → p4=HD → p5=VD

---

## 动词区分分析（Pairwise Analysis）

### 各语言区分维度
- **中文**：5维全部使用（FORCE, HAND, ARM, HD, VD）
- **英文**：3维（FORCE, HAND, ARM）
- ~~德文~~：已移除，不使用德语数据

### FORCE 区分力
- **最轻**：diu 丢 (2.33)
- **中等**：reng 掷 (3.05), pao 抛 (3.16), tou 投 (3.21)
- **较重**：shuai 甩 (3.74)
- **最重**：shuai 摔 (4.55)

### HAND 区分力
- **最低**：diu 丢 (4.63)
- **中低**：reng 掷 (5.43), shuai 甩 (6.27)
- **中等**：pao 抛 (6.65)
- **较高**：shuai 摔 (8.51)
- **最高**：tou 投 (9.13)

### VD (垂直方向) 特殊区分
- **向上**：reng, diu, pao, tou, throw, fling, chuck, cast, hurl, toss
- **向下**：shuai 摔, schmettern, pfeffern
- **无明显方向**：shuai 甩

### HD (水平方向) 特殊区分
- **向前**：大部分动词
- **向侧**：shuai 甩 (唯一)

### ARM (手臂姿势) 特殊区分
- **直臂**：diu 丢, toss (仅这两个)
- **弯臂**：tou 投, throw, fling, chuck, hurl, shuai 摔, shuai 甩
- **无明显模式**：reng 掷, pao 抛, cast
