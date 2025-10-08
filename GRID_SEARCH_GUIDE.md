# 网格搜索使用指南

## 🚀 快速开始

### 1. 快速搜索 (推荐,~6个实验)
```bash
python grid_search.py --dataset caltech101 --backbone ViT-B/16 --mode quick
```
**搜索参数**:
- `pos_alpha`: [1.5, 2.0, 2.5]
- `cal_enabled`: [True, False]
- `cal_fusion_alpha`: [0.8, 0.9]

**耗时**: 约30分钟

---

### 2. 只搜索GR-CLIP参数 (~60个实验)
```bash
python grid_search.py --mode grclip_only
```
**搜索参数**:
- `cal_fusion_alpha`: [0.3, 0.5, 0.7, 0.8, 0.9]
- `cal_confidence_threshold`: [0.6, 0.7, 0.8, 0.9]
- `cal_min_cache_size`: [10, 20, 30]

**耗时**: 约3-4小时

---

### 3. 只搜索TDA参数 (~135个实验)
```bash
python grid_search.py --mode tda_only
```
**搜索参数**:
- `pos_shot_capacity`: [2, 3, 4]
- `pos_alpha`: [1.0, 1.5, 2.0, 2.5, 3.0]
- `pos_beta`: [4.0, 5.0, 6.0]
- `neg_alpha`: [0.1, 0.117]

**耗时**: 约6-8小时

---

### 4. 完整搜索 (~300+实验,非常耗时!)
```bash
python grid_search.py --mode full
```
**搜索参数**: 所有参数的组合

**耗时**: 约12-20小时

---

## 📊 结果输出

### 目录结构
```
results/
└── grid_search_20251006_013000/
    ├── results_final.csv        # 最终结果 (按准确率降序)
    ├── results_partial.csv      # 中间结果 (实时更新)
    └── best_config.txt          # 最佳配置 (YAML格式)
```

### 查看结果
```python
import pandas as pd

# 读取结果
df = pd.read_csv('results/grid_search_XXXXXX/results_final.csv')

# 查看Top 10
print(df.head(10))

# 筛选GR-CLIP启用的实验
df_grclip = df[df['cal_enabled'] == True]
print(df_grclip.head(5))

# 筛选纯TDA实验
df_tda = df[df['cal_enabled'] == False]
print(df_tda.head(5))
```

---

## 🎯 推荐实验流程

### Step 1: 快速搜索 (30分钟)
```bash
python grid_search.py --mode quick
```
**目的**: 快速验证GR-CLIP是否有效

### Step 2: 根据结果选择策略

#### 情况A: GR-CLIP有效 (cal_enabled=True的结果更好)
```bash
# 深入搜索GR-CLIP参数
python grid_search.py --mode grclip_only
```

#### 情况B: GR-CLIP无效 (cal_enabled=False的结果更好)
```bash
# 优化TDA参数
python grid_search.py --mode tda_only
```

### Step 3: 应用最佳配置
将 `results/grid_search_XXXXXX/best_config.txt` 的内容复制到你的配置文件

---

## 🛠️ 自定义搜索空间

编辑 `grid_search.py` 中的 `search_space` 字典:

```python
# 示例: 只搜索fusion_alpha和confidence_threshold
search_space = {
    'cal_fusion_alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
    'cal_confidence_threshold': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
```

**参数命名规则**:
- `pos_*`: 正缓存参数 (如 `pos_alpha`, `pos_beta`)
- `neg_*`: 负缓存参数 (如 `neg_alpha`)
- `cal_*`: GR-CLIP参数 (如 `cal_fusion_alpha`, `cal_enabled`)

---

## 💡 调参建议

### 基于你的实验结果:

| 配置 | 准确率 | 建议 |
|------|--------|------|
| 纯TDA | 94.12% | ✅ 最好,保持这个配置 |
| GR-CLIP (α=0.8) | 93.63% | ❌ 不如纯TDA |

**结论**: Caltech101不需要GR-CLIP

### 下一步:
1. 运行 `--mode tda_only` 优化TDA参数
2. 尝试其他数据集 (ImageNet-A/V/R/S)
3. 在难数据集上测试GR-CLIP效果

---

## 📈 实验追踪

### 记录实验日志
```bash
# 创建实验记录
echo "$(date): Running grid search on Caltech101" >> experiments.log
python grid_search.py --mode quick 2>&1 | tee -a experiments.log
```

### 对比多次实验
```python
import pandas as pd
import glob

# 读取所有实验结果
all_results = []
for csv_file in glob.glob('results/*/results_final.csv'):
    df = pd.read_csv(csv_file)
    df['experiment'] = csv_file.split('/')[1]
    all_results.append(df)

combined = pd.concat(all_results)
print(combined.sort_values('accuracy', ascending=False).head(20))
```

---

## ⚠️ 注意事项

1. **随机性**: 每次运行结果可能略有不同 (±0.1-0.3%)
2. **内存**: 确保有足够GPU显存 (建议8GB+)
3. **中断恢复**: 脚本会实时保存 `results_partial.csv`,可查看进度
4. **并行运行**: 不建议,会导致GPU OOM

---

## 🔍 调试技巧

### 查看某个实验的详细日志
```python
# 在 grid_search.py 中添加日志
print(f"[DEBUG] Config: {config}")
print(f"[DEBUG] Accuracy: {accuracy}")
```

### 单独测试某个配置
```python
# 快速验证某个参数组合
python runner.py --config configs/ --datasets caltech101 --backbone ViT-B/16
# 然后手动修改 configs/caltech101.yaml
```

---

## 📞 常见问题

**Q: 网格搜索太慢怎么办?**
A: 使用 `--mode quick` 或减少搜索空间

**Q: 如何添加新参数?**
A: 在 `search_space` 中添加,使用 `pos_/neg_/cal_` 前缀

**Q: 结果保存在哪?**
A: `results/grid_search_时间戳/` 目录

**Q: 如何恢复中断的搜索?**
A: 目前不支持,建议使用 `--mode quick` 减少实验数量

---

## 🎓 最佳实践

1. **先快后慢**: 先 `quick`,再 `grclip_only` 或 `tda_only`
2. **记录实验**: 使用 `tee` 保存日志
3. **版本控制**: 实验前 `git commit` 记录代码状态
4. **数据分析**: 用 pandas 分析结果,找规律
5. **多数据集**: 在多个数据集上验证最佳配置

---

祝调参顺利! 🚀
