#!/usr/bin/env python
"""
网格搜索脚本 - 自动找到TDA+GR-CLIP的最优超参数

使用方法:
    # 快速搜索 (推荐)
    python grid_search.py --dataset caltech101 --backbone ViT-B/16 --mode quick

    # 只搜索GR-CLIP参数
    python grid_search.py --mode grclip_only

    # 只搜索TDA参数
    python grid_search.py --mode tda_only

    # 完整搜索 (耗时长)
    python grid_search.py --mode full

输出:
    - results/grid_search_YYYYMMDD_HHMMSS/results_final.csv: 所有实验结果
    - results/grid_search_YYYYMMDD_HHMMSS/best_config.txt: 最优配置
"""

import os
import sys
import argparse
import itertools
from datetime import datetime
from pathlib import Path
import pandas as pd
import random
import torch
import clip
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from runner import update_cache, compute_cache_logits, build_visual_prototypes, compute_modality_means, calibrate_with_gr_clip


def run_single_experiment(config, dataset_name, data_root, clip_model, clip_weights, preprocess):
    """
    运行单次实验

    Args:
        config: 配置字典
        dataset_name: 数据集名称
        data_root: 数据集根目录
        clip_model: CLIP模型
        clip_weights: CLIP文本权重
        preprocess: 预处理函数

    Returns:
        accuracy: 测试准确率
    """
    # 构建数据加载器
    test_loader, _, _ = build_test_data_loader(dataset_name, data_root, preprocess)

    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []

        # 提取配置
        pos_cfg = config['positive']
        neg_cfg = config['negative']
        cal_cfg = config['calibrate']

        text_mean, image_mean = None, None
        calibrated_weights = clip_weights

        for i, (images, target) in enumerate(tqdm(test_loader, desc='测试进度', leave=False)):
            # 步骤1: 特征提取
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target = target.cuda()
            prop_entropy = get_entropy(loss, clip_weights)

            # 步骤2: 更新正缓存
            if pos_cfg['enabled']:
                update_cache(pos_cache, pred, [image_features, loss], pos_cfg['shot_capacity'])

            # 步骤3: 更新负缓存
            if neg_cfg['enabled']:
                if neg_cfg['entropy_threshold']['lower'] < prop_entropy < neg_cfg['entropy_threshold']['upper']:
                    update_cache(neg_cache, pred, [image_features, loss, prob_map],
                               neg_cfg['shot_capacity'], True)

            # 步骤4: GR-CLIP校准
            if cal_cfg['enabled'] and len(pos_cache) >= cal_cfg['min_cache_size']:
                if i % cal_cfg['update_interval'] == 0:
                    text_mean, image_mean = compute_modality_means(pos_cache, clip_weights)
                    visual_prototypes = build_visual_prototypes(pos_cache)
                    calibrated_weights = calibrate_with_gr_clip(
                        visual_prototypes, clip_weights, text_mean, image_mean,
                        alpha=cal_cfg['fusion_alpha']
                    )

            # 步骤5: 计算预测
            clip_baseline_logits = 100.0 * image_features @ clip_weights

            if text_mean is not None and image_mean is not None and cal_cfg['enabled']:
                # 自适应GR-CLIP
                clip_confidence = clip_baseline_logits.softmax(1).max().item()
                if clip_confidence < cal_cfg.get('confidence_threshold', 1.0):
                    image_features_centered = image_features - image_mean.unsqueeze(0)
                    image_features_centered = image_features_centered / image_features_centered.norm(dim=1, keepdim=True)
                    gr_clip_logits = 100.0 * image_features_centered @ calibrated_weights
                else:
                    gr_clip_logits = clip_baseline_logits
            else:
                gr_clip_logits = clip_baseline_logits

            # 步骤6: TDA缓存
            tda_logits = torch.zeros_like(gr_clip_logits)
            if pos_cfg['enabled'] and pos_cache:
                tda_logits += compute_cache_logits(image_features, pos_cache,
                                                   pos_cfg['alpha'], pos_cfg['beta'], clip_weights)
            if neg_cfg['enabled'] and neg_cache:
                tda_logits -= compute_cache_logits(image_features, neg_cache,
                                                   neg_cfg['alpha'], neg_cfg['beta'], clip_weights,
                                                   (neg_cfg['mask_threshold']['lower'],
                                                    neg_cfg['mask_threshold']['upper']))

            # 步骤7: 融合预测
            final_logits = gr_clip_logits + tda_logits

            # 评估
            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

        avg_acc = sum(accuracies) / len(accuracies)
        return avg_acc


def grid_search(args, search_space):
    """执行网格搜索"""

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("results") / f"grid_search_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"🔍 TDA+GR-CLIP 网格搜索")
    print(f"{'='*80}")
    print(f"📁 结果目录: {result_dir}")
    print(f"🎯 数据集: {args.dataset}")
    print(f"🧠 Backbone: {args.backbone}")
    print(f"🔧 搜索模式: {args.mode}")

    # 初始化CLIP模型
    print(f"\n📥 加载CLIP模型...")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # 设置随机种子
    random.seed(1)
    torch.manual_seed(1)

    # 加载数据集和文本特征
    print(f"📥 加载数据集和文本特征...")
    from utils import get_config_file, build_test_data_loader, clip_classifier
    cfg = get_config_file(args.config_dir, args.dataset)
    test_loader, classnames, template = build_test_data_loader(args.dataset, args.data_root, preprocess)
    clip_weights = clip_classifier(classnames, template, clip_model)

    # 生成所有参数组合
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))

    total_exps = len(all_combinations)
    print(f"📊 总实验数: {total_exps}")
    print(f"{'='*80}\n")

    # 基础配置
    base_config = {
        'positive': {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        },
        'negative': {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        },
        'calibrate': {
            'enabled': True,
            'min_cache_size': 20,
            'update_interval': 100,
            'fusion_alpha': 0.8,
            'confidence_threshold': 0.7
        }
    }

    # 存储结果
    results = []
    best_acc = 0
    best_params = None
    best_config = None

    # 遍历所有组合
    for exp_id, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        print(f"\n{'─'*80}")
        print(f"🧪 实验 {exp_id}/{total_exps}")
        print(f"{'─'*80}")

        # 创建当前配置
        config = {
            'positive': base_config['positive'].copy(),
            'negative': base_config['negative'].copy(),
            'calibrate': base_config['calibrate'].copy()
        }

        # 更新参数
        for key, value in params.items():
            if key.startswith('pos_'):
                config['positive'][key.replace('pos_', '')] = value
            elif key.startswith('neg_'):
                config['negative'][key.replace('neg_', '')] = value
            elif key.startswith('cal_'):
                config['calibrate'][key.replace('cal_', '')] = value

        # 打印当前配置
        print(f"配置:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        # 运行实验
        try:
            accuracy = run_single_experiment(
                config, args.dataset, args.data_root, clip_model, clip_weights, preprocess
            )
            print(f"✅ 准确率: {accuracy:.2f}%")

            # 记录结果
            result = params.copy()
            result['accuracy'] = accuracy
            result['exp_id'] = exp_id
            results.append(result)

            # 更新最佳结果
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = params.copy()
                best_config = config
                print(f"🌟 新的最佳结果! {best_acc:.2f}%")

        except Exception as e:
            print(f"❌ 实验失败: {e}")
            result = params.copy()
            result['accuracy'] = None
            result['exp_id'] = exp_id
            result['error'] = str(e)
            results.append(result)

        # 保存中间结果
        df = pd.DataFrame(results)
        df.to_csv(result_dir / "results_partial.csv", index=False)

    # 保存最终结果
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=False)
    csv_path = result_dir / "results_final.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print(f"✨ 网格搜索完成!")
    print(f"{'='*80}")
    print(f"📊 结果已保存: {csv_path}")
    print(f"\n🏆 最佳配置:")
    print(f"   准确率: {best_acc:.2f}%")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # 保存最佳配置
    if best_config:
        config_path = result_dir / "best_config.txt"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Accuracy: {best_acc:.2f}%\n\n")
            f.write("Best Configuration:\n")
            f.write(f"{best_config}\n\n")
            f.write("YAML Format:\n")
            f.write(f"positive:\n")
            for k, v in best_config['positive'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nnegative:\n")
            for k, v in best_config['negative'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\ncalibrate:\n")
            for k, v in best_config['calibrate'].items():
                f.write(f"  {k}: {v}\n")

        print(f"💾 最佳配置已保存: {config_path}")

    print(f"{'='*80}\n")

    # 打印Top 5
    print("\n🏅 Top 5 配置:")
    print(df.head(5).to_string(index=False))

    return df, best_params, best_acc


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TDA+GR-CLIP 网格搜索')
    parser.add_argument('--dataset', type=str, default='caltech101',
                        help='数据集名称 (default: caltech101)')
    parser.add_argument('--backbone', type=str, default='ViT-B/16',
                        choices=['RN50', 'ViT-B/16'],
                        help='CLIP backbone (default: ViT-B/16)')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='配置目录 (default: configs)')
    parser.add_argument('--data-root', type=str,
                        default='E:/研究生/提前进组/提示学习方向/code/dataset/',
                        help='数据集根目录')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'grclip_only', 'tda_only'],
                        help='搜索模式 (default: quick)')
    return parser.parse_args()


def main():
    args = get_args()

    # 定义搜索空间
    if args.mode == 'quick':
        # 快速搜索: 只搜索最关键参数
        search_space = {
            'pos_alpha': [1.5, 2.0, 2.5],           # TDA正缓存权重
            'cal_enabled': [True, False],           # 是否启用GR-CLIP
            'cal_fusion_alpha': [0.8, 0.9],        # GR-CLIP融合权重
        }

    elif args.mode == 'grclip_only':
        # 只搜索GR-CLIP参数
        search_space = {
            'cal_enabled': [True],
            'cal_fusion_alpha': [0.3, 0.5, 0.7, 0.8, 0.9],
            'cal_confidence_threshold': [0.6, 0.7, 0.8, 0.9],
            'cal_min_cache_size': [10, 20, 30],
        }

    elif args.mode == 'tda_only':
        # 只搜索TDA参数 (禁用GR-CLIP)
        search_space = {
            'pos_shot_capacity': [2, 3, 4],
            'pos_alpha': [1.0, 1.5, 2.0, 2.5, 3.0],
            'pos_beta': [4.0, 5.0, 6.0],
            'neg_alpha': [0.1, 0.117, 0.15],
            'cal_enabled': [False],
        }

    elif args.mode == 'full':
        # 完整搜索 (非常耗时!)
        search_space = {
            'pos_shot_capacity': [2, 3, 4],
            'pos_alpha': [1.5, 2.0, 2.5],
            'pos_beta': [4.0, 5.0, 6.0],
            'neg_alpha': [0.1, 0.117],
            'cal_enabled': [True, False],
            'cal_fusion_alpha': [0.7, 0.8, 0.9],
            'cal_confidence_threshold': [0.6, 0.7, 0.8],
        }

    # 执行网格搜索
    grid_search(args, search_space)


if __name__ == "__main__":
    main()
