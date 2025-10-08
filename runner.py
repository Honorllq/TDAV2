"""
TDA (训练自由动态适配器) 实现

本模块实现了论文"Efficient Test-Time Adaptation of Vision-Language Models"中描述的核心算法

论文核心概念:
- 动态适配器: 使用键值缓存的无训练机制
- 正缓存: 存储高置信度伪标签及对应特征
- 负缓存: 存储负伪标签以减少噪声影响  
- 测试时适应: 无需反向传播即可适应分布偏移

该方法通过避免梯度计算，在保持或提升精度的同时实现了相比TPT/DiffTPT的优异效率
"""

import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *

# VS Code调试配置
# import debugpy
# try:
#     # 启用远程调试用于开发
#     debugpy.listen(("localhost", 9508))
#     print("等待调试器连接")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def get_arguments():
    """获取测试时适应的命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='E:/研究生/提前进组/提示学习方向/code/dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """
    更新动态缓存，添加新的特征和熵损失
    
    实现论文3.2节描述的动态队列机制。缓存为每个类别维护k-shot样本，按预测熵排序。
    
    论文引用: "给定测试样本，TDA将基于熵条件添加/替换键值对以确保高质量伪标签"
    
    参数:
        cache (dict): 动态缓存 {类别ID: [(特征, 熵, 概率图), ...]}
        pred (int): 预测类别标签(伪标签)  
        features_loss (list): [图像特征, 熵损失, 概率图(可选)]
            - image_features: CLIP编码特征 [1, d] 其中d为特征维度(512)
            - entropy_loss: 预测熵 (标量) 
            - prob_map: 类别概率分布 [1, num_classes] (用于负缓存)
        shot_capacity (int): 每类最大样本数 (论文中的k)
        include_prob_map (bool): 是否包含概率图 (负缓存用)
    
    缓存结构:
        正缓存: [(特征, 熵), ...]  
        负缓存: [(特征, 熵, 概率图), ...]
    """
    with torch.no_grad():
        # 根据缓存类型(正/负)准备缓存项
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        
        if pred in cache:
            # 类别已存在于缓存中
            if len(cache[pred]) < shot_capacity:
                # 条件1: 样本数 < 最大容量k
                # 直接添加新的键值对到缓存
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                # 条件2: 样本数 = 最大容量k
                # 如果当前熵更低则替换最高熵项
                # 确保保留最有信心的预测
                cache[pred][-1] = item
            
            # 维护基于熵的排序(优先队列行为)
            # 低熵 = 高置信度 = 高优先级
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            # 该类别的第一个样本
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """
    使用存储的键值对计算基于缓存的logits
    
    实现论文公式3、6:
    - 正缓存: P_pos(f_test) = A(f_test @ Q_p^T) @ L_p  (公式3)
    - 负缓存: P_neg(f_test) = -A(f_test @ Q_n^T) @ L_n  (公式6)
    
    其中 A(z) = α * exp(-β(1-z)) 是来自Tip-Adapter的适配函数
    
    参数:
        image_features: 测试图像特征 [1, d] 来自CLIP编码器，d=512
        cache: 包含存储特征和标签的动态缓存
        alpha: 残差比率(α) - 缓存预测的权重因子  
        beta: 锐度比率(β) - 控制相似度锐化程度
        clip_weights: CLIP文本嵌入 [d, num_classes]
        neg_mask_thresholds: 负伪标签的(下界,上界)阈值
    
    返回:
        cache_logits: [1, num_classes] 来自缓存适配的logits
    
    数据流程:
        1. 提取缓存键Q和值L 
        2. 计算相似度: f_test @ Q^T [1, num_cached_samples]
        3. 应用适配函数A(相似度)
        4. 计算最终logits: A(相似度) @ L
    """
    with torch.no_grad():
        cache_keys = []    # 将存储Q_p或Q_n: 缓存的图像特征
        cache_values = []  # 将存储L_p或L_n: 伪标签
        
        # 为所有类别提取缓存特征(键)和标签(值)
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])  # image_features [1, d]
                if neg_mask_thresholds:
                    # 负缓存: 使用概率图进行负伪标签
                    cache_values.append(item[2])  # prob_map [1, num_classes]
                else:
                    # 正缓存: 使用类别索引作为one-hot标签
                    cache_values.append(class_index)  # class_id (标量)

        # 准备缓存键: Q^T 形状为 [d, num_cached_samples]
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        
        if neg_mask_thresholds:
            # 负缓存处理 (论文公式4)
            # L_n = -1[p_l < P(Q_n)] 其中p_l是阈值
            cache_values = torch.cat(cache_values, dim=0)  # [num_cached_samples, num_classes]
            
            # 应用负掩码: 值>阈值变为1，其他变为0
            # 这实现了不确定预测的负伪标签
            cache_values = (((cache_values > neg_mask_thresholds[0]) & 
                           (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            # 正缓存处理
            # 将类别索引转换为one-hot编码标签L_p
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), 
                                    num_classes=clip_weights.size(1))).cuda().half()

        # 计算测试特征和缓存特征之间的相似度
        # affinity = f_test @ Q^T 形状为 [1, num_cached_samples]
        affinity = image_features @ cache_keys
        
        # 应用适配函数 A(z) = α * exp(-β(1-z))
        # 这是核心的基于相似度的权重机制
        adaptation_weights = ((-1) * (beta - beta * affinity)).exp()
        
        # 计算最终缓存logits: A(相似度) @ L
        # 形状: [1, num_cached_samples] @ [num_cached_samples, num_classes] = [1, num_classes]
        cache_logits = adaptation_weights @ cache_values
        
        # 应用残差比率α来缩放缓存贡献
        return alpha * cache_logits
def build_visual_prototypes(cache):
    """构建视觉原型"""
    prototypes = {}
    for class_id in cache.keys():
        features = [item[0].squeeze(0) for item in cache[class_id]]
        prototypes[class_id] = torch.stack(features, dim=0).mean(dim=0)
    return prototypes

def compute_modality_means(pos_cache, clip_weights):
    """
    计算模态均值 (GR-CLIP核心)
    
    论文方法: 从校准集计算query/text/image的全局均值
    我们的实现: 从测试缓存动态计算 (更适合TDA场景)
    
    返回:
        text_mean: 文本特征均值 [512]
        image_mean: 图像特征均值 [512]
    """
    with torch.no_grad():
        # 从缓存提取所有特征
        all_image_features = []
        for class_id in pos_cache.keys():
            for item in pos_cache[class_id]:
                all_image_features.append(item[0].squeeze(0))  # [512]
        
        if len(all_image_features) == 0:
            # 缓存为空,返回零向量
            # ✅ 修复1: 动态获取特征维度 (RN50=1024, ViT-B/16=512)
            feature_dim = clip_weights.size(0)
            return torch.zeros(feature_dim).cuda(), torch.zeros(feature_dim).cuda()
        
        # 计算图像特征均值
        image_mean = torch.stack(all_image_features).mean(dim=0)  # [512]
        
        # 计算文本特征均值 (从CLIP权重)
        text_mean = clip_weights.mean(dim=1)  # [512]
        
        return text_mean, image_mean

def calibrate_with_gr_clip(visual_prototypes, clip_weights, text_mean, image_mean, alpha=0.7):
    """
    使用GR-CLIP方法校准: 去除模态间隙后融合

    步骤:
    1. 去中心化: 减去模态均值
    2. 标准化: L2归一化
    3. 融合: 在去偏后的空间中融合

    参数:
        visual_prototypes: 视觉原型 {class_id: [d]}
        clip_weights: 文本特征 [d, num_classes]
        text_mean: 文本均值 [d]
        image_mean: 图像均值 [d]
        alpha: 融合权重 (文本占比, 默认0.7)

    返回:
        calibrated_weights: 校准后的权重 [d, num_classes]
    """
    with torch.no_grad():
        num_classes = clip_weights.size(1)
        calibrated_weights = torch.zeros_like(clip_weights)  # [512, num_classes]
        
        for class_id in range(num_classes):
            # 提取原始特征
            T_c = clip_weights[:, class_id]  # 文本特征 [512]
            
            if class_id in visual_prototypes:
                V_c = visual_prototypes[class_id]  # 视觉原型 [512]
                
                # 🔑 GR-CLIP核心: 去中心化
                T_c_centered = T_c - text_mean
                V_c_centered = V_c - image_mean
                
                # 标准化
                T_c_centered = T_c_centered / T_c_centered.norm()
                V_c_centered = V_c_centered / V_c_centered.norm()

                # 融合 (现在在同一空间中)
                # ✅ 修复2: 使用传入的alpha参数而非硬编码
                fused = alpha * T_c_centered + (1 - alpha) * V_c_centered
                
                # 最终标准化
                calibrated_weights[:, class_id] = fused / fused.norm()
            else:
                # 无视觉原型,仅去中心化文本
                T_c_centered = T_c - text_mean
                calibrated_weights[:, class_id] = T_c_centered / T_c_centered.norm()
        
        return calibrated_weights


def run_test_tda(pos_cfg, neg_cfg, calibrate_cfg, loader, clip_model, clip_weights):
    """
    TDA + GR-CLIP集成版本
    """
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        # 初始化模态均值 (GR-CLIP)
        text_mean, image_mean = None, None
        calibrated_weights = clip_weights

        for i, (images, target) in enumerate(tqdm(loader, desc='已处理测试图像: ')):
            # 步骤1: 特征提取
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

            # 步骤2: 更新缓存
            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])
            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            # 步骤3: GR-CLIP校准 (定期更新)
            if calibrate_cfg['enabled'] and len(pos_cache) >= calibrate_cfg['min_cache_size']:
                if i % calibrate_cfg.get('update_interval', 100) == 0:
                    # 计算模态均值
                    text_mean, image_mean = compute_modality_means(pos_cache, clip_weights)

                    # 构建视觉原型
                    visual_prototypes = build_visual_prototypes(pos_cache)

                    # ✅ 修复2: 从配置读取fusion_alpha参数
                    fusion_alpha = calibrate_cfg.get('fusion_alpha', 0.7)

                    # GR-CLIP校准
                    calibrated_weights = calibrate_with_gr_clip(
                        visual_prototypes, clip_weights, text_mean, image_mean, alpha=fusion_alpha
                    )
                    
                    if i % 1000 == 0:
                        print(f"\n[GR-CLIP] Updated at step {i}")
                        print(f"  Text mean norm: {text_mean.norm():.4f}")
                        print(f"  Image mean norm: {image_mean.norm():.4f}")
                        print(f"  Modality gap: {(text_mean - image_mean).norm():.4f}")
                        print(f"  Fusion alpha: {fusion_alpha}")
                        print(f"  Cached classes: {len(pos_cache)}")

                        # 诊断: 计算校准前后的相似度变化
                        if len(visual_prototypes) > 0:
                            sample_class = list(visual_prototypes.keys())[0]
                            V_c = visual_prototypes[sample_class]
                            T_c = clip_weights[:, sample_class]
                            sim_before = (V_c @ T_c).item()

                            V_c_cal = (V_c - image_mean) / (V_c - image_mean).norm()
                            T_c_cal = (T_c - text_mean) / (T_c - text_mean).norm()
                            sim_after = (V_c_cal @ T_c_cal).item()

                            print(f"  Cross-modal similarity: {sim_before:.4f} → {sim_after:.4f} (gain: {sim_after-sim_before:+.4f})")

            # 步骤4: 自适应使用GR-CLIP (仅对低置信度样本)
            clip_baseline_logits = 100.0 * image_features @ clip_weights

            if text_mean is not None and image_mean is not None:
                # 计算CLIP基线的置信度
                clip_confidence = clip_baseline_logits.softmax(1).max().item()
                print(clip_confidence)
                confidence_threshold = calibrate_cfg.get('confidence_threshold', 0.9)

                if clip_confidence < confidence_threshold:
                    # 低置信度: 使用GR-CLIP校准
                    image_features_centered = image_features - image_mean.unsqueeze(0)
                    image_features_centered = image_features_centered / image_features_centered.norm(dim=1, keepdim=True)
                    gr_clip_logits = 100.0 * image_features_centered @ calibrated_weights
                else:
                    # 高置信度: 保持原始CLIP
                    gr_clip_logits = clip_baseline_logits
            else:
                # 缓存不足,使用原始CLIP
                gr_clip_logits = clip_baseline_logits

            # 步骤5: 计算TDA缓存贡献
            # ✅ 修复3: 使用原始特征保持TDA逻辑一致性 (分离GR-CLIP和TDA路径)
            tda_logits = torch.zeros_like(gr_clip_logits)
            if pos_enabled and pos_cache:
                tda_logits += compute_cache_logits(image_features, pos_cache,
                                                   pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                tda_logits -= compute_cache_logits(image_features, neg_cache,
                                                   neg_params['alpha'], neg_params['beta'], clip_weights,
                                                   (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

            # 步骤6: 融合GR-CLIP和TDA
            final_logits = gr_clip_logits + tda_logits

            # 步骤7: 评估
            acc = cls_acc(final_logits, target)
            accuracies.append(acc)
            wandb.log({"平均测试精度": sum(accuracies)/len(accuracies)}, commit=True)

            if i % 1000 == 0:
                print(f"---- TDA测试精度: {sum(accuracies)/len(accuracies):.2f}. ----")
        
        print(f"---- 最终TDA测试精度: {sum(accuracies)/len(accuracies):.2f}. ----\n")
        with open('outputs/result.txt', 'a') as f:
            f.write("Top1- {:.2f}\n".format(sum(accuracies)/len(accuracies)))
        return sum(accuracies)/len(accuracies)



def main():
    """
    运行TDA测试时适应实验的主函数
    
    该函数协调完整的TDA流水线:
    1. 加载和配置CLIP模型
    2. 顺序处理多个数据集  
    3. 为每个数据集运行TDA适应
    4. 记录结果到wandb进行实验跟踪
    
    命令行用法:
    python tda_runner.py --config configs/ --datasets I/A/V/R/S --backbone RN50 --wandb-log
    """
    args = get_arguments()
    config_path = args.config

    # 初始化CLIP模型(ResNet-50或ViT-B/16)
    # clip_model包含图像编码器E_v和文本编码器E_t
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()  # 设置为评估模式(无梯度计算)

    # 设置随机种子以确保可重复性
    random.seed(1)
    torch.manual_seed(1)

    # 初始化wandb实验跟踪
    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # 在每个数据集上顺序运行TDA
    # 支持OOD基准(I/A/V/R/S)和跨域基准
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"正在处理 {dataset_name} 数据集.")
        
        # 从YAML配置加载数据集特定的超参数
        cfg = get_config_file(config_path, dataset_name)
        print("\n运行数据集配置:")
        print(cfg, "\n")
        
        # 准备数据集和CLIP组件
        # test_loader: batch_size=1的DataLoader用于流式适应
        # classnames: 数据集的类别名称列表
        # template: 文本生成的提示模板
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        
        # 使用CLIP文本编码器生成文本嵌入W_c [d, num_classes]，d=512
        clip_weights = clip_classifier(classnames, template, clip_model)

        # 为此数据集初始化wandb运行
        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        # 运行TDA测试时适应算法
        # 处理所有测试样本后返回最终精度
        acc = run_test_tda(cfg['positive'], cfg['negative'],cfg['calibrate'], test_loader, clip_model, clip_weights)

        # 记录最终结果并清理
        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()