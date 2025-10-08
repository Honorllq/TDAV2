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
    """
    构建视觉原型
    """
    prototypes={}
    for class_id in cache.keys():
        features=[item[0].squeeze(0) for item in cache[class_id]]
        prototypes[class_id]=torch.stack(features,dim=0).mean(dim=0)
    return prototypes

def calibrate_text_features(visual_prototypes,clip_weights,beta=0.9):
    """
    校准文本特征
    """
    calibrated = {}
    for class_id,T_c in enumerate(clip_weights.T):
        if class_id in visual_prototypes:
            P_c=visual_prototypes[class_id]
            T_prime=T_c*beta+P_c*(1-beta)
            calibrated[class_id]=T_prime /T_prime.norm()
        else:
            calibrated[class_id]=T_c

    return torch.stack([calibrated[i] for i in sorted(calibrated.keys())],dim=1)

# def calibrate_text_features(visual_prototypes, clip_weights, alpha=0.05):
#     """alpha=0.05意味着只做5%的修正,保持文本结构"""
#     calibrated = {}
#     for class_id, T_c in enumerate(clip_weights.T):
#         if class_id in visual_prototypes:
#             P_c = visual_prototypes[class_id]
#             # 小幅残差修正
#             T_prime = T_c + alpha * (P_c - T_c)
#             calibrated[class_id] = T_prime / T_prime.norm()
#         else:
#             calibrated[class_id] = T_c
#     return torch.stack([calibrated[i] for i in sorted(calibrated.keys())], dim=1)
# def calibrate_text_features(visual_prototypes, clip_weights, alpha=0.1):
#     """
#     残差校准: T' = T + α * (P - T)
    
#     优势: 保持文本特征的主要结构,只做小幅调整
#     """
#     calibrated = {}
#     for class_id, T_c in enumerate(clip_weights.T):
#         if class_id in visual_prototypes:
#             P_c = visual_prototypes[class_id]
            
#             # 计算残差: 视觉原型相对文本的偏移
#             residual = P_c - T_c
            
#             # 残差投影: 只保留与T_c正交的部分 (避免改变T_c的主方向)
#             residual_orth = residual - (residual @ T_c) * T_c
            
#             # 小幅修正
#             T_prime = T_c + alpha * residual_orth
#             calibrated[class_id] = T_prime / T_prime.norm()
#         else:
#             calibrated[class_id] = T_c

#     return torch.stack([calibrated[i] for i in sorted(calibrated.keys())], dim=1)

def run_test_tda(pos_cfg, neg_cfg,calibrate_cfg, loader, clip_model, clip_weights):
    """
    TDA测试时适应主循环
    
    实现论文3.2节描述的核心TDA算法:
    1. 顺序处理每个测试样本(流式设置)
    2. 基于熵条件更新正负缓存
    3. 结合CLIP预测与缓存预测(公式7)
    
    最终预测: P_TDA = f_test @ W_c^T + P_pos + P_neg  (公式7)
    
    参数:
        pos_cfg: 来自YAML的正缓存配置
        neg_cfg: 来自YAML的负缓存配置  
        loader: 测试数据加载器(batch_size=1用于流式处理)
        clip_model: 预训练CLIP模型
        clip_weights: 文本嵌入W_c [d, num_classes]，d=512
    
    返回:
        float: 所有样本的最终测试精度
    """
    with torch.no_grad():
        # 初始化动态缓存和精度跟踪
        pos_cache, neg_cache, accuracies = {}, {}, []
        
        # 从配置文件提取超参数
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        # 测试时适应循环 - 顺序处理样本
        for i, (images, target) in enumerate(tqdm(loader, desc='已处理测试图像: ')):
            """
            单个测试样本处理流程:
            输入: images [1,3,224,224], target [1]
            输出: 更新的缓存 + 当前样本精度
            
            关键数据维度:
            - 图像特征: [1, 512] (RN50/ViT-B16都是512维)
            - CLIP logits: [1, num_classes]
            - 缓存键: [512, num_cached_samples] 
            - 缓存值: [num_cached_samples, num_classes]
            - 相似度: [1, num_cached_samples]
            - 最终预测: [1, num_classes]
            """
            
            # 步骤1: 从CLIP提取特征和预测
            # image_features: [1, d] 标准化CLIP图像特征，d=512
            # clip_logits: [1, num_classes] 原始CLIP预测  
            # loss: 预测熵(置信度度量)
            # prob_map: [1, num_classes] softmax概率
            # pred: 预测类别ID(伪标签)
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

            # 步骤2: 用高置信度预测更新正缓存
            if pos_enabled:
                # 无论熵如何都添加到正缓存(质量由缓存更新逻辑控制)
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            

            # 步骤3: 用中等不确定预测更新负缓存
            # 条件γ(f_test): τ_l < H(f_test @ W_c^T) < τ_h  (公式5)
            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                # 只包含中等不确定性的样本用于负学习
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            #校准文本特征
            if calibrate_cfg['enabled'] and len(pos_cache) >= calibrate_cfg['min_cache_size']:
                visual_prototypes = build_visual_prototypes(pos_cache)
                # calibrated_weights = calibrate_text_features(visual_prototypes, clip_weights, calibrate_cfg['beta'])
                calibrated_weights = calibrate_text_features(visual_prototypes, clip_weights,alpha=0.05)
            else:
                calibrated_weights = clip_weights

            if i%100==0:
                print(F.cosine_similarity(clip_weights, calibrated_weights))

            # 步骤4: 组合预测(论文公式7)
            # P_TDA(f_test) = f_test @ W_c^T + P_pos(f_test) + P_neg(f_test)
            #final_logits = clip_logits.clone()  # 从CLIP基线开始: f_test @ W_c^T [1, num_classes]

            final_logits = 100*image_features @ calibrated_weights
            
            # 添加正缓存贡献: + P_pos(f_test) [1, num_classes]
            # 正缓存增强高置信度预测，提供支持性证据
            if pos_enabled and pos_cache:
                pos_contribution = compute_cache_logits(image_features, pos_cache, 
                                                      pos_params['alpha'], pos_params['beta'], clip_weights)
                final_logits += pos_contribution  # 维度保持 [1, num_classes]
            
            # 减去负缓存贡献: + P_neg(f_test) = - (-P_neg) [1, num_classes]  
            # 负缓存抑制不确定预测中的错误倾向，提供反向证据
            if neg_enabled and neg_cache:
                neg_contribution = compute_cache_logits(image_features, neg_cache, 
                                                      neg_params['alpha'], neg_params['beta'], clip_weights, 
                                                      (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                final_logits -= neg_contribution  # 维度保持 [1, num_classes]

            # 步骤5: 评估预测并跟踪运行精度
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            wandb.log({"平均测试精度": sum(accuracies)/len(accuracies)}, commit=True)

            # 定期记录
            if i % 1000 == 0:
                print("---- TDA测试精度: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        # 最终结果
        print("---- TDA测试精度: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
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