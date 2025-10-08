import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    """
    将熵归一化到[0,1]范围用于阈值比较
    
    参数:
        loss: 原始熵值(标量张量)
        clip_weights: 文本嵌入 [d, num_classes] 用于获取类别数
    
    返回:
        float: [0,1]范围内的归一化熵，其中1=最大不确定性
    
    用于负缓存过滤条件(公式5):
    γ(f_test): τ_l < H(f_test @ W_c^T) < τ_h
    """
    max_entropy = math.log2(clip_weights.size(1))  # log2(类别数)
    return float(loss / max_entropy)


def softmax_entropy(x):
    """
    计算softmax熵: H(p) = -Σ p_i * log(p_i)
    
    参数:
        x: logits张量 [batch_size, num_classes]
        
    返回:
        熵值 [batch_size] - 更高的值表示更多不确定性
        
    这测量预测置信度 - 低熵 = 高置信度
    用于缓存更新决策和基于置信度的样本选择
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    """
    计算多个增强视图的平均熵
    
    当处理同一图像的多个增强视图时使用，以获得鲁棒的熵估计
    用于带增强的OOD数据集
    
    参数:
        outputs: 多个视图的logits [num_views, num_classes]
        
    返回:
        跨视图的平均熵(标量)
    """
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    """
    计算分类精度
    
    参数:
        output: 模型输出logits [batch_size, num_classes] 
        target: 真实标签 [batch_size]
        topk: top-k精度，默认为1(top-1精度)
    
    返回:
        acc: 百分比精度值(0-100)
    """
    pred = output.topk(topk, 1, True, True)[1].t()  # 获取top-k预测 [topk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 检查预测是否正确
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]  # 转换为百分比
    return acc


def clip_classifier(classnames, template, clip_model):
    """
    使用CLIP文本编码器为所有类别生成文本嵌入(W_c)
    
    这创建了用于核心CLIP预测的文本嵌入W_c:
    f_test @ W_c^T 其中f_test是图像特征，W_c是文本嵌入
    
    参数:
        classnames: 类别名称列表 [num_classes]
        template: 提示模板列表 (例如 ["a photo of a {}"])
        clip_model: 带有文本编码器E_t的预训练CLIP模型
        
    返回:
        clip_weights: 文本嵌入W_c [d, num_classes] 其中d是嵌入维度(512)
        
    数据流程:
        1. 对每个类别: 使用模板生成提示
        2. 标记化提示并用CLIP文本编码器E_t编码  
        3. 跨模板平均嵌入(提示集成)
        4. L2标准化嵌入
        5. 堆叠成最终权重矩阵
    """
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # 准备类别名称(将下划线替换为空格)
            classname = classname.replace('_', ' ')
            
            # 使用模板生成提示(提示集成技术)
            # 例如 "a photo of a dog", "a picture of a dog" 等
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()  # 形状: [num_templates, seq_len]
            
            # 使用CLIP文本编码器E_t编码提示
            class_embeddings = clip_model.encode_text(texts)  # [num_templates, d] d=512
            
            # L2标准化单个嵌入
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            # 跨模板平均以获得鲁棒表示(提示集成)
            class_embedding = class_embeddings.mean(dim=0)  # [d]
            
            # 最终L2标准化
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        # 堆叠所有类别嵌入: [d, num_classes]
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    """
    为测试样本提取CLIP预测和特征
    
    该函数处理单图像和增强图像批次，实现基于置信度的样本选择以获得鲁棒预测
    
    参数:
        images: 输入图像 - 单个 [1,3,224,224] 或批次 [batch_size,3,224,224]  
        clip_model: 带图像编码器E_v的预训练CLIP模型
        clip_weights: 文本嵌入W_c [d, num_classes] d=512
        
    返回:
        image_features: 标准化CLIP图像特征 [1, d]
        clip_logits: CLIP预测 [1, num_classes] 
        loss: 预测熵(标量) - 置信度度量
        prob_map: Softmax概率 [1, num_classes]  
        pred: 预测类别ID(整数) - 伪标签ĥ
        
    数据流程:
        1. 用CLIP图像编码器E_v编码图像
        2. 计算logits: 100 * f @ W_c^T (温度缩放)
        3. 对于多视图: 选择最有信心的10%并平均
        4. 计算熵用于缓存更新决策
    """
    with torch.no_grad():
        # 处理不同输入格式
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        # 使用CLIP图像编码器E_v提取和标准化图像特征
        image_features = clip_model.encode_image(images)  # [batch_size, d] d=512
        image_features /= image_features.norm(dim=-1, keepdim=True)  # L2标准化

        # 计算CLIP logits: 与文本嵌入的缩放点积
        # 100.0是CLIP中使用的温度缩放因子
        clip_logits = 100. * image_features @ clip_weights  # [batch_size, num_classes]

        if image_features.size(0) > 1:
            # 多增强视图情况(用于OOD数据集)
            # 选择最有信心的10%视图进行鲁棒预测
            
            # 计算每个视图的熵(更低=更有信心)
            batch_entropy = softmax_entropy(clip_logits)  # [batch_size]
            
            # 选择top 10%最有信心的视图(最低熵)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]  # [selected_size, num_classes]
            
            # 跨选定视图平均特征和logits
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)  # [1, d]
            clip_logits = output.mean(0).unsqueeze(0)  # [1, num_classes]

            # 从选定视图计算最终度量
            loss = avg_entropy(output)  # 跨视图的平均熵
            prob_map = output.softmax(1).mean(0).unsqueeze(0)  # [1, num_classes]
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            # 单图像情况(标准测试时适应)
            loss = softmax_entropy(clip_logits)  # 标量熵
            prob_map = clip_logits.softmax(1)  # [1, num_classes]
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])  # 预测类别

        return image_features, clip_logits, loss, prob_map, pred


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r', encoding='utf-8') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template