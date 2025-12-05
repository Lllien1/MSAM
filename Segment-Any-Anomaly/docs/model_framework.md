# 模型整体框架（Segment Any Anomaly）

本文档概述 `SAA/model.py` 中 `Model` 类的完整推理流程，并列出每个阶段涉及的模块与关键接口，方便快速理解或二次开发。

## 1. 组件总览

| 模块 | 代码位置 | 主要职责 | 关键接口 |
| --- | --- | --- | --- |
| **Grounding DINO** | `SAA/model.py` → `load_dino` | 根据文本提示产出候选框与 token logits | `build_model`, `anomaly_region_generator(captions=...)`, `tokenizer` |
| **SAM (Segment Anything)** | `SAA/model.py` → `self.anomaly_region_refiner` | 将候选框细化为像素级掩码 | `SamPredictor.set_image`, `predict_torch(boxes=...)` |
| **ModelINet / Visual Saliency** | `SAA/modelinet.py` | 生成自相似显著性图，衡量区域与对象的差异 | `ModelINet.forward`, `set_img_size`, `preprocess` |
| **Prompt 管理** | `SAA/hybrid_prompts.py` | 提供对象/缺陷/属性/过滤词提示 | `build_general_prompts`, `manul_prompts`, `property_prompts` |
| **图像预处理** | `SAA/model.py` → `self.transform` | 将输入转换为 Grounding DINO 需要的张量 | `T.Compose(RandomResize → ToTensor → Normalize)` |

## 2. 推理流水线（`Model.forward`）

1. **对象提示 (TGMP)**  
   - 调用 `ensemble_text_guided_mask_proposal`，使用 `self.object_prompt` 在 Grounding DINO 中定位主体。  
   - 产出对象掩码、置信度及最大框面积，为后续缺陷面积阈值提供尺度。

2. **缺陷提示 (`𝒫ᴸ`,`𝒫ˢ`)**  
   - 再次调用 `ensemble_text_guided_mask_proposal`，此时输入 `self.defect_prompt_list` 与 `self.filter_prompt_list`。  
   - 阈值 `defect_max_area = object_area * defect_area_threshold` 控制框面积。  
   - 结果：候选缺陷掩码 + 初始 logits。

3. **显著性提示 (`𝒫ˢ`)**  
   - `saliency_prompting`：  
     - 调用 `visual_saliency_calculation`（单实例走 `single_object_similarity`，多实例走 `region_feature_extraction`）。  
     - 调用 `rescore` 依据显著性图重新加权缺陷掩码得分。

4. **置信度提示 (`𝒫ᶜ`)**  
   - `confidence_prompting` 选取得分最高的 `k_mask` 个掩码，执行逐像素加权平均并插值到 `out_size` 得到最终异常图。

5. **附加输出**  
   - 返回 `(anomaly_map, {'similarity_map': similarity_map})`，供可视化或下游任务使用。

## 3. 关键子流程

### 3.1 文本驱动的候选生成 (`ensemble_text_guided_mask_proposal`)

1. **预处理**：`self.transform` 将 `PIL` 图像转换为 DINO 输入；初始化 `SamPredictor`（仅首帧）。  
2. **文本 → 框**：`text_guided_region_proposal` 调用 Grounding DINO `anomaly_region_generator`.  
3. **属性过滤**：`bbox_suppression` 应用五种约束：logits 阈值、面积上下限、文本匹配阈值、背景过滤词。  
4. **框 → 掩码**：`region_refine` 使用 SAM `predict_torch` 输出掩码集合。若无候选，回退到全零掩码。

### 3.2 显著性估计

- **单对象 (`object_number == 1`)**：`single_object_similarity` 通过 ModelINet (缩放到 256×256) 计算特征自相似度，生成热力图。  
- **多对象**：`visual_saliency_extractor` 在 1024×1024 上提特征，`region_feature_extraction` 提取当前对象与其余对象特征，计算最大相似度后取 `(1 - sim)` 作为异常得分。

### 3.3 置信度重加权

- `rescore`：对每个缺陷掩码，计算其在显著性图上的均值，乘以原始 `logit` 得到 `defect_rescores`。  
- `confidence_prompting`：仅保留最高的 `k_mask`（默认 3）个掩码，根据得分进行像素级融合，得到平滑的异常概率图。

## 4. Prompt 体系

- **对象提示**：`self.object_prompt`（默认“object”），用于定位主体。  
- **缺陷提示列表**：`self.defect_prompt_list = general_prompts + manual_prompts[dataset][class_name]`。  
- **过滤提示**：`self.filter_prompt_list` 与缺陷提示一一对应，过滤背景词。  
- **属性提示**：`property_prompts[dataset][class_name]` 用于面积/滤词配置（`set_property_text_prompts` 会更新 `object_max_area` 等阈值）。

## 5. 接口速查

| 功能 | 方法 | 输入 | 输出 |
| --- | --- | --- | --- |
| 加载 Grounding DINO | `load_dino(config, checkpoint, device)` | 配置路径、权重、设备 | DINO `nn.Module` |
| 生成候选掩码 | `ensemble_text_guided_mask_proposal(image, prompts, filters, area_max, area_min, box_thr, text_thr)` | BGR 图像 & 提示 | 掩码列表、logits、最大面积 |
| 显著性重排序 | `saliency_prompting(image, object_masks, defect_masks, defect_logits)` | 图像 & 掩码 | 重打分的掩码、显著性图 |
| 得分融合 | `confidence_prompting(defect_masks, defect_scores, similarity_map)` | 掩码与分数 | 归一化异常图 |

## 6. 开发者提示

- 若需要替换主干（如改用其它 CLIP/VLM），只需在 `load_dino` 或 `ModelINet` 初始化处调整。  
- 所有阈值（`box_threshold`, `text_threshold`, `defect_area_threshold`, `k_mask` 等）均在 `Model.__init__` 中配置，方便通过命令行参数注入。  
- `Model.forward` 无状态（除 `is_sam_set`），可直接部署于批量推理；但当前实现仅支持单张图像输入。

