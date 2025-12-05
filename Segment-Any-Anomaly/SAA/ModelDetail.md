# ModelDetail.md
面向在 Segment-Any-Anomaly (SAA) 仓库内深化 `model.py` 的模型架构速查指南。

## 论文模块

### 1. 异常区域生成器（Anomaly Region Generator，ARG）
- **论文定位**：Fig.1、Sec.3.1 中提出的 “prompt-guided object detection foundation model”，通过语言提示生成候选框。
- **代码映射**：`Model.load_dino`/`get_grounding_output` (`model.py:80-125`)、`ensemble_text_guided_mask_proposal` (`model.py:173-244`) 负责加载 Grounding DINO、执行随机缩放标准化、并批量调用 `text_guided_region_proposal` (`model.py:249-264`)。  
- **要点**：  
  1. `set_ensemble_text_prompts` (`model.py:128-146`) 将论文中的 class-agnostic / class-specific prompts 解析为 `(defect_prompt, filter_prompt)`；其源头即论文提供的专家语言提示，仓库中由 `hybrid_prompts.py` 与 `prompts/*.py`（如 `prompts/mvtec_parameters.py`）维护。  
  2. `bbox_suppression` (`model.py:267-343`) 即论文所谓 “object property prompts (P^P)”——利用 box/logit 阈值与面积过滤器将属性约束编码为显式规则，而非继续堆叠语言提示。  
  3. `set_property_text_prompts` (`model.py:148-171`) 解析 PDF 中描述的 “object count / area” 先验（如对象数、k-mask、面积比例等），驱动 ARG 将大候选区域削减至论文设定的范围。  

### 2. 异常区域细化器（Anomaly Region Refiner，ARR）
- **论文定位**：Fig.1 中 “prompt-guided segmentation foundation model” 模块，即 SAM。  
- **代码映射**：`self.anomaly_region_refiner = SamPredictor(build_sam(...))` (`model.py:58-64`) 负责实例化；`region_refine` (`model.py:345-372`) 把 ARG 输出的框转为像素级掩码，是论文中“从 RB 到 R” 的具体实现。  
- **要点**：`region_refine` 将 `boxes_filtered` 送入 `SamPredictor.predict_torch`，对应论文所述的“mask decoder accepts bounding boxes as prompts”机制，并承担 raise-case：若无候选则返回全零掩码，保证流水线可继续往下传递。

### 3. 混合提示（Hybrid Prompt Regularization）
混合提示正是论文标题强调的贡献，代码按提示类型拆解为4段：

| 提示类型 | 论文记号 | 代码锚点 | 说明 |
| --- | --- | --- | --- |
| 语言提示 | $\mathcal{P}^L$ (class-agnostic + class-specific) | `set_ensemble_text_prompts` (`model.py:128-146`)、`hybrid_prompts.py`、`prompts/*.py` | 语言提示组合 `(det_prompt, filter_prompt)`，既包含通用 “defect on …” 也包含专家短语（如 `mvtec_parameters.py` 中的 “broken part. contamination.”）。 |
| 属性提示 | $\mathcal{P}^P$ | `set_property_text_prompts` (`model.py:148-171`)、`bbox_suppression` (`model.py:267-343`) | 解析对象数量、面积阈值、top-k 掩码数等，并以逻辑过滤的方式约束候选框（与论文 “object property prompts formulated as rules” 对应）。 |
| 显著性提示 | $\mathcal{P}^S$ | `saliency_prompting` (`model.py:375-421`)、`single_object_similarity` (`model.py:397-430`)、`visual_saliency_calculation` (`model.py:432-493`)、`region_feature_extraction` (`model.py:494-532`) | 复现论文 Sec.3.3 “visual saliency regularization”：使用 `ModelINet` (`modelinet.py`) 提取特征，针对单/多实例分别计算自相似热图，再调用 `rescore` (`model.py:533-558`) 重新加权缺陷分数。 |
| 置信度提示 | $\mathcal{P}^C$ | `confidence_prompting` (`model.py:559-589`) | 对应论文中 “confidence ranking” 部分，依据显著性重排后的分数挑选前 `k_mask` 个掩码，以加权平均的方式生成最终异常热图。 |

这四种提示与论文 Fig.2 的 “domain expert knowledge + target image context” 一一呼应：`hybrid_prompts.py`/`prompts/*` 承载专家知识，`saliency_prompting`/`confidence_prompting` 则直接作用于目标图像上下文。

## 论文架构

> 目标：为开发者概览 `Model.forward` (`model.py:591-635`) 的模块顺序，可据此封装接口或插拔新功能。

1. **提示初始化阶段**  
   - 数据集入口调用 `set_ensemble_text_prompts` / `set_property_text_prompts`，并从 `hybrid_prompts.py` 载入相应语言＆属性配置。  
   - 这一阶段决定了 ARG/ARR 的输入（语言）与硬阈值（属性），等价于论文在推理前配置 “hybrid prompts”。

2. **对象级 TGMP（Text-Guided Mask Proposal）**  
   - `forward` 首先以 `[self.object_prompt]` 调用 `ensemble_text_guided_mask_proposal`，得到 `object_masks` 与对象面积 (`model.py:601-612`)。  
   - 这是论文流程中的 “retrieve coarse anomaly/object regions via ARG” 步骤，同时产出的面积被用来自适应调整缺陷阈值。

3. **缺陷级提案与属性约束**  
   - 再次调用 `ensemble_text_guided_mask_proposal`，但这次输入 `self.defect_prompt_list` / `self.filter_prompt_list`，并使用 `defect_max_area = object_area * defect_area_threshold` (`model.py:614-626`)。  
   - 这一部分集成了语言提示 $\mathcal{P}^L$ 与属性提示 $\mathcal{P}^P$，对应论文 Fig.2 中 “hybrid prompts at proposal stage”。

4. **显著性重加权**  
   - `saliency_prompting` (`model.py:627-634`) 以对象掩码做上下文，调 `visual_saliency_extractor` 计算自相似度，随后在 `rescore` 中对缺陷分数做指数放大/抑制。  
   - 该模块即论文 “visual saliency & similarity regularization”，若需要替换 backbone，可在 `modelinet.py` 中扩展 `ModelINet` 或调整 `ResizeLongestSide` 缩放策略。

5. **置信度融合输出**  
   - `confidence_prompting` 将显著性加权后的分数排序，选出 `k_mask` 个掩码加权平均，并在 `cv2.resize` 后输出 `self.out_size` 的异常热图 (`model.py:559-589`)。  
   - `forward` 返回 `(anomaly_map, {'similarity_map': similarity_map})`，既提供最终结果也暴露调试所需的显著性附录，便于开发者插入新的评分/可视化接口。

### 接口封装建议
- **替换/扩展提示**：直接编辑 `prompts/*.py` 并在 `hybrid_prompts.py` 注册新条目；若需要运行期热更新，可在调用 `Model` 前准备动态的 `(def_prompt, filter_prompt)` 列表传入 `set_ensemble_text_prompts`。  
- **聚焦某一模块**：  
  1. 希望单独复用 ARG，可调用 `ensemble_text_guided_mask_proposal` 取得框 / mask / logits；  
  2. 若只关心显著性图，可直接调用 `visual_saliency_calculation`，输入对象掩码即可输出热图。  
- **多模态扩展**：论文提出“foundation model assembly”思想，若要替换成其他检测/分割模型，只需保持接口 `(boxes, logits)` → `region_refine` 的输入契约，即可无缝衔接后续提示模块。

通过本文件即可同时复现论文模块与仓库实现的映射，也能在后续工程中快速定位可插拔的功能段落。
