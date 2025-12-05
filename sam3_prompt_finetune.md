## 目标与约束

- 基于 SAM3，严格遵循 `sam3/README_TRAIN.md` 的数据组织、增广、优化器、调度和损失，只在文本侧加入“固定模板 + 每类可学习 token”。
- 不使用 CLIP 投影，不引入 FiLo 的 HQ-Loc/属性提示；文本编码器使用 SAM3 原生接口。
- 任务：MVTec-AD 像素级分割微调（15 类）。

## 模块设计（贴合 SAM3 接口）

### 可学习文本提示
- 每类维护 K 个可学习 token：`P_c = {p_c^1, ..., p_c^K}`，维度与 SAM3 文本嵌入一致。
- 前向（单样本类别 c）：
  - 模板序列：`T_c`
  - 模板嵌入：`E_T = TextEmbed(T_c) ∈ R^{L×D}`
  - 可学习 token：`P_c ∈ R^{K×D}`
  - 拼接：`E = concat(E_T, P_c) → TextEncoder(E, attn_mask)`
  - 输出：文本提示特征 `H_c`，送入 SAM3 解码器。
- 直观：模板提供稳定语义锚点，可学习 token 自适应类内纹理/外观。

### 损失（遵循 README_TRAIN 组合）
- 掩码 logits：`M_hat`，标注掩码：`M ∈ {0,1}`。
- 分割损失：`L = λ_dice·Dice(M_hat, M) + λ_ce·CE/BCE(M_hat, M)`；λ 与 README 保持一致（若 README 只用 CE/MaskCE，则沿用原配方）。
- Dice（soft，前景=1）：`Dice = 1 - (2·∑(M_hat·M)+ε)/(∑M_hat + ∑M + ε)`.
- 多类别批次：每样本按其 class_id 选择对应模板与 `P_c`。

### 训练策略（与 SAM3 流程对齐）
- 冻结：SAM3 主干与文本编码器，默认也冻结大部分解码层；先只训练可学习 token。若验证不足，再解冻解码尾端 1–2 层（lr 调低 0.1×）。
- 优化器/调度：沿用 README（如 AdamW + warmup + cosine/poly/step）；可学习 token 放单独 param group，lr 1e-3~3e-4，weight decay 0~1e-4。其余参数组保持 README 默认。
- 数据与增广：完全遵守 README 的 resize、翻转、尺度抖动、颜色策略；不额外加入重色扰动。
- 训练轮次与 batch：依 README；显存不足可用梯度累积。

## 关键代码草案（与 SAM3 接口一致）
```python
# 假设 sam3_model(images, text_embeds, text_attn_mask) → {"masks": logits}
class PromptLearner(nn.Module):
    def __init__(self, num_classes, embed_dim, K, templates):
        super().__init__()
        self.templates = templates  # list[str], len = num_classes
        self.K = K
        self.token = nn.Parameter(torch.zeros(num_classes, K, embed_dim))
        nn.init.normal_(self.token, std=0.02)

    def build_text(self, class_ids, tokenizer, text_encoder):
        embeds, masks = [], []
        for cid in class_ids:  # per sample
            tpl = self.templates[cid]
            tpl_ids = tokenizer(tpl).to(self.token.device)         # [L]
            tpl_embed = text_encoder.embed_tokens(tpl_ids)         # [L, D]
            p = self.token[cid]                                    # [K, D]
            e = torch.cat([tpl_embed, p], dim=0)                   # [L+K, D]
            embeds.append(e)
        max_len = max(e.size(0) for e in embeds)
        for i, e in enumerate(embeds):
            pad = max_len - e.size(0)
            if pad > 0:
                embeds[i] = torch.cat([e, torch.zeros(pad, e.size(1), device=e.device)], dim=0)
            masks.append(torch.tensor([1]*e.size(0) + [0]*pad, device=e.device))
        return torch.stack(embeds), torch.stack(masks)

# 训练主循环（保持 README 的优化器/调度/损失组合）
prompt = PromptLearner(num_classes=15, embed_dim=TEXT_DIM, K=3, templates=tpl_list).to(device)
param_groups = [
    {"params": prompt.parameters(), "lr": 1e-3, "weight_decay": 0.0},
    # 其他 SAM3 参数组按 README 设置；若解冻尾端层，lr 设为主干 lr 的 0.1×
]
optimizer = AdamW(param_groups, betas=(0.9, 0.999))  # 其余配置沿用 README

for images, masks, class_ids in loader:
    text_embeds, attn_mask = prompt.build_text(class_ids, tokenizer, text_encoder)
    out = sam3_model(images, text_embeds=text_embeds, text_attn_mask=attn_mask)
    logits = out["masks"]
    loss = dice_loss(logits, masks)*λ_dice + ce_or_bce_loss(logits, masks)*λ_ce
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
```

## 超参数建议
- K（每类可学习 token 数）：2–4；若过拟合降至 1–2。
- λ_dice / λ_ce：与 README 默认保持一致（常见 0.7/0.3），若 README 有不同配方，完全照抄。
- lr warmup：沿用 README（如 200–500 step）+ cosine/poly；prompt 组 lr 比主干高一个量级。
- 文本增强：多模板随机抽取，等价于文本 dropout，提升泛化。

## MVTec-AD 固定模板（短、名词化，贴合 SAM3 短语习惯）
- 类别占位符 `{cls}`：bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper。
- 避免长句，保持 <6–8 token，尽量名词/短语。

### 套餐 S1（最短对偶）
- 正常：`"{cls} normal"`
- 缺陷：`"{cls} defect"`

### 套餐 S2（纹理导向）
- 正常：`"{cls} smooth"`
- 缺陷：`"{cls} scratch"`

### 套餐 S3（完好/损伤）
- 正常：`"{cls} intact"`
- 缺陷：`"{cls} damaged"`

### 套餐 S4（局部区域）
- 正常：`"{cls} clean area"`
- 缺陷：`"{cls} defective area"`

### 套餐 S5（中英混合）
- 正常：`"正常 {cls}"`
- 缺陷：`"{cls} 缺陷"`

使用建议：
- 选 1–2 套作为主模板，其余用于文本增强（同一图随机挑一句）。
- 对少见词（如 metal_nut）可在 tokenizer 中配置别名 `"metal nut"`，保持分词友好。
- 训练与推理使用同一模板集合，避免分布漂移。

## 训练与验证步骤（遵循 SAM3 官方流程）
1) 数据与增广：完全按 `sam3/README_TRAIN.md` 准备（长边 resize、翻转、尺度抖动、颜色策略），masks 转 float 0/1。
2) 初始化 PromptLearner（15 类，K=2–4）与模板列表（如 S1+S3）。
3) 冻结主干/文本编码器/大部分解码层；仅训练 prompt 参数（param group 高 lr）。跑 5–10 epoch 观察验证 Dice/IoU。
4) 若收敛不足或小缺陷漏检，解冻解码尾端 1–2 层（lr 下降 0.1×），其余设置不变。
5) 评估：Dice/IoU per-class；若需 anomaly 级别指标，可对掩码 logits 做 max/mean 得到图级分数，再算 PRO/AUROC。
6) 推理：按类选模板 + 对应可学习 token，输出掩码；阈值策略沿用 README（如 0.5 或 Otsu）。

## 可能的变体与动机
- 减少 K：小数据防过拟合；观察 train/val gap 决定。
- 轻量正则：对可学习 token 加 L2（如 1e-4）限制漂移。
- 文本 dropout：多模板随机抽取，提升跨图泛化。
- 仅当小缺陷持续漏检时，再考虑在解码尾端加 1 层轻量 cross-attn（文本→最后一层特征），不动主干。
