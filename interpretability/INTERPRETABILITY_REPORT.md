# ChemMambaDDI (GASPDDI) 顶刊级多层级可解释性研究报告与论文素材库

本报告面向生物信息学与计算化学顶刊（*Briefings in Bioinformatics*, *Journal of Chemical Information and Modeling*, *Bioinformatics*, *Nature Machine Intelligence*），基于训练完成的 **Task 1（热启动 Pair-Split）**、**Task 2（单药冷启动 Seen–Unseen）** 与 **Task 3（双药全冷启动 Unseen–Unseen）** 模型权重（`no_unimol` 与 `unimol`），对 `ChemMambaDDI` 进行了从**原子/子结构层**、**指纹特征层**、**对称算子与冷启动泛化层**、**86类药理流形层**到 **3D 蒸馏负迁移机理诊断层**的五维度完整可解释性剖析。

---

## 1. 维度一：原子/子结构级药效团显著性映射与 Mamba 门控动力学 (Figure 1 & Table 1)

![Figure 1: Clinical DDI Case Studies & Substructure Saliency](figures/fig1_case_studies_substructure_saliency.png)

*数据表：`tables/table1_clinical_case_studies.csv`*

- **Case A (`idx=169450`, $Y=73$, 肾小管 OAT1/OAT3 排泄竞争抑制)**：`Methotrexate (DB00563)` + `Ibuprofen (DB01050)` ($P_{\text{No3D}}=0.9819$)。模型在两分子上同时将 **`MACCS#123`（羧酸基团 `O-C-O`）** 识别为第一顺位正贡献药效团（甲氨蝶呤谷氨酸双羧基尾 `+0.356`，布洛芬丙酸羧基 `+0.410`），精准复现了有机阴离子转运体的竞争性阻断机制。
- **Case B (`idx=49918`, $Y=47$, 肝脏 CYP2C9 代谢竞争性抑制)**：`Sulfamethoxazole (DB01015)` + `Fluconazole (DB00196)` ($P_{\text{No3D}}=0.9547$)。精准高亮氟康唑配位 CYP2C9 血红素铁的 **1,2,4-三氮唑环（`MACCS#95/#79`, `+0.163`）** 与叔醇羟基（`MACCS#139`, `+0.130`），以及磺胺甲恶唑的**芳香磺胺核心（`MACCS#133`, `+0.175`）**与异恶唑环（`MACCS#65`, `+0.158`）。
- **Case C (`idx=21856`, $Y=20$, 心肌 hERG 通道与 QTc 延长协同毒性)**：`Fluoxetine (DB00472)` + `Ciprofloxacin (DB00537)` ($P_{\text{No3D}}=0.9383$)。高亮氟西汀的甲氨基丙胺侧链（`MACCS#93/#111`, `+0.312`）与苯氧醚键（`MACCS#143`, `+0.221`），以及环丙沙星的 C6-氟原子（`MACCS#134`, `+0.256`）、喹诺酮酮酸核（`MACCS#146`, `+0.244`）与哌嗪环（`MACCS#100`, `+0.203`）。
- **1D Mamba 选择性扫描门控（$\bar{\Delta}_t$）**：在脂肪碳链处保持低水平记忆保留，而在高电负性杂原子（`O`, `N`, `F`）、羰基双键与环闭合位点产生显著脉冲，与 1D-CNN（$k=7$）局部官能团检测器形成双通路互补。

---

## 2. 维度二：全局与机制特异性的 2D 化学指纹归因谱 (Figure 2 & Table 2)

![Figure 2: Global 2D Fingerprint Bit Attribution](figures/fig2_fingerprint_bit_attribution.png)

*数据表：`tables/table2_fingerprint_bit_importance.csv`*

- **跨任务 Top MACCS 药效团**：`MACCS#145`（双环/多环六元环骨架，Mean $|IG|=0.0543$）、`MACCS#125`（多芳香环体系，`0.0527`）、`MACCS#142`（多氮杂环/胺基，`0.0520`）、`MACCS#164`（含氧氢键位点，`0.0500`）、`MACCS#158`（`C-N` 胺/酰胺键，`0.0472`）。
- **跨任务 Top Morgan ECFP4 圆形位点**：`Bit #214`（杂芳环共轭环境，`0.0562`）、`Bit #337`（羰基/酰胺中心，`0.0496`）、`Bit #138`（羧基/羟基末端氧，`0.0464`）。

---

## 3. 维度三：对称融合算子解耦与冷启动化学相似度边界 (Figure 3 & Tables 3a–3c)

![Figure 3: Modality Share, Symmetric Operators & Cold-Start Generalization](figures/fig3_modality_and_operator_decomposition.png)

*数据表：`tables/table3a_fusion_block_attributions.csv`, `tables/table3b_counterfactual_operator_ablations.csv`, `tables/table3c_coldstart_similarity_bins.csv`*

- **反事实算子消融**：移除绝对差算子 $|u_a - u_b|$ 导致 Task 1 AUC 从 `0.9950` 跌至 `0.9092`（F1 从 `0.9755` 跌至 `0.8334`），证明药理不对称互补关系主要由差分算子编码；哈达玛积 $u_a \odot u_b$ 则在冷启动中提升共享靶点识别（Task 2 F1 `+5.84%`）。
- **冷启动 Tanimoto 相似度缩放定律与动态阈值拯救**：在 Task 2 中，测试集 AUC 从全新骨架区间（$S_{\max} < 0.25$）的 `0.6180` 单调上升至高同源区间（$S_{\max} > 0.55$）的 `0.8945`。验证集动态阈值校准（$\tau^* \approx 0.31$）有效补偿了陌生分子骨架上的概率压缩效应，在低相似度区间将 F1 提升 **`+15.8% ~ +27.3%`**。

---

## 4. 维度四：DDI 隐空间交互流形与 86 类关系层次聚类 (Figure 4 & Table 4)

![Figure 4: Pairwise Latent Manifold & 86-Relation Pharmacological Hierarchy](figures/fig4_latent_manifold_and_relation_hierarchy.png)

*数据表：`tables/table4_superfamily_performance.csv`*

- **无监督药理分类学重构**：对分类器最后一层权重矩阵（`86 × 512`）进行余弦相似度 Ward 聚类，模型自发将 86 类关系聚合为 **PK 肝脏代谢簇（`Y04, Y47`）**、**PK 排泄与血药浓度簇（`Y03, Y73, Y75, Y11`）**、**PD 心血管与 QTc 簇（`Y20, Y10, Y60, Y25, Y15`）** 及 **PD 凝血出血簇（`Y06, Y66`）**。
- **超家族冷启动差异**：在 Task 2/3 冷启动中，PD 中枢神经（`AUC = 0.8585 / 0.7932`）与 PD 心血管（`0.8584 / 0.7288`）泛化性明显优于 PK CYP450 代谢（`0.7797 / 0.6441`），揭示了酶活性口袋对全新骨架立体结构的严苛敏感性。

---

## 5. 维度五：为什么 3D Uni-Mol 蒸馏（`unimol`）反而不如 `no_unimol`？ (Figure 5 & Table 5)

![Figure 5: Quantitative Diagnosis of Why 3D Uni-Mol Distillation Degrades DDI Performance](figures/fig5_why_3d_distillation_fails_diagnosis.png)

*数据表：`tables/table5_3d_distillation_diagnosis.csv`*

1. **损失量级失衡导致梯度劫持（Panel A）**：InfoNCE 对齐损失（$\tau=0.1$）在 Task 1–3 全周期均值达到 Focal DDI 损失的 **`1.37x`、`0.99x`、`1.16x`**（早期最高超 `10.8x`），喧宾夺主地主导了 1D Mamba 的反向传播梯度。
2. **静态单构象 3D 空间与真实 DDI 邻域拓扑失配（Panel B）**：在 10-NN 药物邻域图中，静态 Uni-Mol 3D 教师表征的真实 DDI 伙伴 Jaccard 一致性仅为 **`0.159`**（并将受蒸馏的 1D Mamba 拉低至 **`0.235`**），而无 3D 蒸馏的任务驱动表征高达 **`0.413`（高出 `2.60 倍`）**。
3. **3D 构象生成局限与模态正则化不对称（Panels C & D）**：DrugBank 中有 `99/1,706`（`5.8%`）的药物（多肽、大环等）无法生成标准 RDKit `ETKDGv3` 3D 构象；同时单一真空 MMFF94 构象无法代表生理结合构象，强行蒸馏反而破坏了高信噪比 2D 药效团特征的判别边界。

