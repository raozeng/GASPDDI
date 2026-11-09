# Figure 6 重画代码

运行：`python redraw_figure6.py`，依赖Pillow、Matplotlib（仅用于原字体定位）。不需要训练、t-SNE或聚类依赖。

按已确认方案，删除顶部总标题及其留白；B恢复20条完整关系标签，不再使用[:38]，加宽标签区；A3的Y60图例改为Increased Hypotensive Activity；B的Y06改为Increased Anticoagulant Activity。CYP Induction、CYP Inhibition、Y73的Excretion/PK括注完整保留，B原标题和20 Major DrugBank Relations副标题保留。

assets直接复用原300 dpi图片内的A1–A3图形、B热图、C整幅图。A3仅覆盖Y60图例文字区域，保留紫色标记及其他图例行；B只替换原纵轴文字。散点位置、轮廓系数、热图矩阵及顺序、C柱值不重新计算。data记录B的原关系顺序及完整标签。

输出：`../../figures/fig4_latent_manifold_and_relation_hierarchy_v2.png`，6000×3040，300 dpi。脚本检查原代码、PNG/PDF哈希和B/C图形像素一致性；来源、素材裁剪坐标和输出哈希见source_manifest.json。原文件和TeX引用保留，未生成新PDF。
