# Figure 5 重画代码

运行：`python redraw_figure5.py`，依赖NumPy、Matplotlib、Pillow。

去掉顶部总标题；A用原CSV重新绘制相同横柱，16条MACCS完整描述按需换行，增加标签区和行间空间；B保留原热图、数值、色条及旋转横轴标签，重新放置完整关系标签；C复用原PDF内嵌六幅分子图，仅将标题Mean |IG|改成Mean |Grad × Bit|、半径改成r≤2，原分数、Motif及Prevalence保留。长标题分行，未扩大化学高亮。

assets保存原分子素材及原热图区域，data保存原CSV逐字节副本和原关系标签。热图为原图像素材，未将两位小数标注当作完整原始数据重新推算。图形素材准备时只读提取原PDF，未新建或编辑PDF。

输出：`../../figures/fig2_fingerprint_bit_attribution_v2.png`，7340×4000，300 dpi。脚本校验原代码、PNG/PDF与CSV哈希不变，校验B的矩阵、色条及右侧轴标签像素不变；来源信息见source_manifest.json。原文件保留，TeX引用未切换。


用户后续排版调整：A/B标题改成单行；C卡片标题压缩为两行，将Prevalence并入第二行。A/B纵向间隔缩小，画布由6600×4450调整为7340×4000；六幅分子素材等比例放大到1400像素宽，使用Lanczos显示缩放，不改变结构或高亮范围。保留完整纵轴描述，不缩短标签。
