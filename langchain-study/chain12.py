import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm

# Use a CJK-capable font so Chinese characters render correctly
_CJK_FONTS = ['Noto Sans CJK JP', 'Noto Serif CJK JP',
               'Noto Sans CJK SC', 'AR PL UMing CN', 'WenQuanYi Micro Hei']
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _CJK_FONTS:
    if _font in _available:
        matplotlib.rcParams['font.family'] = _font
        break

def draw_react_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')
    fig.patch.set_facecolor('#F0F4F8')
    ax.set_facecolor('#F0F4F8')

    # ── helper functions ──────────────────────────────────────────────
    def rounded_box(ax, cx, cy, w, h, color, text, fontsize=11,
                    text_color='white', radius=0.35, bold=False,
                    sub_text=None):
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                             boxstyle=f"round,pad={radius}",
                             facecolor=color, edgecolor='white',
                             linewidth=2, zorder=3)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        y_text = cy if sub_text is None else cy + 0.15
        ax.text(cx, y_text, text, ha='center', va='center',
                fontsize=fontsize, color=text_color,
                fontweight=weight, zorder=4)
        if sub_text:
            ax.text(cx, cy - 0.25, sub_text, ha='center', va='center',
                    fontsize=fontsize - 2, color=text_color,
                    fontstyle='italic', zorder=4)

    def diamond(ax, cx, cy, w, h, color, text, fontsize=10):
        dx, dy = w/2, h/2
        xs = [cx,      cx + dx, cx,      cx - dx, cx]
        ys = [cy + dy, cy,      cy - dy, cy,      cy + dy]
        ax.fill(xs, ys, color=color, zorder=3)
        ax.plot(xs, ys, color='white', linewidth=2, zorder=4)
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold', zorder=5)

    def arrow(ax, x1, y1, x2, y2, color='#455A64', label=None,
              label_side='right'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=2.0), zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ox = 0.25 if label_side == 'right' else -0.25
            ax.text(mx + ox, my, label, ha='center', va='center',
                    fontsize=8.5, color=color,
                    bbox=dict(facecolor='#F0F4F8', edgecolor='none',
                              pad=1))

    def curved_arrow(ax, x1, y1, x2, y2, rad, color='#455A64',
                     label=None):
        style = f"arc3,rad={rad}"
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=2.0,
                                   connectionstyle=style), zorder=5)
        if label:
            mx = (x1+x2)/2 + rad*1.5
            my = (y1+y2)/2
            ax.text(mx, my, label, ha='center', va='center',
                    fontsize=8.5, color=color,
                    bbox=dict(facecolor='#F0F4F8', edgecolor='none',
                              pad=1))

    # ── title ─────────────────────────────────────────────────────────
    ax.text(7, 17.4, 'ReAct 算法流程图', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1A237E',
            bbox=dict(facecolor='#E8EAF6', edgecolor='#3F51B5',
                      boxstyle='round,pad=0.5', linewidth=2))

    # ── main flow nodes (x=7, top-to-bottom) ─────────────────────────
    # 1. User Input
    rounded_box(ax, 7, 16.0, 3.6, 0.9, '#1565C0',
                '用户输入 (User Input)', fontsize=12, bold=True)

    # 2. LLM Core
    rounded_box(ax, 7, 14.5, 3.6, 0.9, '#283593',
                'LLM 推理引擎', fontsize=12, bold=True,
                sub_text='(Language Model Core)')

    # 3. Thought
    rounded_box(ax, 7, 13.0, 3.6, 0.9, '#00838F',
                '思考 (Thought)', fontsize=12, bold=True,
                sub_text='分析问题，规划下一步行动')

    # 4. Decision diamond
    diamond(ax, 7, 11.5, 3.2, 1.1, '#E65100',
            '是否需要调用工具？', fontsize=9)

    # 5. Action
    rounded_box(ax, 7, 9.9, 3.6, 0.9, '#2E7D32',
                '行动 (Action)', fontsize=12, bold=True,
                sub_text='生成工具调用指令')

    # 6. Observation
    rounded_box(ax, 7, 8.3, 3.6, 0.9, '#6A1B9A',
                '观察环境反馈 (Observation)', fontsize=11, bold=True,
                sub_text='接收工具返回结果')

    # 7. Final Output
    rounded_box(ax, 7, 2.4, 3.6, 0.9, '#1565C0',
                '最终输出结果 (Final Answer)', fontsize=11, bold=True)

    # ── vertical arrows (main flow) ───────────────────────────────────
    arrow(ax, 7, 15.55, 7, 14.95)              # Input → LLM
    arrow(ax, 7, 14.05, 7, 13.45)              # LLM → Thought
    arrow(ax, 7, 12.55, 7, 12.05)              # Thought → Diamond

    # Diamond → Action  (Yes)
    arrow(ax, 7, 10.95, 7, 10.35,
          color='#2E7D32', label='是 (Yes)')

    # Action → Observation
    arrow(ax, 7, 9.45, 7, 8.75)

    # Observation → LLM  (loop back)
    ax.annotate('', xy=(4.8, 14.5), xytext=(4.8, 8.3),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A',
                                lw=2.0), zorder=5)
    ax.plot([5.2, 4.8], [8.3, 8.3],  color='#6A1B9A', lw=2, zorder=5)
    ax.plot([4.8, 4.8], [8.3, 14.5], color='#6A1B9A', lw=2, zorder=5)
    ax.plot([4.8, 5.2], [14.5, 14.5], color='#6A1B9A', lw=2, zorder=5)
    ax.text(3.9, 11.4, '继续循环\n(Loop)', ha='center', va='center',
            fontsize=8.5, color='#6A1B9A', style='italic',
            bbox=dict(facecolor='#F0F4F8', edgecolor='none', pad=1))

    # Diamond → Final  (No: answer known)
    arrow(ax, 7, 10.95, 7, 2.85,
          color='#C62828', label='否 (No) →\n直接回答')

    # ── external tools panel ──────────────────────────────────────────
    # Panel background
    tools_bg = FancyBboxPatch((9.8, 6.8), 3.8, 4.4,
                              boxstyle='round,pad=0.3',
                              facecolor='#FFF8E1', edgecolor='#F9A825',
                              linewidth=2, zorder=2)
    ax.add_patch(tools_bg)
    ax.text(11.7, 11.35, '外部工具 (External Tools)',
            ha='center', va='center', fontsize=10,
            fontweight='bold', color='#E65100')

    # Search Engine
    rounded_box(ax, 11.7, 10.5, 3.0, 0.75, '#F57F17',
                '[S]  Search Engine', fontsize=10)

    # Database
    rounded_box(ax, 11.7, 9.45, 3.0, 0.75, '#EF6C00',
                '[DB]  Database', fontsize=10)

    # Calculator / API
    rounded_box(ax, 11.7, 8.4, 3.0, 0.75, '#E65100',
                '[API]  Calculator / API', fontsize=10)

    # ── tool interaction arrows ───────────────────────────────────────
    # Action → tools  (调用)
    ax.annotate('', xy=(10.15, 9.9), xytext=(8.8, 9.9),
                arrowprops=dict(arrowstyle='->', color='#2E7D32',
                                lw=1.8), zorder=5)
    ax.text(9.48, 10.15, '调用 (Call)', ha='center', va='center',
            fontsize=8, color='#2E7D32')

    # tools → Observation  (结果)
    ax.annotate('', xy=(8.8, 8.3), xytext=(10.15, 8.6),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A',
                                lw=1.8), zorder=5)
    ax.text(9.48, 8.25, '结果 (Result)', ha='center', va='center',
            fontsize=8, color='#6A1B9A')

    # ── loop annotation box ───────────────────────────────────────────
    loop_bg = FancyBboxPatch((0.4, 7.8), 4.0, 6.5,
                             boxstyle='round,pad=0.3',
                             facecolor='none', edgecolor='#90CAF9',
                             linewidth=2, linestyle='dashed', zorder=2)
    ax.add_patch(loop_bg)
    ax.text(2.4, 14.55, '>>  推理循环 (ReAct Loop)',
            ha='center', va='center', fontsize=9,
            color='#1565C0', fontweight='bold',
            bbox=dict(facecolor='#E3F2FD', edgecolor='#90CAF9',
                      boxstyle='round,pad=0.3', linewidth=1))

    # ── legend ────────────────────────────────────────────────────────
    legend_y = 5.6
    ax.text(7, legend_y + 0.7, '图例 (Legend)',
            ha='center', va='center', fontsize=10,
            fontweight='bold', color='#37474F')
    legend_items = [
        ('#00838F', '思考节点 (Thought)'),
        ('#2E7D32', '行动节点 (Action)'),
        ('#6A1B9A', '观察节点 (Observation)'),
        ('#F57F17', '外部工具 (External Tools)'),
        ('#E65100', '决策节点 (Decision)'),
    ]
    for i, (color, label) in enumerate(legend_items):
        col = i % 3
        row = i // 3
        lx = 2.2 + col * 3.8
        ly = legend_y - 0.05 - row * 0.7
        patch = FancyBboxPatch((lx - 0.35, ly - 0.22), 0.7, 0.44,
                               boxstyle='round,pad=0.1',
                               facecolor=color, edgecolor='white',
                               linewidth=1, zorder=3)
        ax.add_patch(patch)
        ax.text(lx + 0.55, ly, label, ha='left', va='center',
                fontsize=8, color='#37474F')

    # ── step labels on left side ──────────────────────────────────────
    steps = [
        (16.0, '① 接收用户输入'),
        (14.5, '② LLM 理解问题'),
        (13.0, '③ 思考推理'),
        (11.5, '④ 判断是否需工具'),
        (9.9,  '⑤ 执行行动'),
        (8.3,  '⑥ 获取观察结果'),
        (2.4,  '⑦ 输出最终答案'),
    ]
    for y, label in steps:
        ax.text(0.35, y, label, ha='left', va='center',
                fontsize=7.5, color='#546E7A',
                bbox=dict(facecolor='white', edgecolor='#B0BEC5',
                          boxstyle='round,pad=0.2', linewidth=0.8))

    plt.tight_layout(pad=0.5)
    output_path = 'imgs/react_flow.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"ReAct 流程图已保存至: {output_path}")
    plt.close()


if __name__ == '__main__':
    draw_react_flowchart()
