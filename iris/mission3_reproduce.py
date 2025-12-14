# mission3_true_probability_map_clean.py
"""
任务三：真正的3D Probability Map
显示三维空间中的概率分布
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    """主函数 - 生成真正的3D概率地图"""
    print("开始生成真正的3D Probability Map...")
    print("="*60)
    
    # 1. 加载数据
    print("步骤1: 加载Iris数据集...")
    iris = load_iris()
    X = iris.data[:, :3]
    y = iris.target
    
    feature_names = iris.feature_names[:3]
    target_names = iris.target_names
    
    print(f"  特征: {feature_names}")
    print(f"  类别: {target_names}")
    
    # 2. 训练概率模型
    print("\n步骤2: 训练逻辑回归模型...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  训练准确率: {train_acc:.3f}, 测试准确率: {test_acc:.3f}")
    
    # 3. 创建3D概率网格
    print("\n步骤3: 创建3D概率网格...")
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    z_min, z_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    
    grid_res = 15
    x_vals = np.linspace(x_min, x_max, grid_res)
    y_vals = np.linspace(y_min, y_max, grid_res)
    z_vals = np.linspace(z_min, z_max, grid_res)
    
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    print(f"  预测3D空间概率 ({len(grid_points)}个点)...")
    grid_scaled = scaler.transform(grid_points)
    probabilities = model.predict_proba(grid_scaled)
    
    prob_grid = probabilities[:, 0].reshape(grid_res, grid_res, grid_res)
    
    # 4. 创建图形
    print("\n步骤4: 生成可视化图表...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    target_names_display = ['Setosa', 'Versicolor', 'Virginica']
    
    # 5. 绘制概率等值面
    print("  绘制概率等值面...")
    
    level = 0.5
    high_prob_mask = prob_grid > level
    
    step = 2
    mask_sample = high_prob_mask[::step, ::step, ::step]
    
    if np.any(mask_sample):
        x_sample = xx[::step, ::step, ::step]
        y_sample = yy[::step, ::step, ::step]
        z_sample = zz[::step, ::step, ::step]
        prob_sample = prob_grid[::step, ::step, ::step]
        
        high_prob_points = mask_sample
        if np.any(high_prob_points):
            points_x = x_sample[high_prob_points]
            points_y = y_sample[high_prob_points]
            points_z = z_sample[high_prob_points]
            points_prob = prob_sample[high_prob_points]
            
            scatter = ax.scatter(points_x, points_y, points_z,
                               c=points_prob, cmap='Reds',
                               s=20,
                               alpha=0.4,
                               vmin=level, vmax=1.0,
                               edgecolor='none',
                               label=f'Probability > {level}')
    
    # 6. 绘制原始数据点
    print("  绘制数据点...")
    
    markers = ['o', '^', 's']
    
    for i, (color, marker, label) in enumerate(zip(colors, markers, target_names_display)):
        mask = y == i
        X_class = X[mask]
        
        ax.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2],
                  c=color, marker=marker, s=100,
                  edgecolor='black', linewidth=2.0, alpha=0.95,
                  depthshade=True, label=label)
    
    # 7. 在坐标平面上添加概率投影
    print("  添加坐标平面投影...")
    
    grid_res_2d = 30
    z_projection = z_min
    
    xx_2d, yy_2d = np.meshgrid(
        np.linspace(x_min, x_max, grid_res_2d),
        np.linspace(y_min, y_max, grid_res_2d)
    )
    
    grid_2d_points = np.c_[xx_2d.ravel(), yy_2d.ravel(), 
                           np.full(xx_2d.ravel().shape, z_projection)]
    
    grid_2d_scaled = scaler.transform(grid_2d_points)
    probs_2d = model.predict_proba(grid_2d_scaled)[:, 0]
    probs_2d_grid = probs_2d.reshape(grid_res_2d, grid_res_2d)
    
    ax.contourf(xx_2d, yy_2d, probs_2d_grid, 
                zdir='z', offset=z_projection, 
                cmap='Reds', alpha=0.2, levels=20)
    
    ax.text(x_max, y_min, z_projection, 'XY Plane\n(Probability Projection)',
           fontsize=9, color='darkred', alpha=0.8,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # 8. 设置图表属性
    ax.set_xlabel(feature_names[0], fontsize=12, labelpad=15)
    ax.set_ylabel(feature_names[1], fontsize=12, labelpad=15)
    ax.set_zlabel(feature_names[2], fontsize=12, labelpad=15)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    ax.view_init(elev=25, azim=135)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 9. 创建图例
    print("  创建图例...")
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='red', alpha=0.4, 
              label=f'High Probability Region (p > {level})'),
        
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0],
               markersize=10, label=target_names_display[0]),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors[1],
               markersize=10, label=target_names_display[1]),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[2],
               markersize=10, label=target_names_display[2]),
        
        Patch(facecolor='red', alpha=0.2, 
              label='XY Plane Probability Projection'),
    ]
    
    ax.legend(handles=legend_elements, 
              loc='upper right',
              fontsize=10,
              framealpha=0.9,
              fancybox=True,
              shadow=True)
    
    # 10. 添加标题
    ax.set_title('3D Probability Map for Iris Dataset\n' +
                'Setosa Class Probability Distribution',
                fontsize=16, fontweight='bold', pad=25)
    
    # 11. 添加颜色条
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Setosa Probability', fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # 12. 添加简要说明
    ax.text2D(0.02, 0.98, 
              '3D Probability Field Visualization',
              transform=ax.transAxes,
              fontsize=11,
              fontweight='bold',
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 13. 保存和显示
    plt.tight_layout()
    
    output_file = '3d_probability_map_clean.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n3D Probability Map已保存: {output_file}")
    
    print("显示图表...")
    plt.show()
    
    print("\n" + "="*60)
    print("3D Probability Map生成完成！")
    print("="*60)
    print("\n图表包含:")
    print("  高概率区域")
    print("  原始数据点")
    print("  XY平面概率投影")
    print("  图例")
    print("  概率颜色条")
    print("="*60)

if __name__ == "__main__":
    main()