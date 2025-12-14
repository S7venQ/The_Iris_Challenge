# mission_extension_3d_visualization.py
"""
任务拓展：完整的3D分类可视化
集成3D决策边界和概率映射的可视化系统
p0, p1, p2 = f(x1, x2, x3)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

def load_and_prepare_data():
    """加载并准备完整数据集"""
    iris = load_iris()
    
    # 使用前三个特征进行3D可视化
    X = iris.data[:, :3]
    y = iris.target
    
    feature_names = iris.feature_names[:3]
    target_names = iris.target_names
    
    print("完整数据集信息:")
    print(f"特征形状: {X.shape}")
    print(f"特征名: {feature_names}")
    print(f"类别名: {target_names}")
    print(f"类别分布: {np.bincount(y)}")
    
    return {
        'X_full': X,
        'y_full': y,
        'feature_names': feature_names,
        'target_names': target_names
    }

def train_classification_model(X_train, y_train):
    """训练分类模型"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 使用逻辑回归模型（支持概率输出）
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def calculate_3d_decision_boundaries(model, scaler, X_range):
    """计算3D决策边界"""
    # 创建3D网格
    x_min, x_max, y_min, y_max, z_min, z_max = X_range
    grid_res = 20
    
    # 生成网格点
    x_vals = np.linspace(x_min, x_max, grid_res)
    y_vals = np.linspace(y_min, y_max, grid_res)
    z_vals = np.linspace(z_min, z_max, grid_res)
    
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # 预测网格点类别
    grid_points_scaled = scaler.transform(grid_points)
    predictions = model.predict(grid_points_scaled)
    predictions_grid = predictions.reshape(xx.shape)
    
    # 预测网格点概率
    probabilities = model.predict_proba(grid_points_scaled)
    
    # 重塑概率为网格形状
    prob_grids = []
    for i in range(probabilities.shape[1]):
        prob_grid = probabilities[:, i].reshape(xx.shape)
        prob_grids.append(prob_grid)
    
    return {
        'grid_coords': (xx, yy, zz),
        'predictions': predictions_grid,
        'probabilities': prob_grids,
        'grid_points': grid_points
    }

def create_3d_boundary_visualization(data, grid_data, model, scaler):
    """创建3D决策边界可视化"""
    fig = plt.figure(figsize=(18, 8))
    
    X = data['X_full']
    y = data['y_full']
    feature_names = data['feature_names']
    target_names = data['target_names']
    
    xx, yy, zz = grid_data['grid_coords']
    predictions = grid_data['predictions']
    
    # 设置颜色
    colors = ['#FF5252', '#4CAF50', '#2196F3']  # 红色, 绿色, 蓝色
    markers = ['o', '^', 's']
    
    # 扩展坐标范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    z_min, z_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    
    # 子图1: 3D决策边界
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 绘制决策区域（使用抽样以减少计算量）
    step = 2
    xx_sample = xx[::step, ::step, ::step]
    yy_sample = yy[::step, ::step, ::step]
    zz_sample = zz[::step, ::step, ::step]
    pred_sample = predictions[::step, ::step, ::step]
    
    # 为每个类别绘制决策区域
    for class_idx in range(3):
        mask = pred_sample == class_idx
        if np.any(mask):
            ax1.scatter(xx_sample[mask], yy_sample[mask], zz_sample[mask],
                       c=colors[class_idx], alpha=0.1, s=1, label=f'{target_names[class_idx]} Region')
    
    # 绘制数据点
    for i in range(3):
        mask = y == i
        X_class = X[mask]
        
        ax1.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2],
                   c=colors[i], marker=markers[i], s=60,
                   edgecolor='black', linewidth=1.0, alpha=0.9,
                   depthshade=True, label=target_names[i])
    
    ax1.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
    ax1.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
    ax1.set_zlabel(feature_names[2], fontsize=12, labelpad=10)
    
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([z_min, z_max])
    
    ax1.set_title('3D Decision Boundary\nAll Three Classes', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=25, azim=45)
    
    # 子图2: Setosa类别的概率映射
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 获取Setosa的概率网格
    setosa_prob = grid_data['probabilities'][0]
    setosa_prob_sample = setosa_prob[::step, ::step, ::step]
    
    # 绘制高概率区域（p > 0.5）
    high_prob_mask = setosa_prob_sample > 0.5
    if np.any(high_prob_mask):
        prob_values = setosa_prob_sample[high_prob_mask]
        
        scatter = ax2.scatter(xx_sample[high_prob_mask], 
                            yy_sample[high_prob_mask], 
                            zz_sample[high_prob_mask],
                            c=prob_values, cmap='Reds',
                            s=30, alpha=0.4,
                            vmin=0.5, vmax=1.0,
                            edgecolor='none')
    
    # 绘制所有数据点
    for i in range(3):
        mask = y == i
        X_class = X[mask]
        
        ax2.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2],
                   c=colors[i], marker=markers[i], s=60,
                   edgecolor='black', linewidth=1.0, alpha=0.9,
                   depthshade=True, label=target_names[i])
    
    # 添加XY平面上的概率投影
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
    
    ax2.contourf(xx_2d, yy_2d, probs_2d_grid, 
                zdir='z', offset=z_projection, 
                cmap='Reds', alpha=0.2, levels=20)
    
    ax2.text(x_max, y_min, z_projection, 'XY Plane Projection',
            fontsize=9, color='darkred', alpha=0.8,
            verticalalignment='top')
    
    ax2.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
    ax2.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
    ax2.set_zlabel(feature_names[2], fontsize=12, labelpad=10)
    
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([z_min, z_max])
    
    ax2.set_title('3D Probability Map\nSetosa Class Probability', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=25, azim=45)
    
    # 添加颜色条
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=0.5, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap='Reds')
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Setosa Probability (p > 0.5)', fontsize=11)
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # 添加整体标题
    fig.suptitle('3D Classification Visualization: Decision Boundary + Probability Map\n' +
                '$p_0, p_1, p_2 = f(x_1, x_2, x_3)$',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig

def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\n模型性能评估:")
    print(f"  训练准确率: {train_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  过拟合程度: {train_acc - test_acc:.4f}")
    
    return train_acc, test_acc

def main():
    """主函数"""
    print("开始生成完整的3D分类可视化...")
    print("="*60)
    
    # 1. 加载和准备数据
    print("步骤1: 加载和准备数据...")
    data = load_and_prepare_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data['X_full'], data['y_full'], 
        test_size=0.3, random_state=42, stratify=data['y_full']
    )
    
    # 2. 训练分类模型
    print("\n步骤2: 训练分类模型...")
    model, scaler = train_classification_model(X_train, y_train)
    
    # 3. 评估模型性能
    train_acc, test_acc = evaluate_model_performance(model, 
        scaler.transform(X_train), y_train,
        scaler.transform(X_test), y_test)
    
    # 4. 计算3D决策边界
    print("\n步骤3: 计算3D决策边界和概率...")
    
    X = data['X_full']
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    
    X_range = (x_min, x_max, y_min, y_max, z_min, z_max)
    grid_data = calculate_3d_decision_boundaries(model, scaler, X_range)
    
    # 5. 创建可视化
    print("\n步骤4: 创建3D可视化图表...")
    fig = create_3d_boundary_visualization(data, grid_data, model, scaler)
    
    # 6. 保存图片
    output_file = '3d_complete_visualization.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  已保存: {output_file}")
    
    # 7. 显示图表
    print("\n显示可视化图表...")
    plt.show()
    
    print("\n" + "="*60)
    print("完整的3D分类可视化完成！")
    print("="*60)
    print(f"生成的文件: {output_file}")
    print(f"模型准确率: {test_acc:.4f}")
    print("="*60)
    
    return fig

if __name__ == "__main__":
    main()