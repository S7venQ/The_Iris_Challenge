# mission2_reproduce.py
"""
任务二：可视化3D Boundary
两分类/三个特征: p0, p1 = f(x1, x2, x3)
实现三维决策边界可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    """加载数据并准备特征"""
    iris = load_iris()
    
    # 为了两分类，只取前两个类别
    X = iris.data[:100, :3]
    y = iris.target[:100]
    
    # 将类别1改为0，类别2改为1
    y = np.where(y == 0, 0, 1)
    
    feature_names = iris.feature_names[:3]
    target_names = ['Setosa', 'Versicolor']
    
    print("数据集信息:")
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

def calculate_optimal_plane(X, y):
    """计算最优分割平面"""
    # 计算每个类别的中心点
    center_0 = X[y == 0].mean(axis=0)
    center_1 = X[y == 1].mean(axis=0)
    
    # 计算中心点的方向向量
    w_dir = center_1 - center_0
    w_dir = w_dir / np.linalg.norm(w_dir)
    
    # 计算两个类别中心点的中点
    midpoint = (center_0 + center_1) / 2
    
    # 平面方程: w·(x - midpoint) = 0
    w = w_dir
    b = -np.dot(w, midpoint)
    
    print(f"\n优化分割平面方程:")
    print(f"  {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {w[2]:.3f}*x3 + {b:.3f} = 0")
    
    return w, b

def create_3d_visualization(data, w, b):
    """创建3D决策边界可视化"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X = data['X_full']
    y = data['y_full']
    feature_names = data['feature_names']
    
    # 扩展坐标范围
    x_min, x_max = X[:, 0].min() - 1.2, X[:, 0].max() + 1.2
    y_min, y_max = X[:, 1].min() - 1.2, X[:, 1].max() + 1.2
    z_min, z_max = X[:, 2].min() - 1.2, X[:, 2].max() + 1.2
    
    # 生成网格点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    
    # 计算决策边界平面
    if abs(w[2]) > 1e-10:
        # 平面方程: w1*x + w2*y + w3*z + b = 0
        zz = -(w[0] * xx + w[1] * yy + b) / w[2]
        zz = np.clip(zz, z_min, z_max)
    else:
        zz = np.full_like(xx, X[:, 2].mean())
    
    # 绘制决策边界平面
    surf = ax.plot_surface(xx, yy, zz, alpha=0.35, color='royalblue', 
                          rstride=1, cstride=1, linewidth=0.5,
                          antialiased=True)
    
    # 绘制数据点
    colors = ['#FF5252', '#4CAF50']
    markers = ['o', 's']
    
    np.random.seed(42)
    offsets = np.random.uniform(-0.08, 0.08, size=X.shape)
    
    for i in range(2):
        mask = (y == i)
        X_class = X[mask] + offsets[mask]
        
        ax.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2],
                  c=colors[i], marker=markers[i], s=85,
                  edgecolor='black', linewidth=1.5, alpha=0.95,
                  depthshade=True, label=data['target_names'][i])
    
    # 设置坐标轴标签
    ax.set_xlabel(feature_names[0], fontsize=14, labelpad=12)
    ax.set_ylabel(feature_names[1], fontsize=14, labelpad=12)
    ax.set_zlabel(feature_names[2], fontsize=14, labelpad=12)
    
    # 设置坐标轴范围
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 设置标题
    ax.set_title('3D Decision Boundary with Optimized Plane\nTwo-class Separation in Feature Space', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置视角
    ax.view_init(elev=22, azim=40)
    
    # 添加平面方程文本
    plane_eq = f'Separation Plane: {w[0]:.2f}x + {w[1]:.2f}y + {w[2]:.2f}z + {b:.2f} = 0'
    ax.text2D(0.05, 0.95, plane_eq, transform=ax.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def evaluate_separation_quality(X, y, w, b):
    """评估平面分割质量"""
    # 计算点到平面的距离
    distances = w[0]*X[:, 0] + w[1]*X[:, 1] + w[2]*X[:, 2] + b
    
    # 根据距离预测类别
    predictions = (distances > 0).astype(int)
    
    # 计算准确率
    accuracy = np.mean(predictions == y)
    
    print(f"\n平面分割质量评估:")
    print(f"  分割准确率: {accuracy:.4f}")
    print(f"  正确分类: {np.sum(predictions == y)}/{len(y)} 个点")
    print(f"  误分点数: {np.sum(predictions != y)}/{len(y)} 个点")
    
    return accuracy

def main():
    """主函数"""
    print("开始生成3D决策边界可视化...")
    print("="*60)
    
    # 1. 加载和准备数据
    print("步骤1: 加载和准备数据...")
    data = load_and_prepare_data()
    
    # 2. 计算最优分割平面
    print("\n步骤2: 计算最优分割平面...")
    w, b = calculate_optimal_plane(data['X_full'], data['y_full'])
    
    # 3. 评估分割质量
    accuracy = evaluate_separation_quality(data['X_full'], data['y_full'], w, b)
    
    # 4. 创建3D可视化
    print("\n步骤3: 创建3D决策边界可视化...")
    fig = create_3d_visualization(data, w, b)
    
    # 5. 保存图片
    fig.savefig('3d_decision_boundary_optimized.png', dpi=200, bbox_inches='tight')
    print("  已保存: 3d_decision_boundary_optimized.png")
    
    # 6. 显示图表
    print("\n显示可视化图表...")
    plt.show()
    
    print("\n" + "="*60)
    print("3D可视化完成！")
    print(f"生成的文件: 3d_decision_boundary_optimized.png")
    print(f"分割准确率: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()