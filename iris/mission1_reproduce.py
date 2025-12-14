# classifier_comparison_final.py
"""
可视化不同分类器的结果
三分类/两个特征: 花瓣长度和宽度
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    """加载数据并准备特征"""
    # 加载Iris数据集
    iris = load_iris()
    # 只使用后两个特征：花瓣长度和宽度
    X = iris.data[:, 2:]
    # 目标标签
    y = iris.target
    # 特征名称
    feature_names = ['Petal Length', 'Petal Width']
    
    # 划分训练集和测试集，使用分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 返回所有需要的数据
    return X_train, X_test, y_train, y_test, X, y, feature_names

def create_classifiers():
    """创建四个分类器"""
    # 1. 使用样条变换的逻辑回归模型
    # 样条变换用于生成非线性特征
    spline_model = Pipeline([
        ('spline', SplineTransformer(n_knots=5, degree=3)),
        ('logistic', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 2. 使用分箱变换的逻辑回归模型
    # 分箱变换将连续特征离散化
    binned_model = Pipeline([
        ('bins', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('logistic', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 3. 梯度提升分类器
    # 集成学习方法，组合多个弱学习器
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    
    # 4. 使用RBF核的支持向量机
    # RBF核可以处理非线性决策边界
    rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    # 将分类器组织成字典
    classifiers = {
        'Spline feature': spline_model,
        'Binned feature': binned_model, 
        'Gradient Boosting': gb_model,
        'RBF feature': rbf_model
    }
    
    return classifiers

def create_meshgrid(X, h=0.02):
    """创建用于绘制决策边界的网格"""
    # 计算特征的最小和最大值，并添加边界扩展
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # 生成网格点坐标
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # 返回网格和边界信息
    return xx, yy, x_min, x_max, y_min, y_max

def train_and_predict(classifiers, X_train, y_train, X, xx, yy):
    """训练分类器并预测网格点的概率"""
    # 存储预测结果的字典
    predictions = {}
    
    # 遍历每个分类器
    for name, clf in classifiers.items():
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 将网格点展平为一维数组
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 检查模型是否支持概率预测
        if hasattr(clf, "predict_proba"):
            # 直接获取概率
            probs = clf.predict_proba(grid_points)
        else:
            # 对于不支持概率的模型，使用决策函数
            if hasattr(clf, "decision_function"):
                # 获取决策函数值
                decision = clf.decision_function(grid_points)
                # 将决策函数转换为概率（使用softmax）
                probs = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            else:
                # 对于既不支持概率也不支持决策函数的模型，使用硬预测
                classes = clf.predict(grid_points)
                # 创建概率矩阵
                probs = np.zeros((len(classes), 3))
                for i, cls in enumerate(classes):
                    probs[i, cls] = 1.0
        
        # 将一维概率数组重塑为网格形状
        # 形状为 (网格高度, 网格宽度, 类别数)
        probs_grid = probs.reshape(xx.shape[0], xx.shape[1], 3)
        
        # 计算每个网格点的最大概率类别
        max_class = np.argmax(probs, axis=1).reshape(xx.shape)
        
        # 存储预测结果
        predictions[name] = {
            'probs': probs_grid,      # 每个类别的概率网格
            'max_class': max_class,   # 最大概率类别网格
            'classifier': clf         # 训练好的分类器
        }
    
    # 返回所有分类器的预测结果
    return predictions

def plot_classifier_comparison(predictions, X, y, xx, yy, feature_names):
    """绘制分类器比较图（16个子图）"""
    # 设置三个类别的颜色：金色、绿色、蓝色
    class_colors = ['#FFD700', '#32CD32', '#1E90FF']
    
    # 创建4行4列的图形，设置图形大小
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    
    # 设置主标题
    fig.suptitle('Classifier Comparison for Iris Dataset\nThree-class Classification with Two Features', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 获取分类器名称列表
    classifier_names = list(predictions.keys())
    
    # 遍历每个分类器，绘制四列子图
    for row, classifier_name in enumerate(classifier_names):
        # 获取当前分类器的预测数据
        pred_data = predictions[classifier_name]
        probs = pred_data['probs']          # 概率网格
        max_class = pred_data['max_class']  # 最大类别网格
        
        # 第1列：Class 0 的概率分布
        ax = axes[row, 0]
        # 提取Class 0的概率
        prob_class0 = probs[:, :, 0]
        
        # 创建从白色到金色的颜色映射
        cmap0 = mcolors.LinearSegmentedColormap.from_list(
            'class0_cmap', ['white', class_colors[0]], N=256)
        
        # 绘制概率等高线填充图
        contour0 = ax.contourf(xx, yy, prob_class0, levels=20, cmap=cmap0, alpha=0.8, vmin=0, vmax=1)
        # 绘制原始数据点
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=mcolors.ListedColormap(class_colors),
                  edgecolor='black', s=40, alpha=0.7)
        
        # 设置坐标轴标签
        if row == 3:  # 最后一行显示x轴标签
            ax.set_xlabel(feature_names[0], fontsize=10)
        # 每行都显示y轴标签
        ax.set_ylabel(feature_names[1], fontsize=10)
        
        # 设置子图标题
        ax.set_title('Class 0', fontsize=11, pad=10)
        
        # 设置坐标轴范围
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # 隐藏刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 在第一行添加分类器名称
        if row == 0:
            ax.text(0.5, 1.15, 'Spline feature', transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif row == 1:
            ax.text(0.5, 1.15, 'Binned feature', transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif row == 2:
            ax.text(0.5, 1.15, 'Gradient Boosting', transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif row == 3:
            ax.text(0.5, 1.15, 'RBF feature', transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar0 = plt.colorbar(contour0, ax=ax, orientation='vertical', shrink=0.8)
        cbar0.set_label('Probability', fontsize=8)
        
        # 第2列：Class 1 的概率分布
        ax = axes[row, 1]
        # 提取Class 1的概率
        prob_class1 = probs[:, :, 1]
        
        # 创建从白色到绿色的颜色映射
        cmap1 = mcolors.LinearSegmentedColormap.from_list(
            'class1_cmap', ['white', class_colors[1]], N=256)
        
        # 绘制Class 1的概率分布
        contour1 = ax.contourf(xx, yy, prob_class1, levels=20, cmap=cmap1, alpha=0.8, vmin=0, vmax=1)
        # 绘制数据点
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=mcolors.ListedColormap(class_colors),
                  edgecolor='black', s=40, alpha=0.7)
        
        # 最后一行显示x轴标签
        if row == 3:
            ax.set_xlabel(feature_names[0], fontsize=10)
        
        # 设置子图标题
        ax.set_title('Class 1', fontsize=11, pad=10)
        
        # 设置坐标轴范围和刻度
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加颜色条
        cbar1 = plt.colorbar(contour1, ax=ax, orientation='vertical', shrink=0.8)
        cbar1.set_label('Probability', fontsize=8)
        
        # 第3列：Class 2 的概率分布
        ax = axes[row, 2]
        # 提取Class 2的概率
        prob_class2 = probs[:, :, 2]
        
        # 创建从白色到蓝色的颜色映射
        cmap2 = mcolors.LinearSegmentedColormap.from_list(
            'class2_cmap', ['white', class_colors[2]], N=256)
        
        # 绘制Class 2的概率分布
        contour2 = ax.contourf(xx, yy, prob_class2, levels=20, cmap=cmap2, alpha=0.8, vmin=0, vmax=1)
        # 绘制数据点
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=mcolors.ListedColormap(class_colors),
                  edgecolor='black', s=40, alpha=0.7)
        
        # 最后一行显示x轴标签
        if row == 3:
            ax.set_xlabel(feature_names[0], fontsize=10)
        
        # 设置子图标题
        ax.set_title('Class 2', fontsize=11, pad=10)
        
        # 设置坐标轴范围和刻度
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加颜色条
        cbar2 = plt.colorbar(contour2, ax=ax, orientation='vertical', shrink=0.8)
        cbar2.set_label('Probability', fontsize=8)
        
        # 第4列：最大类别（最终分类结果）
        ax = axes[row, 3]
        
        # 绘制最大类别区域
        # 使用imshow显示类别区域，alpha控制透明度
        im = ax.imshow(max_class, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                      origin='lower', alpha=0.6,
                      cmap=mcolors.ListedColormap(class_colors))
        
        # 绘制数据点
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=mcolors.ListedColormap(class_colors),
                  edgecolor='black', s=40, alpha=0.7)
        
        # 最后一行显示x轴标签
        if row == 3:
            ax.set_xlabel(feature_names[0], fontsize=10)
        
        # 设置子图标题
        ax.set_title('Max class', fontsize=11, pad=10)
        
        # 设置坐标轴范围和刻度
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 只在第一个分类器的Max class图中添加图例
        if row == 0:
            from matplotlib.patches import Patch
            # 创建图例元素
            legend_elements = [
                Patch(facecolor=class_colors[0], edgecolor='black', label='Class 0 (Setosa)'),
                Patch(facecolor=class_colors[1], edgecolor='black', label='Class 1 (Versicolor)'),
                Patch(facecolor=class_colors[2], edgecolor='black', label='Class 2 (Virginica)')
            ]
            # 添加图例，位置在图形右上方
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9)
    
    # 调整布局，为图例留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # 返回图形对象
    return fig

def main():
    """主函数"""
    print("开始生成分类器比较可视化...")
    print("="*60)
    
    # 1. 加载和准备数据
    print("步骤1: 加载和准备数据...")
    X_train, X_test, y_train, y_test, X, y, feature_names = load_and_prepare_data()
    print(f"  特征: {feature_names}")
    print(f"  数据形状: {X.shape}")
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 2. 创建分类器
    print("\n步骤2: 创建分类器...")
    classifiers = create_classifiers()
    for name in classifiers.keys():
        print(f"  {name}")
    
    # 3. 创建网格
    print("\n步骤3: 创建预测网格...")
    xx, yy, x_min, x_max, y_min, y_max = create_meshgrid(X, h=0.02)
    print(f"  网格范围: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    
    # 4. 训练和预测
    print("\n步骤4: 训练分类器并生成预测...")
    predictions = train_and_predict(classifiers, X_train, y_train, X, xx, yy)
    print("  所有分类器训练完成！")
    
    # 5. 绘制可视化
    print("\n步骤5: 生成可视化图表...")
    fig = plot_classifier_comparison(predictions, X, y, xx, yy, feature_names)
    # 保存图形到文件
    fig.savefig('classifier_comparison.png', dpi=150, bbox_inches='tight')
    print("  已保存: classifier_comparison.png")
    
    # 6. 显示图表
    print("\n显示可视化图表...")
    plt.show()
    
    # 7. 打印模型性能
    print("\n" + "="*60)
    print("分类器性能评估:")
    print("="*60)
    # 遍历每个分类器，计算训练和测试准确率
    for name, pred_data in predictions.items():
        clf = pred_data['classifier']
        # 计算训练集准确率
        train_score = clf.score(X_train, y_train)
        # 计算测试集准确率
        test_score = clf.score(X_test, y_test)
        print(f"\n{name}:")
        print(f"  训练准确率: {train_score:.4f}")
        print(f"  测试准确率: {test_score:.4f}")
        # 计算过拟合程度（训练准确率 - 测试准确率）
        print(f"  过拟合程度: {train_score - test_score:.4f}")
    
    # 输出完成信息
    print("\n" + "="*60)
    print("可视化完成！")
    print("生成的文件: classifier_comparison.png")
    print("="*60)

# 程序入口
if __name__ == "__main__":
    main()