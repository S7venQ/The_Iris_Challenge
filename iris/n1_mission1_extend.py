# classifier_comparison.py
"""
扩展功能：可视化不同分类器在三分类/两个特征上的结果
比较多种机器学习分类器在Iris数据集上的决策边界和性能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 导入多种分类器
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# 设置中文字体（如果需要显示中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """加载和准备数据"""
    # 加载Iris数据集
    iris = load_iris()
    
    # 选择后两个特征（花瓣长度和宽度）用于2D可视化
    X = iris.data[:, 2:]  # 只使用最后两个特征
    y = iris.target
    feature_names = iris.feature_names[2:]
    target_names = iris.target_names
    
    print("数据集信息:")
    print(f"特征形状: {X.shape}")
    print(f"特征名: {feature_names}")
    print(f"类别名: {target_names}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_full': X,
        'y_full': y,
        'feature_names': feature_names,
        'target_names': target_names,
        'scaler': scaler
    }

def create_classifiers():
    """创建要比较的分类器列表"""
    classifiers = [
        ('逻辑回归', LogisticRegression(max_iter=1000, random_state=42)),
        ('线性SVM', SVC(kernel='linear', C=1.0, probability=True, random_state=42)),
        ('RBF SVM', SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True, random_state=42)),
        ('K近邻 (k=5)', KNeighborsClassifier(n_neighbors=5)),
        ('决策树', DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('随机森林', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('高斯朴素贝叶斯', GaussianNB()),
        ('神经网络', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
    ]
    return classifiers

def train_and_evaluate_classifiers(classifiers, data):
    """训练并评估所有分类器"""
    results = {}
    
    print("\n" + "="*60)
    print("分类器性能评估")
    print("="*60)
    
    for name, clf in classifiers:
        print(f"\n训练 {name}...")
        
        # 训练模型
        clf.fit(data['X_train_scaled'], data['y_train'])
        
        # 预测
        y_pred = clf.predict(data['X_test_scaled'])
        y_pred_train = clf.predict(data['X_train_scaled'])
        
        # 计算准确率
        test_acc = accuracy_score(data['y_test'], y_pred)
        train_acc = accuracy_score(data['y_train'], y_pred_train)
        
        # 保存结果
        results[name] = {
            'classifier': clf,
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(data['y_test'], y_pred)
        }
        
        print(f"  训练准确率: {train_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  是否过拟合: {'是' if train_acc - test_acc > 0.1 else '否'}")
    
    return results

def plot_decision_boundaries(classifiers_results, data):
    """绘制所有分类器的决策边界"""
    # 创建网格用于绘制决策边界
    X = data['X_full']
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 设置颜色
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF']  # 三种类别的颜色
    cmap_light = mcolors.ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
    cmap_bold = mcolors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # 计算子图的行列数
    n_classifiers = len(classifiers_results)
    n_cols = 4
    n_rows = int(np.ceil(n_classifiers / n_cols))
    
    # 创建图形
    fig = plt.figure(figsize=(16, 4 * n_rows))
    
    for idx, (name, result) in enumerate(classifiers_results.items(), 1):
        clf = result['classifier']
        accuracy = result['test_accuracy']
        
        # 标准化网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_scaled = data['scaler'].transform(grid_points)
        
        # 预测网格点的类别
        if hasattr(clf, "predict_proba"):
            Z = clf.predict_proba(grid_points_scaled)
            Z = np.argmax(Z, axis=1)
        else:
            Z = clf.predict(grid_points_scaled)
        
        Z = Z.reshape(xx.shape)
        
        # 创建子图
        ax = plt.subplot(n_rows, n_cols, idx)
        
        # 绘制决策边界
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        
        # 绘制训练数据点
        scatter = ax.scatter(data['X_train'][:, 0], data['X_train'][:, 1], 
                            c=data['y_train'], cmap=cmap_bold, 
                            edgecolor='black', s=50, alpha=0.8)
        
        # 设置图形属性
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'{name}\n准确率: {accuracy:.3f}', fontsize=10)
        ax.set_xlabel(data['feature_names'][0])
        ax.set_ylabel(data['feature_names'][1])
        
        # 只在第一列添加y轴标签
        if idx % n_cols != 1:
            ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('classifier_comparison_decision_boundaries.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_performance_comparison(results, data):
    """绘制性能比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 准确率比较条形图
    ax1 = axes[0, 0]
    names = list(results.keys())
    test_accs = [results[name]['test_accuracy'] for name in names]
    train_accs = [results[name]['train_accuracy'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='训练准确率', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, test_accs, width, label='测试准确率', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('分类器')
    ax1.set_ylabel('准确率')
    ax1.set_title('训练集 vs 测试集准确率比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加准确率数值标签
    for i, (train, test) in enumerate(zip(train_accs, test_accs)):
        ax1.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 过拟合程度雷达图
    ax2 = axes[0, 1]
    overfit_scores = [train_accs[i] - test_accs[i] for i in range(len(names))]
    
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    overfit_scores += overfit_scores[:1]  # 闭合图形
    angles += angles[:1]  # 闭合角度
    
    ax2 = plt.subplot(2, 2, 2, projection='polar')
    ax2.plot(angles, overfit_scores, 'o-', linewidth=2)
    ax2.fill(angles, overfit_scores, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(names)
    ax2.set_title('过拟合程度雷达图\n(训练准确率 - 测试准确率)', va='bottom')
    ax2.grid(True)
    
    # 3. 最佳分类器的混淆矩阵热图
    ax3 = axes[1, 0]
    best_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_result = results[best_name]
    cm = best_result['confusion_matrix']
    
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title(f'最佳分类器: {best_name}\n混淆矩阵')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax3)
    
    # 添加文本标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax3.set_xticks(range(len(data['target_names'])))
    ax3.set_yticks(range(len(data['target_names'])))
    ax3.set_xticklabels(data['target_names'])
    ax3.set_yticklabels(data['target_names'])
    ax3.set_ylabel('真实标签')
    ax3.set_xlabel('预测标签')
    
    # 4. 特征重要性（仅适用于树模型和线性模型）
    ax4 = axes[1, 1]
    
    # 尝试获取特征重要性
    feature_importance_data = []
    importance_labels = []
    
    for name, result in results.items():
        clf = result['classifier']
        
        if hasattr(clf, 'coef_'):
            # 线性模型系数
            if len(clf.coef_.shape) == 2:  # 多分类情况
                importance = np.abs(clf.coef_).mean(axis=0)
            else:
                importance = np.abs(clf.coef_[0])
            feature_importance_data.append(importance)
            importance_labels.append(name)
        elif hasattr(clf, 'feature_importances_'):
            # 树模型特征重要性
            feature_importance_data.append(clf.feature_importances_)
            importance_labels.append(name)
    
    if feature_importance_data:
        importance_matrix = np.array(feature_importance_data)
        x = np.arange(len(data['feature_names']))
        width = 0.8 / len(importance_labels)
        
        for i, (label, importance) in enumerate(zip(importance_labels, feature_importance_data)):
            offset = width * (i - len(importance_labels) / 2)
            ax4.bar(x + offset, importance, width, label=label, alpha=0.7)
        
        ax4.set_xlabel('特征')
        ax4.set_ylabel('重要性')
        ax4.set_title('特征重要性比较')
        ax4.set_xticks(x)
        ax4.set_xticklabels(data['feature_names'])
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '特征重要性不可用\n(仅部分模型支持)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('特征重要性')
    
    plt.tight_layout()
    plt.savefig('classifier_comparison_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_detailed_report(results, data):
    """打印详细的分类报告"""
    print("\n" + "="*60)
    print("详细性能报告")
    print("="*60)
    
    # 找到最佳分类器
    best_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_result = results[best_name]
    
    print(f"\n🏆 最佳分类器: {best_name}")
    print(f"   测试准确率: {best_result['test_accuracy']:.4f}")
    print(f"   训练准确率: {best_result['train_accuracy']:.4f}")
    
    print("\n📊 混淆矩阵:")
    print(best_result['confusion_matrix'])
    
    print("\n📈 所有分类器排名:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    for rank, (name, result) in enumerate(sorted_results, 1):
        print(f"   {rank}. {name}: {result['test_accuracy']:.4f} "
              f"(训练: {result['train_accuracy']:.4f}, "
              f"过拟合: {result['train_accuracy'] - result['test_accuracy']:.4f})")

def main():
    """主函数"""
    print("🚀 开始比较不同分类器在三分类/两个特征上的表现")
    print("="*60)
    
    # 1. 加载和准备数据
    print("\n📊 步骤1: 加载和准备数据...")
    data = load_and_prepare_data()
    
    # 2. 创建分类器
    print("\n🔧 步骤2: 创建分类器...")
    classifiers = create_classifiers()
    
    # 3. 训练和评估
    print("\n⚙️ 步骤3: 训练和评估分类器...")
    results = train_and_evaluate_classifiers(classifiers, data)
    
    # 4. 可视化决策边界
    print("\n🎨 步骤4: 绘制决策边界...")
    plot_decision_boundaries(results, data)
    
    # 5. 可视化性能比较
    print("\n📈 步骤5: 绘制性能比较图...")
    plot_performance_comparison(results, data)
    
    # 6. 打印详细报告
    print_detailed_report(results, data)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("已生成以下文件:")
    print("  - classifier_comparison_decision_boundaries.png")
    print("  - classifier_comparison_performance.png")
    print("="*60)

if __name__ == "__main__":
    main()