# main.py
import subprocess
import sys

def main():
    print("="*60)
    print("鸢尾花数据集可视化项目启动")
    print("="*60)
    
    while True:
        print("\n请选择要运行的程序：")
        print("1. 数据探索与可视化 (data_preview.py)")
        print("2. 分类器决策边界可视化 (classifier2d.py)")
        print("3. 多分类器比较可视化 (mission1_reproduce.py)")
        print("4. 3D决策边界可视化 (mission2_reproduce.py)")
        print("5. 3D概率映射可视化 (mission3_reproduce.py)")
        print("6. 扩展：多分类器系统 (n1_mission1_extend.py)")
        print("7. 扩展：3D完整可视化 (n2_mission4_try.py)")
        print("8. 运行所有程序")
        print("9. 退出")
        
        choice = input("\n请输入选项 (1-9): ")
        
        if choice == '1':
            print("\n正在运行数据探索程序...")
            subprocess.run([sys.executable, "data_preview.py"])
        
        elif choice == '2':
            print("\n正在运行分类器可视化程序...")
            subprocess.run([sys.executable, "classifier2d.py"])
        
        elif choice == '3':
            print("\n正在运行多分类器比较可视化程序...")
            subprocess.run([sys.executable, "mission1_reproduce.py"])
        
        elif choice == '4':
            print("\n正在运行3D决策边界可视化程序...")
            subprocess.run([sys.executable, "mission2_reproduce.py"])
        
        elif choice == '5':
            print("\n正在运行3D概率映射可视化程序...")
            subprocess.run([sys.executable, "mission3_reproduce.py"])
        
        elif choice == '6':
            print("\n正在运行多分类器系统扩展程序...")
            subprocess.run([sys.executable, "n1_mission1_extend.py"])
        
        elif choice == '7':
            print("\n正在运行3D完整可视化扩展程序...")
            subprocess.run([sys.executable, "n2_mission4_try.py"])
        
        elif choice == '8':
            print("\n正在运行所有程序...")
            print("\n1. 数据探索与可视化...")
            subprocess.run([sys.executable, "data_preview.py"])
            input("\n按Enter键继续运行分类器可视化...")
            
            print("\n2. 分类器决策边界可视化...")
            subprocess.run([sys.executable, "classifier2d.py"])
            input("\n按Enter键继续运行多分类器比较...")
            
            print("\n3. 多分类器比较可视化...")
            subprocess.run([sys.executable, "mission1_reproduce.py"])
            input("\n按Enter键继续运行3D决策边界...")
            
            print("\n4. 3D决策边界可视化...")
            subprocess.run([sys.executable, "mission2_reproduce.py"])
            input("\n按Enter键继续运行3D概率映射...")
            
            print("\n5. 3D概率映射可视化...")
            subprocess.run([sys.executable, "mission3_reproduce.py"])
            input("\n按Enter键继续运行扩展程序...")
            
            print("\n6. 多分类器系统扩展...")
            subprocess.run([sys.executable, "n1_mission1_extend.py"])
            input("\n按Enter键继续运行3D完整可视化...")
            
            print("\n7. 3D完整可视化扩展...")
            subprocess.run([sys.executable, "n2_mission4_try.py"])
        
        elif choice == '9':
            print("\n程序退出。")
            break
        
        else:
            print("\n无效选项，请重新选择。")

if __name__ == "__main__":
    main()