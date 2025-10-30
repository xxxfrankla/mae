#!/usr/bin/env python3
"""
MAE 实验管理工具
查看、对比和管理不同实验的结果
"""

import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class ExperimentManager:
    def __init__(self, results_dir='./visualization_results'):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            print(f"❌ 结果目录不存在: {results_dir}")
            return
        
        self.experiments = self._scan_experiments()
        print(f"📁 找到 {len(self.experiments)} 个实验")

    def _scan_experiments(self):
        """扫描所有实验文件夹"""
        experiments = []
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith('mae_results_'):
                # 解析时间戳
                timestamp_str = exp_dir.name.replace('mae_results_', '')
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    experiments.append({
                        'name': exp_dir.name,
                        'path': exp_dir,
                        'timestamp': timestamp,
                        'timestamp_str': timestamp_str
                    })
                except:
                    pass
        
        # 按时间排序
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        return experiments

    def list_experiments(self):
        """列出所有实验"""
        print("📋 实验列表:")
        print("-" * 80)
        print(f"{'序号':<4} {'时间':<20} {'文件夹名':<30} {'状态':<10}")
        print("-" * 80)
        
        for i, exp in enumerate(self.experiments):
            # 检查实验完整性
            status = self._check_experiment_status(exp['path'])
            time_str = exp['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i+1:<4} {time_str:<20} {exp['name']:<30} {status:<10}")
        
        print("-" * 80)

    def _check_experiment_status(self, exp_path):
        """检查实验状态"""
        required_files = [
            'experiment_summary.md',
            'training_curves/training_curves.png',
            'reconstructions/mae_reconstruction_comparison.png'
        ]
        
        missing = []
        for file_path in required_files:
            if not (exp_path / file_path).exists():
                missing.append(file_path)
        
        if not missing:
            return "✅ 完整"
        elif len(missing) < len(required_files):
            return "⚠️  部分"
        else:
            return "❌ 不完整"

    def show_experiment_details(self, exp_index):
        """显示实验详情"""
        if exp_index < 1 or exp_index > len(self.experiments):
            print(f"❌ 无效的实验序号: {exp_index}")
            return
        
        exp = self.experiments[exp_index - 1]
        print(f"\n📊 实验详情: {exp['name']}")
        print("=" * 60)
        
        # 读取训练统计
        stats_file = exp['path'] / 'analysis' / 'training_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print("🏋️  训练统计:")
            print(f"  训练轮数: {stats['total_epochs']}")
            print(f"  初始损失: {stats['initial_loss']:.4f}")
            print(f"  最终损失: {stats['final_loss']:.4f}")
            print(f"  损失下降: {stats['loss_reduction']:.1f}%")
            print(f"  最高学习率: {stats['max_lr']:.2e}")
        
        # 读取重建统计
        recon_file = exp['path'] / 'analysis' / 'reconstruction_stats.json'
        if recon_file.exists():
            with open(recon_file, 'r') as f:
                recon_stats = json.load(f)
            
            print("\n🎨 重建性能:")
            for key, data in recon_stats.items():
                mask_ratio = data['mask_ratio']
                print(f"  掩码 {mask_ratio*100:.0f}%: 损失 {data['loss']:.4f}, 误差 {data['mean_error']:.4f}")
        
        # 显示文件结构
        print(f"\n📁 文件结构:")
        self._print_tree(exp['path'], max_depth=2)

    def _print_tree(self, directory, prefix="", max_depth=3, current_depth=0):
        """打印目录树"""
        if current_depth >= max_depth:
            return
        
        items = sorted([item for item in directory.iterdir() if not item.name.startswith('.')])
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            
            if item.is_file():
                size = item.stat().st_size / 1024  # KB
                print(f"{prefix}{current_prefix}{item.name} ({size:.1f} KB)")
            else:
                print(f"{prefix}{current_prefix}{item.name}/")
                if current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    self._print_tree(item, prefix + extension, max_depth, current_depth + 1)

    def compare_experiments(self, exp_indices):
        """对比多个实验"""
        if len(exp_indices) < 2:
            print("❌ 至少需要选择2个实验进行对比")
            return
        
        print(f"\n📊 对比实验: {exp_indices}")
        print("=" * 60)
        
        # 收集数据
        comparison_data = []
        
        for idx in exp_indices:
            if idx < 1 or idx > len(self.experiments):
                print(f"❌ 无效的实验序号: {idx}")
                continue
            
            exp = self.experiments[idx - 1]
            stats_file = exp['path'] / 'analysis' / 'training_stats.json'
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                comparison_data.append({
                    'name': exp['name'],
                    'timestamp': exp['timestamp'].strftime('%m-%d %H:%M'),
                    'epochs': stats['total_epochs'],
                    'initial_loss': stats['initial_loss'],
                    'final_loss': stats['final_loss'],
                    'loss_reduction': stats['loss_reduction'],
                    'max_lr': stats['max_lr']
                })
        
        if not comparison_data:
            print("❌ 没有找到有效的实验数据")
            return
        
        # 创建对比表格
        df = pd.DataFrame(comparison_data)
        print("\n📈 训练对比:")
        print(df.to_string(index=False))
        
        # 可视化对比
        self._plot_comparison(comparison_data)

    def _plot_comparison(self, comparison_data):
        """绘制对比图表"""
        if len(comparison_data) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        names = [data['timestamp'] for data in comparison_data]
        
        # 损失对比
        initial_losses = [data['initial_loss'] for data in comparison_data]
        final_losses = [data['final_loss'] for data in comparison_data]
        
        x = range(len(names))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], initial_losses, width, label='初始损失', alpha=0.7)
        ax1.bar([i + width/2 for i in x], final_losses, width, label='最终损失', alpha=0.7)
        ax1.set_xlabel('实验')
        ax1.set_ylabel('损失')
        ax1.set_title('损失对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失下降百分比
        loss_reductions = [data['loss_reduction'] for data in comparison_data]
        bars = ax2.bar(names, loss_reductions, color='skyblue', alpha=0.7)
        ax2.set_xlabel('实验')
        ax2.set_ylabel('损失下降 (%)')
        ax2.set_title('损失下降对比')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, loss_reductions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 训练轮数
        epochs = [data['epochs'] for data in comparison_data]
        ax3.bar(names, epochs, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('实验')
        ax3.set_ylabel('训练轮数')
        ax3.set_title('训练轮数对比')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 学习率
        max_lrs = [data['max_lr'] for data in comparison_data]
        ax4.bar(names, max_lrs, color='orange', alpha=0.7)
        ax4.set_xlabel('实验')
        ax4.set_ylabel('最高学习率')
        ax4.set_title('学习率对比')
        ax4.tick_params(axis='x', rotation=45)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = self.results_dir / f'experiment_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 对比图保存: {comparison_path}")
        plt.show()

    def open_experiment(self, exp_index):
        """打开实验文件夹"""
        if exp_index < 1 or exp_index > len(self.experiments):
            print(f"❌ 无效的实验序号: {exp_index}")
            return
        
        exp = self.experiments[exp_index - 1]
        exp_path = exp['path'].absolute()
        
        print(f"📂 打开实验文件夹: {exp_path}")
        
        # 尝试在文件管理器中打开
        import subprocess
        import sys
        
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(exp_path)])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["explorer", str(exp_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(exp_path)])
        except:
            print(f"请手动打开: {exp_path}")

def main():
    """主函数"""
    print("🔬 MAE 实验管理工具")
    print("=" * 50)
    
    manager = ExperimentManager()
    
    if not manager.experiments:
        print("❌ 没有找到任何实验")
        print("💡 请先运行 organized_visualization.py 生成实验结果")
        return
    
    while True:
        print("\n📋 可用操作:")
        print("1. 列出所有实验")
        print("2. 查看实验详情")
        print("3. 对比实验")
        print("4. 打开实验文件夹")
        print("5. 退出")
        
        try:
            choice = input("\n请选择操作 (1-5): ").strip()
            
            if choice == '1':
                manager.list_experiments()
            
            elif choice == '2':
                manager.list_experiments()
                exp_num = int(input("请输入实验序号: "))
                manager.show_experiment_details(exp_num)
            
            elif choice == '3':
                manager.list_experiments()
                indices_str = input("请输入要对比的实验序号 (用空格分隔): ")
                indices = [int(x) for x in indices_str.split()]
                manager.compare_experiments(indices)
            
            elif choice == '4':
                manager.list_experiments()
                exp_num = int(input("请输入实验序号: "))
                manager.open_experiment(exp_num)
            
            elif choice == '5':
                print("👋 再见！")
                break
            
            else:
                print("❌ 无效选择，请重试")
        
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()


