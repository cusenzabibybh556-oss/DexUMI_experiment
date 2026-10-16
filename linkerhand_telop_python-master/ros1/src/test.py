import sys
import numpy as np
from collections import deque
import threading
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGridLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg

class MultiRealtimePlotterQt:
    """
    高性能多曲线实时数据显示类（基于PyQtGraph）
    3行1列布局，每行一个图表
    
    参数：
    - max_points: 每个图表显示的最大数据点数
    - titles: 每个图表的标题列表
    - ylabels: 每个图表的Y轴标签列表
    - xlabel: 所有图表的X轴标签
    - window_size: 窗口尺寸
    - update_rate: 更新频率（Hz）
    """
    
    def __init__(self, max_points=1000, titles=None, ylabels=None, 
                 xlabel="时间", window_size=(1200, 800),
                 update_rate=30):
        self.max_points = max_points
        self.update_rate = update_rate
        
        # 默认标题和标签
        if titles is None:
            titles = ["曲线1", "曲线2", "曲线3"]
        if ylabels is None:
            ylabels = ["数值1", "数值2", "数值3"]
            
        if len(titles) != 3 or len(ylabels) != 3:
            raise ValueError("titles和ylabels必须包含3个元素")
        
        self.titles = titles
        self.ylabels = ylabels
        self.xlabel = xlabel
        
        # 数据缓冲区 - 每个图表独立
        self.data_buffers = [deque(maxlen=max_points) for _ in range(3)]
        self.time_buffers = [deque(maxlen=max_points) for _ in range(3)]
        
        # 创建Qt应用
        self.app = QApplication(sys.argv if sys.argv else [])
        
        # 创建主窗口
        self.window = QMainWindow()
        self.window.setWindowTitle("多通道实时数据监控")
        self.window.resize(*window_size)
        
        # 创建中心部件
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        
        # 使用网格布局：3行1列
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10)  # 设置子图间距
        
        # 创建3个图表部件
        self.plot_widgets = []
        self.curves = []
        
        for i in range(3):
            # 创建图表部件
            plot_widget = pg.PlotWidget()
            plot_widget.setLabel('left', self.ylabels[i])
            plot_widget.setLabel('bottom', self.xlabel)
            plot_widget.setTitle(self.titles[i], size="12pt", color='k', bold=True)
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setMinimumHeight(200)  # 设置最小高度
            
            # 设置不同颜色的曲线
            colors = ['b', 'r', 'g']  # 蓝色、红色、绿色
            curve = plot_widget.plot(pen=pg.mkPen(colors[i], width=2))
            
            # 添加到底部布局
            main_layout.addWidget(plot_widget, i, 0)
            
            self.plot_widgets.append(plot_widget)
            self.curves.append(curve)
        
        # 设置布局比例
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 1)
        main_layout.setRowStretch(2, 1)
        
        # 运行标志
        self.running = False
        self.timer = None
        
        # 初始化数据
        self.clear_all()
        
        # 状态栏显示信息
        self.status_label = pg.LabelItem(justify='right')
        self.plot_widgets[2].getPlotItem().layout.addItem(self.status_label, 4, 1)
        self._update_status()
    
    def start(self):
        """启动实时显示"""
        if not self.running:
            self.running = True
            self.window.show()
            
            # 设置定时器更新图表
            self.timer = pg.QtCore.QTimer()
            self.timer.timeout.connect(self._update_all_plots)
            self.timer.start(int(1000 / self.update_rate))
            
            # 启动Qt事件循环
            if not QApplication.instance():
                self.app = QApplication([])
            QApplication.processEvents()
            
            print(f"多通道实时监控已启动，每通道最大点数: {self.max_points}")
    
    def add_data(self, channel, value, timestamp=None):
        """
        添加新的数据点到指定通道
        
        参数：
        - channel: 通道号（0, 1, 2）
        - value: 数据值
        - timestamp: 时间戳（如为None则使用相对时间）
        """
        if channel not in [0, 1, 2]:
            raise ValueError("channel必须是0, 1或2")
        
        if timestamp is None:
            timestamp = len(self.time_buffers[channel])
        
        self.data_buffers[channel].append(value)
        self.time_buffers[channel].append(timestamp)
    
    def add_batch_data(self, channel, values, timestamps=None):
        """
        批量添加数据到指定通道
        
        参数：
        - channel: 通道号（0, 1, 2）
        - values: 数据值列表
        - timestamps: 时间戳列表（如为None则使用相对时间）
        """
        if channel not in [0, 1, 2]:
            raise ValueError("channel必须是0, 1或2")
        
        if timestamps is None:
            start_idx = len(self.time_buffers[channel])
            timestamps = range(start_idx, start_idx + len(values))
        
        self.data_buffers[channel].extend(values)
        self.time_buffers[channel].extend(timestamps)
    
    def add_multi_data(self, values, timestamps=None):
        """
        同时添加3个通道的数据
        
        参数：
        - values: 包含3个值的列表 [value1, value2, value3]
        - timestamp: 时间戳（如为None则使用相对时间）
        """
        if len(values) != 3:
            raise ValueError("values必须包含3个值")
        
        if timestamps is None:
            timestamps = len(self.time_buffers[0])  # 使用第一个通道的时间
        
        for i in range(3):
            self.data_buffers[i].append(values[i])
            self.time_buffers[i].append(timestamps)
    
    def _update_plot(self, channel):
        """更新指定通道的图表"""
        if len(self.data_buffers[channel]) > 0:
            # 转换为numpy数组以提高性能
            x_data = np.array(self.time_buffers[channel])
            y_data = np.array(self.data_buffers[channel])
            
            # 更新曲线数据
            self.curves[channel].setData(x_data, y_data)
            
            # 自动调整视图范围
            if len(x_data) > 1:
                self.plot_widgets[channel].setXRange(min(x_data), max(x_data), padding=0.02)
                
                y_min = np.min(y_data)
                y_max = np.max(y_data)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else abs(y_min) * 0.1
                self.plot_widgets[channel].setYRange(y_min - margin, y_max + margin, padding=0.1)
    
    def _update_all_plots(self):
        """更新所有图表的显示"""
        for i in range(3):
            self._update_plot(i)
        self._update_status()
    
    def _update_status(self):
        """更新状态显示"""
        status_text = []
        for i in range(3):
            if len(self.data_buffers[i]) > 0:
                latest = self.data_buffers[i][-1]
                count = len(self.data_buffers[i])
                status_text.append(f"{self.titles[i]}: {latest:.3f} ({count}点)")
            else:
                status_text.append(f"{self.titles[i]}: 无数据")
        
        self.status_label.setText("<br>".join(status_text))
    
    def set_y_limits(self, channel, ymin, ymax):
        """设置指定通道的Y轴范围"""
        if channel in [0, 1, 2]:
            self.plot_widgets[channel].setYRange(ymin, ymax)
    
    def set_x_limits(self, channel, xmin, xmax):
        """设置指定通道的X轴范围"""
        if channel in [0, 1, 2]:
            self.plot_widgets[channel].setXRange(xmin, xmax)
    
    def clear_channel(self, channel):
        """清除指定通道的数据"""
        if channel in [0, 1, 2]:
            self.data_buffers[channel].clear()
            self.time_buffers[channel].clear()
            self.curves[channel].setData([], [])
    
    def clear_all(self):
        """清除所有通道的数据"""
        for i in range(3):
            self.clear_channel(i)
    
    def get_channel_data(self, channel):
        """获取指定通道的数据"""
        if channel in [0, 1, 2]:
            return list(self.time_buffers[channel]), list(self.data_buffers[channel])
        return [], []
    
    def get_all_data(self):
        """获取所有通道的数据"""
        all_data = []
        for i in range(3):
            all_data.append(self.get_channel_data(i))
        return all_data
    
    def stop(self):
        """停止实时显示"""
        if self.timer:
            self.timer.stop()
        self.running = False
        self.window.close()
        print("实时监控已停止")
    
    def save_screenshot(self, filename="multi_plot_screenshot.png"):
        """保存当前显示为图片"""
        import pyqtgraph.exporters as exporters
        
        # 创建图像导出器
        exporter = exporters.ImageExporter(self.plot_widgets[0].getPlotItem().scene())
        
        # 获取所有图表的场景并合并
        scenes = [pw.getPlotItem().scene() for pw in self.plot_widgets]
        
        # 导出图像
        exporter.export(filename)
        print(f"截图已保存: {filename}")
    
    def run(self):
        """运行Qt应用"""
        if self.running:
            sys.exit(self.app.exec_())


# 示例使用方式
if __name__ == "__main__":
    # 创建3通道实时图表
    plotter = MultiRealtimePlotterQt(
        max_points=1000,  # 每通道只保存1000点
        titles=["温度传感器", "压力传感器", "湿度传感器"],
        ylabels=["温度 (°C)", "压力 (kPa)", "湿度 (%)"],
        xlabel="采样时间",
        update_rate=30  # 30H1z更新
    )
    
    # 启动图表
    plotter.start()
    
    # 设置Y轴范围
    plotter.set_y_limits(0, 0, 100)    # 温度范围 0-100°C
    plotter.set_y_limits(1, 90, 110)   # 压力范围 90-110kPa
    plotter.set_y_limits(2, 0, 100)    # 湿度范围 0-100%
    
    # 模拟数据生成
    def generate_multi_sensor_data():
        """模拟3个传感器的数据生成"""
        sample_rate = 100  # 100Hz采样率
        interval = 1.0 / sample_rate
        time_counter = 0
        
        while plotter.running:
            # 生成3个通道的模拟数据
            t = time_counter
            
            # 通道1: 温度 (带缓慢变化的基波)
            temp = 25 + 10 * np.sin(2 * np.pi * 0.1 * t) + np.random.randn() * 0.5
            
            # 通道2: 压力 (带高频成分)
            pressure = 100 + 5 * np.sin(2 * np.pi * 1.0 * t) + 2 * np.sin(2 * np.pi * 5.0 * t)
            
            # 通道3: 湿度 (带趋势变化)
            humidity = 50 + 20 * np.sin(2 * np.pi * 0.05 * t) + np.random.randn() * 1.0
            
            # 添加数据到3个通道
            plotter.add_multi_data([temp, pressure, humidity], time_counter)
            
            time_counter += interval
            time.sleep(interval)
    
    # 启动数据生成线程
    data_thread = threading.Thread(target=generate_multi_sensor_data, daemon=True)
    data_thread.start()
    
    # 测试单个通道添加数据（示例）
    def test_single_channel():
        """测试单个通道数据添加"""
        for i in range(100):
            plotter.add_data(0, np.random.randn() * 10 + 50, i)
            time.sleep(0.01)
    
    # 启动测试线程
    # test_thread = threading.Thread(target=test_single_channel, daemon=True)
    # test_thread.start()
    
    # 运行Qt事件循环
    try:
        plotter.run()
    except KeyboardInterrupt:
        plotter.stop()
        print("\n程序已停止")   