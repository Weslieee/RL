import numpy as np
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk

# --- 1. 全局配置 ---
CFG = {
    'ROWS': 5,
    'COLS': 5,
    'UNIT': 80,           # 格子大小
    'BG': '#1e1e2e',      # 全局背景
    'PANEL': '#2b2b40',   # 侧边栏背景
    'GRID': '#313244',    # 格子线条
    'ACCENT': '#89b4fa',  # 智能体颜色
    'TARGET': '#f9e2af',  # 终点颜色
    'TEXT': '#cdd6f4',    # 文本颜色
    'PLOT_BG': '#181825', # 绘图背景
    'PLOT_LINE': '#a6e3a1',# 曲线颜色 (绿)
    'FONT_MAIN': ('Segoe UI', 10),
    'FONT_BOLD': ('Segoe UI', 10, 'bold'),
    'FONT_EMOJI': ('Segoe UI Emoji', 30),
}

ACTIONS = ['up', 'down', 'left', 'right']
Q_TABLE = pd.DataFrame(columns=ACTIONS, dtype=np.float64)

# --- 2. 核心算法逻辑 ---
def check_state(state):
    s = str(state)
    if s not in Q_TABLE.index:
        Q_TABLE.loc[s] = [0.0] * 4

def choose_action(state):
    check_state(state)
    # Epsilon-Greedy: 10% 随机探索
    if np.random.uniform() < 0.1 or (Q_TABLE.loc[str(state)] == 0).all():
        return np.random.choice(ACTIONS)
    return Q_TABLE.loc[str(state)].idxmax()

# --- 3. 界面与交互类 ---
class CyberMaze(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('🤖 Q-Learning 训练监控')
        self.configure(bg=CFG['BG'])
        
        self.offset = 0 # [修改] 去掉偏移量

        # 窗口布局计算
        # 左侧地图 + 右侧面板 (300px) + 偏移量
        w = CFG['COLS'] * CFG['UNIT'] + 320 + self.offset
        h = max(CFG['ROWS'] * CFG['UNIT'] + 40 + self.offset, 600) 
        self.geometry(f'{w}x{h}')
        self.resizable(False, False)

        # 状态变量
        self.is_running = False
        self.episode = 0
        self.step_count = 0
        self.state = [0, 0]
        self.history = [] # 记录每局步数 [105, 50, 20, 8...]

        self.setup_ui()
        
    def setup_ui(self):
        # --- 左侧：游戏地图 ---
        game_frame = tk.Frame(self, bg=CFG['BG'], padx=20, pady=20)
        game_frame.pack(side='left', fill='y')
        
        # [修改] 画布大小增加了 offset，以便画坐标轴
        self.canvas = tk.Canvas(game_frame, bg=CFG['BG'],
                                height=CFG['ROWS'] * CFG['UNIT'] + self.offset,
                                width=CFG['COLS'] * CFG['UNIT'] + self.offset,
                                highlightthickness=0)
        self.canvas.pack()

        # 绘制网格背景
        for r in range(CFG['ROWS']):
            for c in range(CFG['COLS']):
                # [修改] 加上偏移量
                x0 = c * CFG['UNIT'] + self.offset
                y0 = r * CFG['UNIT'] + self.offset
                
                self.canvas.create_rectangle(
                    x0+4, y0+4, x0+CFG['UNIT']-4, y0+CFG['UNIT']-4,
                    fill=CFG['GRID'], outline=''
                )
                
        # --- 右侧：控制面板 ---
        self.panel = tk.Frame(self, bg=CFG['PANEL'], width=300)
        self.panel.pack(side='right', fill='y', ipadx=15)
        self.panel.pack_propagate(False)

        # 1. 标题
        tk.Label(self.panel, text="Reinforcement\nLearning", font=('Impact', 27),
                 bg=CFG['PANEL'], fg=CFG['ACCENT'], justify='left').pack(pady=(25, 10), anchor='w')

        # 2. 统计数据卡片
        self.var_ep = tk.StringVar(value="EPISODE: 0")
        self.var_step = tk.StringVar(value="CURRENT STEPS: 0")
        self._create_stat_card(self.var_ep)
        self._create_stat_card(self.var_step)

        # 3. 实时曲线图 (Canvas)
        tk.Label(self.panel, text="TRAINING CURVE (Steps/Episode)", font=('Arial', 8, 'bold'),
                 bg=CFG['PANEL'], fg='#6c7086').pack(anchor='w', pady=(20, 5))
        
        self.plot_h = 150
        self.plot_w = 260
        self.plot_canvas = tk.Canvas(self.panel, bg=CFG['PLOT_BG'], 
                                     height=self.plot_h, width=self.plot_w,
                                     highlightthickness=0)
        self.plot_canvas.pack(anchor='w')
        # 画基准坐标轴
        self._draw_baseline()

        # 4. 控制区
        # 速度滑块
        tk.Label(self.panel, text="SPEED", font=('Arial', 8, 'bold'),
                 bg=CFG['PANEL'], fg='#6c7086').pack(anchor='w', pady=(20, 5))
        self.scale_speed = tk.Scale(self.panel, from_=1, to=100, orient='horizontal',
                                    bg=CFG['PANEL'], fg=CFG['TEXT'], troughcolor=CFG['BG'],
                                    showvalue=0, highlightthickness=0, length=260)
        self.scale_speed.set(50) # 默认中间
        self.scale_speed.pack(anchor='w')

        # 开始按钮
        self.btn_start = tk.Button(self.panel, text="START TRAINING ▶", command=self.start_training,
                                   bg=CFG['ACCENT'], fg='#1e1e2e', font=('Arial', 10, 'bold'),
                                   relief='flat', padx=20, pady=10, cursor='hand2')
        self.btn_start.pack(side='bottom', pady=30, fill='x')

        # 初始化画面
        self.reset_env_view()

    def _create_stat_card(self, var):
        tk.Label(self.panel, textvariable=var, font=('Consolas', 12),
                 bg=CFG['BG'], fg=CFG['TEXT'], padx=10, pady=8, width=25, anchor='w').pack(pady=5)

    def _draw_baseline(self):
        # [修改] 绘制固定坐标轴线和 X 轴刻度
        # 预留左边和下边的边距
        margin_l = 30
        margin_b = 20
        x0, y0 = margin_l, self.plot_h - margin_b
        
        # 轴线
        self.plot_canvas.create_line(x0, y0, self.plot_w, y0, fill='#6c7086', width=1) # X轴
        self.plot_canvas.create_line(x0, 0, x0, y0, fill='#6c7086', width=1) # Y轴
        
        # X 轴刻度 (Episode)
        self.plot_canvas.create_text(x0, y0+10, text="0", fill='#6c7086', font=('Arial', 8))
        self.plot_canvas.create_text(x0 + (self.plot_w-x0)/2, y0+10, text="25", fill='#6c7086', font=('Arial', 8))
        self.plot_canvas.create_text(self.plot_w-10, y0+10, text="50", fill='#6c7086', font=('Arial', 8))

    def update_plot(self):
        """实时绘制折线图"""
        self.plot_canvas.delete("line")
        self.plot_canvas.delete("point")
        self.plot_canvas.delete("y_label") # 清除旧的Y轴数值
        self.plot_canvas.delete("opt_line") # 清除旧的最优线
        
        if not self.history: return
        
        # 定义绘图区域 (需要减去边距)
        margin_l = 30
        margin_b = 20
        draw_w = self.plot_w - margin_l
        draw_h = self.plot_h - margin_b
        
        # 数据归一化
        max_ep = 50 # 预设总局数
        max_steps = max(max(self.history), 20) # 动态Y轴最大值，防止初期太扁
        
        # [新增] 绘制动态 Y 轴刻度
        self.plot_canvas.create_text(margin_l-15, self.plot_h - margin_b, text="0", fill='#6c7086', font=('Arial', 8), tags="y_label")
        self.plot_canvas.create_text(margin_l-15, 10, text=str(max_steps), fill='#6c7086', font=('Arial', 8), tags="y_label")
        
        # [新增] 绘制最优步数虚线 (8步)
        y_opt = (self.plot_h - margin_b) - (8 / max_steps) * draw_h
        if y_opt > 0: # 只有在显示范围内才画
            self.plot_canvas.create_line(margin_l, y_opt, self.plot_w, y_opt, fill='#45475a', dash=(2, 2), tags="opt_line")
            self.plot_canvas.create_text(self.plot_w-20, y_opt-8, text="Best(8)", fill='#45475a', font=('Arial', 7), tags="opt_line")

        points = []
        for i, steps in enumerate(self.history):
            x = margin_l + (i / max_ep) * draw_w
            # 限制 y 不超出画布
            norm_step = min(steps, max_steps)
            y = (self.plot_h - margin_b) - (norm_step / max_steps) * draw_h
            points.append(x)
            points.append(y)
            
            # 画小圆点
            self.plot_canvas.create_oval(x-2, y-2, x+2, y+2, fill=CFG['ACCENT'], outline='', tags="point")

        if len(points) >= 4:
            self.plot_canvas.create_line(points, fill=CFG['PLOT_LINE'], width=2, tags="line", smooth=True)

    def reset_env_view(self):
        self.canvas.delete("agent")
        self.canvas.delete("target")
        # 画终点
        tx, ty = CFG['COLS']-1, CFG['ROWS']-1
        cx, cy = self._get_center(tx, ty)
        self.canvas.create_oval(cx-30, cy-30, cx+30, cy+30, fill=CFG['TARGET'], outline='')
        self.canvas.create_text(cx, cy, text="💎", font=CFG['FONT_EMOJI'])
        # 画起点 Agent
        self.draw_agent(0, 0)

    def draw_agent(self, r, c):
        self.canvas.delete("agent")
        cx, cy = self._get_center(c, r)
        # [修改] 加上偏移量计算矩形位置
        x0 = c * CFG['UNIT'] + self.offset
        y0 = r * CFG['UNIT'] + self.offset
        self.canvas.create_rectangle(
            x0+10, y0+10, x0+CFG['UNIT']-10, y0+CFG['UNIT']-10,
            fill=CFG['ACCENT'], outline='', tags="agent"
        )
        self.canvas.create_text(cx, cy, text="🤖", font=CFG['FONT_EMOJI'], tags="agent")

    def _get_center(self, c, r):
        # [修改] 计算中心点时加上偏移量
        return c * CFG['UNIT'] + CFG['UNIT']/2 + self.offset, r * CFG['UNIT'] + CFG['UNIT']/2 + self.offset

    def start_training(self):
        if self.is_running: return
        self.is_running = True
        self.btn_start.config(state='disabled', text="TRAINING...", bg='#45475a')
        # 重置数据
        global Q_TABLE
        Q_TABLE = pd.DataFrame(columns=ACTIONS, dtype=np.float64)
        self.history = []
        self.plot_canvas.delete("line", "point")
        self.plot_canvas.delete("y_label") # 重置时也要清空标签
        self.plot_canvas.delete("opt_line")
        self.episode = 0
        self.run_episode()

    def run_episode(self):
        """每一局的初始化"""
        if self.episode >= 50: # 跑50局结束
            self.is_running = False
            self.btn_start.config(state='normal', text="RESTART", bg=CFG['ACCENT'])
            print("训练结束！")
            return

        self.state = [0, 0]
        self.step_count = 0
        self.reset_env_view()
        self.var_ep.set(f"EPISODE: {self.episode + 1}")
        
        # 开启步进循环
        self.after(10, self.step_loop)

    def step_loop(self):
        """每一步的逻辑 (递归调用实现动画)"""
        # 1. 速度控制 (反向映射：滑块越大，sleep越短)
        speed_val = self.scale_speed.get() # 1~100
        delay = int(200 - speed_val * 1.8) # 200ms ~ 20ms
        
        # 2. 算法决策
        action = choose_action(self.state)
        
        # 3. 移动
        next_state = self.state.copy()
        if action == 'up':    next_state[0] = max(0, self.state[0]-1)
        elif action == 'down':  next_state[0] = min(CFG['ROWS']-1, self.state[0]+1)
        elif action == 'left':  next_state[1] = max(0, self.state[1]-1)
        elif action == 'right': next_state[1] = min(CFG['COLS']-1, self.state[1]+1)
        
        # 4. 奖励与更新
        reward = 0
        done = False
        if next_state == [CFG['ROWS']-1, CFG['COLS']-1]:
            reward = 1
            done = True
        
        check_state(next_state)
        q_predict = Q_TABLE.loc[str(self.state), action]
        if done:
            q_target = reward
        else:
            q_target = reward + 0.9 * Q_TABLE.loc[str(next_state)].max()
        
        Q_TABLE.loc[str(self.state), action] += 0.1 * (q_target - q_predict)
        
        # 5. UI 更新
        self.state = next_state
        self.step_count += 1
        self.draw_agent(self.state[0], self.state[1])
        self.var_step.set(f"STEPS: {self.step_count}")

        # 6. 判断结束
        if done:
            # 本局结束，记录数据，更新图表
            self.history.append(self.step_count)
            self.update_plot()
            self.episode += 1
            # 暂停一下再开新局
            self.after(500, self.run_episode) 
        else:
            # 继续走下一步
            self.after(delay, self.step_loop)

if __name__ == "__main__":
    app = CyberMaze()
    app.mainloop()
