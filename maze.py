import numpy as np
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk # 用于更好看的控件


CFG = {
    'ROWS': 5,
    'COLS': 5,
    'UNIT': 90,           # 格子稍微大一点
    'BG': '#1e1e2e',      # 全局背景 (深空灰蓝)
    'PANEL': '#2b2b40',   # 侧边栏背景
    'GRID': '#313244',    # 格子背景 (暗色)
    'ACCENT': '#89b4fa',  # 强调色 (亮蓝) - 对应网页 Agent 颜色
    'TARGET': '#f9e2af',  # 终点颜色 (淡黄)
    'TEXT': '#cdd6f4',    # 文字颜色 (云白)
    'FONT_MAIN': ('Segoe UI', 12),
    'FONT_BOLD': ('Segoe UI', 12, 'bold'),
    'FONT_EMOJI': ('Segoe UI Emoji', 30), # 专门显示 Emoji
}

ACTIONS = ['up', 'down', 'left', 'right']
Q_TABLE = pd.DataFrame(columns=ACTIONS, dtype=np.float64)

# --- 2. 界面核心类 ---
class CyberMaze(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('🤖 Q-Learning 迷宫测试')
        self.configure(bg=CFG['BG'])
        
        # 窗口居中计算
        w = CFG['COLS'] * CFG['UNIT'] + 260 # 260是侧边栏宽度
        h = CFG['ROWS'] * CFG['UNIT'] + 40  # 40是留白
        self.geometry(f'{w}x{h}')
        self.resizable(False, False)

        self.setup_ui()
        
    def setup_ui(self):
        # --- 左侧：游戏地图 ---
        # 使用 Frame 包裹 Canvas 实现边距
        game_frame = tk.Frame(self, bg=CFG['BG'], padx=20, pady=20)
        game_frame.pack(side='left')
        
        self.canvas = tk.Canvas(game_frame, bg=CFG['BG'],
                                height=CFG['ROWS'] * CFG['UNIT'],
                                width=CFG['COLS'] * CFG['UNIT'],
                                highlightthickness=0) # 去除丑陋的默认边框
        self.canvas.pack()

        # 画背景网格 (用矩形代替线条，模仿 CSS Grid gap 效果)
        self.cells = {} # 存储格子坐标
        for r in range(CFG['ROWS']):
            for c in range(CFG['COLS']):
                x0, y0 = c * CFG['UNIT'], r * CFG['UNIT']
                x1, y1 = x0 + CFG['UNIT'], y0 + CFG['UNIT']
                
                # 绘制格子底色 (留出 4px 间隙模拟 gap)
                gap = 4
                self.canvas.create_rectangle(
                    x0 + gap, y0 + gap, x1 - gap, y1 - gap,
                    fill=CFG['GRID'], outline=''
                )

        # --- 右侧：控制面板 ---
        self.panel = tk.Frame(self, bg=CFG['PANEL'], width=240)
        self.panel.pack(side='right', fill='y', ipadx=20)
        self.panel.pack_propagate(False) # 固定宽度

        # 标题
        tk.Label(self.panel, text="Reinforcement\nLearning", font=('Impact', 24),
                 bg=CFG['PANEL'], fg=CFG['ACCENT'], justify='left').pack(pady=(30, 20), anchor='w')

        # 数据显示
        self.var_ep = tk.StringVar(value="EPISODE: 0")
        self.var_step = tk.StringVar(value="STEPS: 0")
        
        self._create_stat_card("局数统计", self.var_ep)
        self._create_stat_card("当前步数", self.var_step)

        # 速度控制滑块
        tk.Label(self.panel, text="SIMULATION SPEED", font=('Arial', 8, 'bold'),
                 bg=CFG['PANEL'], fg='#6c7086').pack(anchor='w', pady=(30, 5))
        
        self.scale_speed = tk.Scale(self.panel, from_=0.01, to=0.5, resolution=0.01,
                                    orient='horizontal', length=180,
                                    bg=CFG['PANEL'], fg=CFG['TEXT'], 
                                    troughcolor=CFG['BG'], highlightthickness=0,
                                    label="", showvalue=0)
        self.scale_speed.set(0.1) # 默认速度
        self.scale_speed.pack(anchor='w')
        
        # 底部状态
        self.lbl_status = tk.Label(self.panel, text="READY", font=CFG['FONT_BOLD'],
                                   bg=CFG['PANEL'], fg='#a6adc8')
        self.lbl_status.pack(side='bottom', pady=30)

        # 初始化角色
        self.reset_agent_target()

    def _create_stat_card(self, title, var):
        # 简单的卡片样式
        frame = tk.Frame(self.panel, bg=CFG['BG'], pady=10, padx=10)
        frame.pack(fill='x', pady=5)
        tk.Label(frame, text=title, font=('Arial', 8), bg=CFG['BG'], fg='#6c7086').pack(anchor='w')
        tk.Label(frame, textvariable=var, font=('Arial', 14, 'bold'), bg=CFG['BG'], fg=CFG['TEXT']).pack(anchor='w')

    def reset_agent_target(self):
        self.canvas.delete("agent")
        self.canvas.delete("target")
        
        # 绘制终点 💎
        tx, ty = CFG['COLS']-1, CFG['ROWS']-1
        cx, cy = self._get_center(tx, ty)
        # 发光背景
        self.canvas.create_oval(cx-30, cy-30, cx+30, cy+30, fill=CFG['TARGET'], outline='', tags="target")
        # Emoji
        self.canvas.create_text(cx, cy, text="💎", font=CFG['FONT_EMOJI'], tags="target")

        # 绘制主角 🤖 (初始在 0,0)
        self.agent_pos = [0, 0]
        self.draw_agent(0, 0)

    def draw_agent(self, r, c):
        self.canvas.delete("agent")
        cx, cy = self._get_center(c, r)
        # 绘制圆角矩形背景 (用 oval 模拟圆形光晕)
        self.canvas.create_rectangle(
            c*CFG['UNIT']+8, r*CFG['UNIT']+8, 
            (c+1)*CFG['UNIT']-8, (r+1)*CFG['UNIT']-8,
            fill=CFG['ACCENT'], outline='', tags="agent"
        )
        self.canvas.create_text(cx, cy, text="🤖", font=CFG['FONT_EMOJI'], tags="agent")

    def _get_center(self, c, r):
        return c * CFG['UNIT'] + CFG['UNIT']/2, r * CFG['UNIT'] + CFG['UNIT']/2

    def update_view(self, ep, step, done=False):
        self.var_ep.set(f"EPISODE: {ep+1}")
        self.var_step.set(f"STEPS: {step}")
        if done:
            self.lbl_status.config(text="🎉 SUCCESS!", fg='#a6e3a1') # 绿色
        else:
            self.lbl_status.config(text="TRAINING...", fg='#f9e2af') # 黄色
        self.update()

# --- 3. 算法逻辑 (Q-Learning) ---
def check_state(state):
    state_str = str(state)
    if state_str not in Q_TABLE.index:
        Q_TABLE.loc[state_str] = [0.0] * 4

def choose_action(state):
    check_state(state)
    if np.random.uniform() < 0.1 or (Q_TABLE.loc[str(state)] == 0).all():
        return np.random.choice(ACTIONS)
    return Q_TABLE.loc[str(state)].idxmax()

def run_game():
    env = CyberMaze()
    # 延迟启动，给 UI 渲染时间
    env.after(1000, lambda: train_loop(env))
    env.mainloop()

def train_loop(env):
    for episode in range(50): # 训练50轮
        state = [0, 0]
        env.reset_agent_target()
        is_terminated = False
        step = 0
        
        while not is_terminated:
            # 1. 获取滑块速度
            sleep_t = env.scale_speed.get()
            time.sleep(sleep_t)
            
            # 2. 算法决策
            action = choose_action(state)
            
            # 3. 移动逻辑
            next_state = state.copy()
            if action == 'up':    next_state[0] = max(0, state[0]-1)
            elif action == 'down':  next_state[0] = min(CFG['ROWS']-1, state[0]+1)
            elif action == 'left':  next_state[1] = max(0, state[1]-1)
            elif action == 'right': next_state[1] = min(CFG['COLS']-1, state[1]+1)
            
            # 4. 奖励判断
            reward = 0
            if next_state == [CFG['ROWS']-1, CFG['COLS']-1]:
                reward = 1
                is_terminated = True
            
            # 5. 更新 Q 表
            check_state(next_state)
            q_predict = Q_TABLE.loc[str(state), action]
            if is_terminated:
                q_target = reward
            else:
                q_target = reward + 0.9 * Q_TABLE.loc[str(next_state)].max()
            
            Q_TABLE.loc[str(state), action] += 0.1 * (q_target - q_predict)
            
            # 6. UI 更新
            state = next_state
            env.draw_agent(state[0], state[1])
            env.update_view(episode, step, is_terminated)
            step += 1
        
        # 通关后稍微停顿
        time.sleep(0.5)

if __name__ == "__main__":
    run_game()