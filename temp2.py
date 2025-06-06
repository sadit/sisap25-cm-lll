import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('sys_usage.log', parse_dates=['time'])

# 输出内存的峰值
peak_mem = df['mem_used_GB'].max()
peak_time = df.loc[df['mem_used_GB'].idxmax(), 'time']
print(f"内存峰值: {peak_mem:.2f} GB，出现时间: {peak_time}")

# 绘制 mem_used_GB 折线图
plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['mem_used_GB'], marker='o', label='mem_used_GB')
plt.xlabel('Time')
plt.ylabel('Memory Used (GB)')
plt.title('Memory Usage Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 绘制 dir_used_GB 折线图
plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['dir_used_GB'], marker='o', color='orange', label='dir_used_GB')
plt.xlabel('Time')
plt.ylabel('Dir Used (GB)')
plt.title('Directory Usage Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()