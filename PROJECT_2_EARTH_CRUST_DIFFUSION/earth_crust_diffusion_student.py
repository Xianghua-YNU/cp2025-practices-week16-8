"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # TODO: 设置物理参数
    # TODO: 初始化数组
    # TODO: 实现显式差分格式
    # TODO: 返回计算结果
    # ============= 1. 物理与网格参数（适配9年+1年模拟） =============
    alpha = 1.0e-6       # 热扩散率 (m^2/s)，地壳典型值
    L = 100.0            # 模拟深度 (m)
    years = 10           # 总模拟10年（前9年演化，第10年取四季）
    days_per_year = 365  
    T_total = years * days_per_year * 24 * 3600  # 总时间 (秒)
    
    Nz = 100             # 深度网格数
    Nt = 1000 * years    # 时间步数（按每年1000步估算，可调整）
    
    dz = L / (Nz - 1)    # 深度步长
    dt = T_total / Nt    # 时间步长

    # 显式差分稳定性条件
    r = alpha * dt / (dz ** 2)
    if r > 0.5:
        raise ValueError(f"稳定性不满足！r={r:.4f} > 0.5，需增大Nt或减小Nz")

    # ============= 2. 初始化数组 =============
    depth_array = np.linspace(0, L, Nz)  
    # 温度场矩阵：行=深度，列=时间步；额外存第10年4个时间点
    temperature_matrix = np.zeros((Nz, Nt))  
    temperature_matrix[:, 0] = 10.0  # 初始温度10°C

    # ============= 3. 边界条件（地表年周期变化） =============
    def surface_temperature(t_seconds):
        # 年周期：振幅12°C，平均10°C（匹配题目 A=10, B=12）
        t_days = t_seconds / (24 * 3600)
        return 10 + 12 * np.sin(2 * np.pi * t_days / days_per_year)

    # ============= 4. 显式差分迭代（9年+1年完整模拟） =============
    for n in range(Nt - 1):
        # 地表边界 (z=0)
        t_current = n * dt
        temperature_matrix[0, n+1] = surface_temperature(t_current)
        
        # 内部节点 (1 <= z < Nz-1)
        for z in range(1, Nz - 1):
            temperature_matrix[z, n+1] = (
                temperature_matrix[z, n] + 
                r * (temperature_matrix[z+1, n] - 2 * temperature_matrix[z, n] + temperature_matrix[z-1, n])
            )
        
        # 底部边界 (z=Nz-1)：题目中“地表下20米全年11°C”，可简化为恒定或随深度渐变
        # 这里直接用恒定温度（若需更准，可根据地热梯度调整）
        temperature_matrix[-1, n+1] = 11.0  

    # ============= 5. 提取第10年四季时间点 =============
    # 第10年对应时间范围：9年(3285天) ~ 10年(3650天)
    year10_start_idx = np.argmin(np.abs(np.linspace(0, T_total, Nt) - 9*days_per_year*24*3600))
    season_days = [0, 91, 182, 273]  # 四季大致天数（第10年的 0天=春, 91天=夏, 182天=秋, 273天=冬）
    season_idxs = []
    for s_day in season_days:
        t_season = (9*days_per_year + s_day) * 24 * 3600
        season_idxs.append(np.argmin(np.abs(np.linspace(0, T_total, Nt) - t_season)))
    
    # 提取第10年四季的温度剖面
    year10_snapshots = temperature_matrix[:, season_idxs]

    # ============= 6. 分析振幅衰减与相位延迟（选第10年数据，已达稳态） =============
    # 取第10年地表温度周期，拟合各深度的振幅、相位
    # （简化版：直接用离散数据计算峰谷差、时间延迟，如需更准可FFT）
    year10_temp = temperature_matrix[:, year10_start_idx:]
    t_year10 = np.linspace(0, days_per_year, year10_temp.shape[1])  # 第10年的天数

    # 示例：计算5米、10米、20米深度的振幅
    depths_to_check = [5, 10, 20]
    amp_results = []
    phase_results = []
    for z_idx, z in enumerate(depth_array):
        if not np.isclose(z, np.array(depths_to_check), atol=0.1).any():
            continue
        temp_curve = year10_temp[z_idx]
        amp = (temp_curve.max() - temp_curve.min()) / 2  # 振幅
        peak_time = t_year10[np.argmax(temp_curve)]     # 峰值时间（相位）
        amp_results.append(amp)
        phase_results.append(peak_time)

    # （可选：打印或绘图展示振幅衰减、相位延迟）
    print("=== 振幅衰减（深度 vs 振幅）===")
    for z, amp in zip(depths_to_check, amp_results):
        print(f"深度 {z}m: 振幅 {amp:.2f}°C")
    print("=== 相位延迟（深度 vs 峰值天数）===")
    for z, pt in zip(depths_to_check, phase_results):
        print(f"深度 {z}m: 峰值在第 {pt:.1f} 天（地表在第 ~182.5 天，延迟体现）")

    return depth_array, temperature_matrix, year10_snapshots

if __name__ == "__main__":
    # 测试代码
    try:
        depth, T = solve_earth_crust_diffusion()
        print(f"计算完成，温度场形状: {T.shape}")
                # ============= 可视化：第10年四季温度轮廓 =============
        plt.figure(figsize=(10, 6))
        seasons = ["Spring", "Summer", "Autumn", "Winter"]
        for i, season in enumerate(seasons):
            plt.plot(year10_snaps[:, i], depth, label=season)
        
        plt.ylim(max(depth), 0)  # 深度从地表(0)到地下(100m)，反转y轴
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Depth (m)")
        plt.title("Year 10 Seasonal Temperature Profiles")
        plt.legend()
        plt.grid(True)
        plt.show()
    except NotImplementedError as e:
        print(e)
