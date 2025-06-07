# 提取 RGB 值的函数
def extract_rgb_values_simple(color_string):
    # 去掉 'rgb(' 和 ')', 然后分割并转换为整数
    color_string = color_string[4:-1]  # 去掉 'rgb(' 和 ')'
    return tuple(map(int, color_string.split(",")))


# 计算透明度后的 RGB 值
def apply_opacity_to_rgb(color_string, opacity):
    # 提取原始 RGB 值
    r, g, b = extract_rgb_values_simple(color_string)

    # 计算透明度后的 RGB
    r_final = int((1 - opacity) * 255 + opacity * r)
    g_final = int((1 - opacity) * 255 + opacity * g)
    b_final = int((1 - opacity) * 255 + opacity * b)

    # 返回新的 rgb 字符串
    return f"rgb({r_final},{g_final},{b_final})"
