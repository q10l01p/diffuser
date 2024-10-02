import time  # 导入时间模块，用于计时
import math  # 导入数学模块，用于数学运算
import pdb  # 导入 pdb 模块，用于调试

class Progress:  # 定义一个进度条类

    def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):  # 初始化函数
        self.total = total  # 保存总步数
        self.name = name  # 保存进度条名称
        self.ncol = ncol  # 保存每行参数列数
        self.max_length = max_length  # 保存每个参数的最大长度
        self.indent = indent  # 保存缩进长度
        self.line_width = line_width  # 保存行宽
        self._speed_update_freq = speed_update_freq  # 保存速度更新频率

        self._step = 0  # 初始化当前步数
        self._prev_line = '\033[F'  # 保存上一行的回车符
        self._clear_line = ' ' * self.line_width  # 保存用来清空行的空格字符串

        self._pbar_size = self.ncol * self.max_length  # 计算进度条大小
        self._complete_pbar = '#' * self._pbar_size  # 初始化完整的进度条
        self._incomplete_pbar = ' ' * self._pbar_size  # 初始化未完成的进度条

        self.lines = ['']  # 初始化参数行列表
        self.fraction = '{} / {}'.format(0, self.total)  # 初始化进度分数

        self.resume()  # 开始计时

    def update(self, description, n=1):  # 更新进度条函数
        self._step += n  # 更新当前步数
        if self._step % self._speed_update_freq == 0:  # 如果当前步数是速度更新频率的倍数
            self._time0 = time.time()  # 记录当前时间
            self._step0 = self._step  # 记录当前步数
        self.set_description(description)  # 更新描述信息

    def resume(self):  # 恢复计时函数
        self._skip_lines = 1  # 设置跳过行数
        print('\n', end='')  # 打印换行符
        self._time0 = time.time()  # 记录当前时间
        self._step0 = self._step  # 记录当前步数

    def pause(self):  # 暂停计时函数
        self._clear()  # 清空进度条
        self._skip_lines = 1  # 设置跳过行数

    def set_description(self, params=[]):  # 设置描述信息函数

        if type(params) == dict:  # 如果参数是字典类型
            params = sorted([  # 将字典排序
                    (key, val)  # 获取键值对
                    for key, val in params.items()  # 遍历字典
                ])

        ############
        # Position #  # 注释说明：设置进度条位置
        ############
        self._clear()  # 清空进度条

        ###########
        # Percent #  # 注释说明：计算进度百分比
        ###########
        percent, fraction = self._format_percent(self._step, self.total)  # 格式化进度百分比
        self.fraction = fraction  # 保存进度分数

        #########
        # Speed #  # 注释说明：计算进度速度
        #########
        speed = self._format_speed(self._step)  # 格式化进度速度

        ##########
        # Params #  # 注释说明：格式化参数信息
        ##########
        num_params = len(params)  # 获取参数个数
        nrow = math.ceil(num_params / self.ncol)  # 计算参数行数
        params_split = self._chunk(params, self.ncol)  # 将参数列表分割成多个子列表
        params_string, lines = self._format(params_split)  # 格式化参数字符串和参数行列表
        self.lines = lines  # 保存参数行列表

        description = '{} | {}{}'.format(percent, speed, params_string)  # 构建描述信息
        print(description)  # 打印描述信息
        self._skip_lines = nrow + 1  # 设置跳过行数

    def append_description(self, descr):  # 添加描述信息函数
        self.lines.append(descr)  # 将描述信息添加到参数行列表

    def _clear(self):  # 清空进度条函数
        position = self._prev_line * self._skip_lines  # 计算回车符的个数
        empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])  # 创建一个空的字符串，用于清空进度条
        print(position, end='')  # 打印回车符
        print(empty)  # 打印清空字符串
        print(position, end='')  # 打印回车符

    def _format_percent(self, n, total):  # 格式化进度百分比函数
        if total:  # 如果总步数不为 0
            percent = n / float(total)  # 计算进度百分比

            complete_entries = int(percent * self._pbar_size)  # 计算完成的进度条部分的长度
            incomplete_entries = self._pbar_size - complete_entries  # 计算未完成的进度条部分的长度

            pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]  # 构建进度条字符串
            fraction = '{} / {}'.format(n, total)  # 构建进度分数
            string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent*100))  # 构建进度条字符串
        else:  # 如果总步数为 0
            fraction = '{}'.format(n)  # 构建进度分数
            string = '{} iterations'.format(n)  # 构建进度条字符串
        return string, fraction  # 返回格式化后的进度条字符串和进度分数

    def _format_speed(self, n):  # 格式化进度速度函数
        num_steps = n - self._step0  # 计算步数差
        t = time.time() - self._time0  # 计算时间差
        speed = num_steps / t  # 计算进度速度
        string = '{:.1f} Hz'.format(speed)  # 构建速度字符串
        if num_steps > 0:  # 如果步数差大于 0
            self._speed = string  # 保存速度字符串
        return string  # 返回速度字符串

    def _chunk(self, l, n):  # 将列表分割成多个子列表函数
        return [l[i:i+n] for i in range(0, len(l), n)]  # 分割列表

    def _format(self, chunks):  # 格式化参数字符串和参数行列表函数
        lines = [self._format_chunk(chunk) for chunk in chunks]  # 格式化每个子列表
        lines.insert(0,'')  # 在列表开头插入一个空字符串
        padding = '\n' + ' '*self.indent  # 创建缩进字符串
        string = padding.join(lines)  # 将参数行列表连接成一个字符串
        return string, lines  # 返回格式化后的参数字符串和参数行列表

    def _format_chunk(self, chunk):  # 格式化每个子列表函数
        line = ' | '.join([self._format_param(param) for param in chunk])  # 将每个子列表中的参数格式化为字符串
        return line  # 返回格式化后的字符串

    def _format_param(self, param):  # 格式化参数函数
        k, v = param  # 获取参数的键和值
        return '{} : {}'.format(k, v)[:self.max_length]  # 格式化参数字符串

    def stamp(self):  # 打印进度条函数
        if self.lines != ['']:  # 如果参数行列表不为空
            params = ' | '.join(self.lines)  # 将参数行列表连接成一个字符串
            string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)  # 构建进度条字符串
            self._clear()  # 清空进度条
            print(string, end='\n')  # 打印进度条字符串
            self._skip_lines = 1  # 设置跳过行数
        else:  # 如果参数行列表为空
            self._clear()  # 清空进度条
            self._skip_lines = 0  # 设置跳过行数

    def close(self):  # 关闭进度条函数
        self.pause()  # 暂停计时

class Silent:  # 定义一个静默类

    def __init__(self, *args, **kwargs):  # 初始化函数
        pass  # 不做任何操作

    def __getattr__(self, attr):  # 获取属性函数
        return lambda *args: None  # 返回一个空函数，不做任何操作

if __name__ == '__main__':  # 定义主函数，当脚本直接运行时执行
    silent = Silent()  # 创建一个静默对象
    silent.update()  # 调用静默对象的 update 函数
    silent.stamp()  # 调用静默对象的 stamp 函数

    num_steps = 1000  # 定义总步数
    progress = Progress(num_steps)  # 创建一个进度条对象
    for i in range(num_steps):  # 循环执行 num_steps 步
        progress.update()  # 更新进度条
        params = [  # 创建参数列表
            ['A', '{:06d}'.format(i)],
            ['B', '{:06d}'.format(i)],
            ['C', '{:06d}'.format(i)],
            ['D', '{:06d}'.format(i)],
            ['E', '{:06d}'.format(i)],
            ['F', '{:06d}'.format(i)],
            ['G', '{:06d}'.format(i)],
            ['H', '{:06d}'.format(i)],
        ]
        progress.set_description(params)  # 设置描述信息
        time.sleep(0.01)  # 暂停 0.01 秒
    progress.close()  # 关闭进度条