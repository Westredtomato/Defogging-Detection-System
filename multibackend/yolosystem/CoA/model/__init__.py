try:
    from .Teacher import *
except ImportError as e:
    print("Warning: 无法导入 Teacher 相关模块，训练功能可能不可用：", e)

from .Student import *
from .Student_x import *

