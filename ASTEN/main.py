import numpy as np

class BaseApEn(object):
    """
    近似熵基础类
    """

    def __init__(self, m, r):
        """
        初始化
        :param U:一个矩阵列表，for example:
            U = np.array([85, 80, 89] * 17)
        :param m: 子集的大小，int
        :param r: 阀值基数，0.1---0.2
        """
        self.m = m
        self.r = r

    @staticmethod
    def _maxdist(x_i, x_j):
        """计算矢量之间的距离"""
        return np.max([np.abs(np.array(x_i) - np.array(x_j))])

    @staticmethod
    def _getStdDev(U):
        """
        计算标准差的函数
        :param U:
        :return:
        """
        if not isinstance(U, np.ndarray):
            U = np.array(U)
        return np.std(U, ddof=1)


class ApEn(BaseApEn):# 继承自BaseApEn类
    """
    Pincus提出的算法，计算近似熵的类
    """

    def _normalization(self, U):
        """
        将数据标准化，
        获取平均值
        所有值减去平均值除以标准差
        """
        self.me = np.mean(U)
        self.biao = self._getStdDev(U)
        return np.array([(x - self.me) / self.biao for x in U])

    def _getThreshold(self, U):
        """
        获取阀值
        :param U:
        :return:
        """
        if not hasattr(self, "f"):
            self.f = self._getStdDev(U) * self.r
        return self.f

    def _getEn(self, m, U):
        """
        计算熵值（计算平均相似率）
        参考：https://blog.csdn.net/hdu_lazy_man/article/details/81332972
        :param U:
        :param m:
        :return:
        """
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self._getThreshold(U)]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        # 计算所有的比值列表的平均值（求和后再取平均）
        return np.sum(np.log(list(filter(lambda a: a, C)))) / (len(U) - m + 1.0)

    def _getEn_b(self, m, U):
        """
        标准化数据计算熵值
        :param m:
        :param U:
        :return:
        """
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self.r]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x: x, C)))) / (len(U) - m + 1.0)

    def getApEn(self, U):
        """
        计算近似熵
        :return:
        """
        print("近似熵为:",np.abs(self._getEn(self.m + 1, U) - self._getEn(self.m, U)))
        return np.abs(self._getEn(self.m + 1, U) - self._getEn(self.m, U))

    def getNormalApEn(self, U):
        """
        将原始数据标准化后的近似熵
        :param U:
        :return:
        """
        eeg = self._normalization(U)
        return np.abs(self._getEn_b(self.m + 1, eeg) - self._getEn_b(self.m, eeg))


if __name__ == "__main__":
    U = np.array([2, 4, 6, 8, 10] * 17)
    G = np.array([3, 4, 5, 6, 7] * 17)

    # 先生成对象
    ap = ApEn(2, 0.2)# 子片段长m=2，阈值r=0.2
    # 对象调用方法 计算近似熵
    ap.getApEn(U)  # 计算近似熵