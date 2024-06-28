import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000

def imgprocess(imgs):
    # 创建加权系数
    weights = np.array([0.2989, 0.5870, 0.1140])
    
    # 计算灰度图像
    gray_imgs = np.dot(imgs[..., :3], weights).astype(np.uint8)
    # gray_imgs = np.expand_dims(gray_imgs, axis=-1)
    
    return gray_imgs

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        """
        @param dataset_name: 数据集名称
        @param shuffle: 是否打乱
        @param seed: 随机种子数
        @param init_cls: 初始训练类数
        @param increment: 每次增加的类数
        """
        # 设置数据集名称
        self.dataset_name = dataset_name
        # 设置数据（包括下载数据集，设置转化等）
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."     # 保证初始训练类数比总类数小
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):   # 增量任务类数列表，表示每次增加几个类
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)     # 最后几个类补充进列表中，处理不整除的情况

    @property
    def nb_tasks(self):
        """根据增量任务数列表返回增量任务数"""
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        """
        数据集下载和预处理
        @param indices: 需要的数据的类别索引，比如range(0, 10)表示获取第0类到第9类的数据
        @param source: 数据集类型，如训练"train"，测试"test"
        @param mode: 预处理的模式，对不同的模式应用不同的变换
        @param appendent: 数据缓冲区，存储之前增量训练中data和target的样例
        @param ret_data: 不知道
        @param m_rate: 不知道
        """
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0), # 随机水平翻转
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                # 选择特定范围内的数据和目标
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                # 根据m_rate选择特定范围内的数据和目标
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        # 如果缓冲区appendent不为空，则将之前data和targets的样例添加到当前的data和targets中
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        # 将列表中的数据和目标连接成numpy数组
        data, targets = np.concatenate(data), np.concatenate(targets)

        # 如果ret_data为True，则返回数据、目标和DummyDataset实例
        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            # 否则只返回DummyDataset实例
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        # 获取指定数据集的实例
        idata = _get_idata(dataset_name)
        # 下载数据
        idata.download_data()

        # 设置训练数据和测试数据，以及它们对应的标签
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path  # 设置是否使用路径

        # 设置数据转换方法（以下三个转换的设置位于data.py）
        self._train_trsf = idata.train_trsf  # 训练数据的转换方法
        self._test_trsf = idata.test_trsf    # 测试数据的转换方法
        self._common_trsf = idata.common_trsf  # 训练和测试数据的公共转换方法

        # 生成类别顺序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            # 如果需要打乱顺序，则设置随机种子并打乱顺序
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            # 否则使用数据集中默认的类别顺序
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)  # 记录类别顺序

        # 映射训练和测试数据的标签到新的类别索引
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        # 确保图像和标签的数量相同
        assert len(images) == len(labels), "Data size error!"
        self.images = images  # 图像数据
        self.labels = labels  # 标签数据
        self.trsf = trsf  # 图像转换方法
        self.use_path = use_path  # 是否使用路径来加载图像

    # 返回数据集的大小
    def __len__(self):
        return len(self.images)

    # 根据索引获取数据集中的一个元素
    def __getitem__(self, idx):
        if self.use_path:
            # 如果使用路径加载图像，则调用 pil_loader 加载图像并进行转换
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            # 否则，将 numpy 数组转换为 PIL 图像并进行转换
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]  # 获取对应的标签

        return idx, image, label  # 返回索引、图像和标签


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    """返回对应的数据集实例"""
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()  # -> data.py
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
