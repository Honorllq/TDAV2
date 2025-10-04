import os
from .utils import Datum, DatasetBase, listdir_nohidden

from .imagenet import ImageNet

TO_BE_IGNORED = ["README.txt"]
template = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

class ImageNetA(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-a"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # 修复：图片直接在dataset_dir下，不在嵌套的imagenet-a子目录
        # 检查两种可能的结构
        nested_dir = os.path.join(self.dataset_dir, "imagenet-a")
        if os.path.isdir(nested_dir) and any(d.startswith('n0') for d in os.listdir(nested_dir) if os.path.isdir(os.path.join(nested_dir, d))):
            self.image_dir = nested_dir
        else:
            self.image_dir = self.dataset_dir
        self.template = template

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(test=data) 

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        # 过滤掉非目录项（如classnames.txt文件）
        folders = [f for f in folders if f not in TO_BE_IGNORED and os.path.isdir(os.path.join(image_dir, f))]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items