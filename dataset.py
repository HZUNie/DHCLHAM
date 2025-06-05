from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset#将传入的数据集 dataset 赋值给实例变量 data_set，这样类的其他方法就可以访问这个数据集了。
        self.nums = opt.validation# opt.validation 从 opt 对象中获取 validation 属性，并将其赋值给实例变量 nums，这通常表示验证集的数量或交叉验证的组数。

    def __getitem__(self, index):#它使得类的实例可以通过索引来访问，这符合Python的序列协议
        return (self.data_set['ID'], self.data_set['IM'],
                self.data_set['md'][index]['train'], self.data_set['md'][index]['test'],
                self.data_set['md_p'], self.data_set['md_true'],
                self.data_set['independent'][0]['train'],self.data_set['independent'][0]['test'])

    def __len__(self):
        return self.nums



