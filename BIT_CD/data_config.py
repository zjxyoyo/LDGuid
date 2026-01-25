
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = '/home/zjxeric/projects/def-bereyhia/zjxeric/data/LEVIR'
        elif data_name == 'SVCD':
            self.root_dir = '/home/zjxeric/projects/def-bereyhia/zjxeric/data/SVCD'
        elif data_name == 'WHU_CD':
            self.root_dir = '/home/zjxeric/projects/def-bereyhia/zjxeric/data/WHU_CD'
        elif data_name == 'Fire':
            self.root_dir = '/home/zjxeric/projects/def-bereyhia/zjxeric/data/CaBuAr_12ch'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

