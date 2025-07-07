import torch as t
import warnings


class Config(object):
    # global

    data_path = '/home/postphd/SJD/cas13d_data/final_data/new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
    
    data_root = './data'
    small_data = False

    gpu_ids = [0]

    dataset = 'OnehotLoader30'
    model = 'm3'
    lr = 1e-3
    lr_policy = 'linear'
    val_fold = 'rand'

    load_model = None

    # training parameter
    batch_size = 64
    max_epoch = 100
    loss_function = 'LogCoshLoss'

    # model parameter
    use_struct = True
    use_prob = True
    use_mfe1 = True
    use_mfe2 = True

    def _print(self):
        print('---------------user config:------------------')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
        print('---------------------------------------------')

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

cfg = Config()