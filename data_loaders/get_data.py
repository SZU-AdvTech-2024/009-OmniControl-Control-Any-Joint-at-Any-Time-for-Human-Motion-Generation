# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    # TODO: step 2.0.为读取细粒度数据新增一类get_dataset_class的方法，传入超参数--dataset为detailed_text
    elif name == "detailed_text":  
        from data_loaders.humanml.data.dataset import DetailedTextDataset
        return DetailedTextDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit", "detailed_text"]:    # TODO: step 1.0.这里的name指的是数据集名称，添加我需要的细粒度数据集，转入到t2m_collate对字典进行整合处理
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', control_joint=0, density=100):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "detailed_text"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, control_joint=control_joint, density=density)
    else:
        dataset = DATA(split=split, num_frames=num_frames) 
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', control_joint=0, density=100):
    dataset = get_dataset(name, num_frames, split, hml_mode, control_joint, density)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate,
    )

    return loader