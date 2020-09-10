from pathlib import Path
BASE_DIR = Path('.')
OUTPUTS_DIR=Path('~/dataset/Shang')  / 'outputs'
config = {
    # 'data_dir': BASE_DIR / 'dataset/lcqmc',
    # 'data_dir': BASE_DIR / 'dataset',
    # 'data_dir': Path('/media/u/t1/dataset/Albert') / 'dataset',
    # 'data_dir': Path('/media/u/t1/dataset/albert_pytorch') / 'dataset',
    'data_dir': Path('~/dataset/Shang') / 'dataset',

    'outputs': OUTPUTS_DIR,
    'log_dir': OUTPUTS_DIR / '/logs',
    'figure_dir': OUTPUTS_DIR / "figure",
    'checkpoint_dir': OUTPUTS_DIR / "checkpoints",
    'result_dir': OUTPUTS_DIR / "result",

    # 'bert_dir':BASE_DIR / 'pretrain/pytorch/albert_base_zh',
    # 'albert_config_path': BASE_DIR / 'configs/albert_config_base.json',
    'albert_config_path': BASE_DIR / 'configs/albert_config_shang.json',
    'albert_vocab_path': BASE_DIR / 'configs/vocab_shang.txt',
    'albert_spliter_path': BASE_DIR / 'configs/spliter_cn.txt',
    'pretrain_dir':Path("~/data/self/comment2019zh_corpus")
}

