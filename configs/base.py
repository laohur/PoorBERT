from pathlib import Path
BASE_DIR = Path('.')
config = {
    # 'data_dir': BASE_DIR / 'dataset/lcqmc',
    'data_dir': BASE_DIR / 'dataset',
    # 'data_dir': Path('/media/u/t1/dataset/Albert') / 'dataset',
    # 'data_dir': Path('/media/u/t1/dataset/albert_pytorch') / 'dataset',

    'log_dir': BASE_DIR / 'outputs/logs',
    'figure_dir': BASE_DIR / "outputs/figure",
    # 'outputs': BASE_DIR / 'outputs',

    'outputs':  Path('/media/u/t1/dataset/Albert') / 'outputs',
    'checkpoint_dir': BASE_DIR / "outputs/checkpoints",
    'result_dir': BASE_DIR / "outputs/result",

    'bert_dir':BASE_DIR / 'pretrain/pytorch/albert_base_zh',
    'albert_config_path': BASE_DIR / 'configs/albert_config_base.json',
    'albert_vocab_path': BASE_DIR / 'configs/vocab.txt'
}

