from models.memo import MEMO

def get_model(model_name, args):
    """根据模型名初始化模型并返回"""
    name = model_name.lower()
    if name == 'memo':
        return MEMO(args)   # -> memo.py
    else:
        assert 0
