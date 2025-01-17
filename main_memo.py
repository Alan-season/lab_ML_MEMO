import json
import argparse
from trainer import train

def main():
    # 解析命令行参数
    args = setup_parser().parse_args()
    
    # 根据模型名称读取配置文件参数
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    
    # 将命令行的参数转换为字典，与模型的参数合并，传入训练函数
    args = vars(args)
    param.update(args)
    train(param)    # -> trainer.py


def load_json(settings_path):
    """加载模型参数"""
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--memory_size','-ms',type=int, default=2000)
    parser.add_argument('--init_cls', '-init', type=int, default=10)
    parser.add_argument('--increment', '-incre', type=int, default=10)
    parser.add_argument('--model_name','-model', type=str, default=None, required=True)
    parser.add_argument('--convnet_type','-net', type=str, default='resnet32')
    parser.add_argument('--prefix','-p',type=str, help='exp type', default='benchmark', choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--device','-d', nargs='+', type=int, default=[0,1,2,3])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true",)
    
    # Added in MEMO
    parser.add_argument('--train_base',action='store_true')
    parser.add_argument('--train_adaptive',action='store_true')

    # 初始训练阶段
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr','cosine'])
    parser.add_argument('--init_epoch', type=int, default=200)
    parser.add_argument('--t_max', type=int, default=None)
    parser.add_argument('--init_lr', type=float, default=0.1)
    parser.add_argument('--init_milestones', type=list, default=[60,120,170])
    parser.add_argument('--init_lr_decay', type=float, default=0.1)
    parser.add_argument('--init_weight_decay', type=float, default=0.0005)
    
    # 增量更新阶段
    parser.add_argument('--epochs', type=int, default=170)
    parser.add_argument('--lrate', type=float, default=0.1)
    parser.add_argument('--milestones', type=list, default=[80,120,150])
    parser.add_argument('--lrate_decay', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=2e-4)

    parser.add_argument('--alpha_aux', type=float, default=1.0)
    return parser


if __name__ == '__main__':
    main()
