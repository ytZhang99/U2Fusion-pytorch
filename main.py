import torch
from train import Train, Test
from option import args

torch.manual_seed(args.seed)


def main():
    if args.test_only:
        t = Test()
        t.test()
    else:
        t = Train()
        t.train()


if __name__ == '__main__':
    main()
