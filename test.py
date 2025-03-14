import argparse


parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')

parser.add_argument('--pos_density_only', default=False, type=bool, help='use density difference')

args = parser.parse_args()

print(args.pos_density_only)