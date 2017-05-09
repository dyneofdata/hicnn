import model
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('chromosome_number', type = str,
                        help = 'Specify the number of the human chromosome, e.g. 1, 22, X.')
    parser.add_argument('cell_type', type = str,
                        help = 'Specify the cell type, e.g. Gm12878.')
    parser.add_argument('-l', '--load_only', action='store_true',
                        help = 'To only load an existing model without retraining, use -l or --load_only flag.')
    return parser.parse_args()

def main(args):
    hicnn = model.Model(args.chromosome_number, args.cell_type)
    if (not args.load_only):
        hicnn.train()
    hicnn.test()

if __name__ == '__main__':
    main(parse_args())
