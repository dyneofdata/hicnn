import model
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('chromosome_number', type = str,
                        help = 'Specify the number of the human chromosome, e.g. 1, 22, X.')
    parser.add_argument('cell_type', type = str,
                        help = 'Specify the cell type, e.g. Gm12878.')
    return parser.parse_args()

def main(args):
    hicnn = model.Model(args)
    hicnn.train()
    hicnn.test()

if __name__ == '__main__':
    main(parse_args())
