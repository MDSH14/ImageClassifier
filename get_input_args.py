import argparse
def get_input_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help="data directory(required)")
    parser.add_argument('--save_dir', default = '', type = str, help = 'Path to the folder of flower images')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN Model Architecture')
    parser.add_argument('--learning_rate', type = int, default = 0.001, help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, default = 255, help = 'Hidden units')
    parser.add_argument('--epochs', type = int, default = 3, help = 'No. of epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU?')
    in_args = parser.parse_args()
    
    return in_args