import argparse
def get_predict_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="data directory(required)")
    parser.add_argument('checkpoint', type = str, help = 'Path to checkpoint')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top K')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json')

    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU?')
    in_args = parser.parse_args()
    
    return in_args