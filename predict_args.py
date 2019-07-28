import argparse

def predict_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path',
                        help="Path to the image file")

    parser.add_argument('checkpoint',
                        default='checkpoint.pth',
                        help="Path to the model checkpoint")

    parser.add_argument('--top_k',
                        dest='top_k',
                        default=5,
                        help="Return top K most likely classes",
                        type=int)

    parser.add_argument('--category_names',
                        dest='category_names',
                        default='cat_to_name.json',
                        help='Use a mapping of categories to real names: '
                        )

    parser.add_argument('--gpu', action='store_true',
                        dest='gpu',
                        help='Activate GPU')

    result = parser.parse_args()

    return result