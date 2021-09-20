import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=str, default='yes', help='whether to use gpu options[yes/no]')
    parser.add_argument('--arch', type=str, default='densenet121', help='architecture [available: densenet, vgg]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=2960, help='hidden units for fc layers (comma separated)')
    parser.add_argument('--input_size', type=int, default=25088, help='input size for fc layers (comma separated)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--image_path', type=str, help='path of image to be predicted')
    parser.add_argument('--arch', type=str, default='densenet121', help='architecture [available: densenet, vgg]')
    parser.add_argument('--hidden_units', type=int, default=2960, help='hidden units for fc layers (comma separated)')
    parser.add_argument('--input_size', type=int, default=25088, help='input size for fc layers (comma separated)')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--saved_model_path' , type=str, default='flower102_checkpoint.pth', help='path of your saved model')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

    return parser.parse_args()
    
