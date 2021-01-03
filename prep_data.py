import random
from shutil import copyfile
from os import listdir, makedirs, walk, path

#Format choosen: format contains all the images within separate folders named
#after their respective class names
def create_folders(dir):
     # Set up empty folder structure for training and validation sets
    if not path.exists(dir +'dataset/train'):
        makedirs(dir + 'dataset/train')
    if not path.exists(dir + 'dataset/test'):
        makedirs(dir + 'dataset/test')


def split(dir, train_size, seed):
    #copy images to the subdirectories of the 2 classes
    train_counter = 0
    test_counter = 0
    random.seed(seed)
    subdirs = [subdir for subdir in listdir(dir) if not subdir.endswith("dataset")]
    for subdir in subdirs:
        dir_im = dir + subdir +'/'

        for filename in listdir(dir_im):
            if filename.endswith(".png"):
                rand = random.uniform(0, 1)
                if rand <= train_size:
                    src = dir_im + filename
                    filename = subdir +"_"+ filename
                    dst = dir + 'dataset/train/' + filename
                    copyfile(src, dst)
                    train_counter += 1
                else:
                    src = dir_im + filename
                    filename = subdir +"_"+ filename
                    dst = dir + 'dataset/test/' + filename
                    copyfile(src, dst)
                    test_counter += 1
    return train_counter, test_counter

#To test
if __name__ == '__main__':
    dir = 'cell_images/'
    train_size = 0.8
    seed = 123
    create_folders(dir)
    train_counter, test_counter = split(dir, train_size, seed)
    print('done')
    print('Copied ' + str(train_counter) + ' images to train')
    print('Copied ' + str(test_counter) + ' images to test')
