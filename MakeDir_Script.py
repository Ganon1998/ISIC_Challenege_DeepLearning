import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import shutil

seed = 1
random.seed(seed)
directory = "D:/ISIC/images/"
train = "D:/data/train/"
test = "D:/data/test/"
validation = "D:/data/validation/"

os.makedirs(train + "benign/")
os.makedirs(train + "malignant/")
os.makedirs(test + "benign/")
os.makedirs(test + "malignant/")
os.makedirs(validation + "benign/")
os.makedirs(validation + "malignant/")

test_examples = train_examples = validation_examples = 0

# open labels file and reads from the 1st row on
for line in open("D:/ISIC/ISIC_2019_Training_GroundTruth.csv").readlines()[1:]:
    split_line = line.split(",")
    img_file = split_line[0]
    benign_malignant = split_line[1]

    # this will be used to determine how many test and validation sets should have
    random_num = random.random()

    if random_num < 0.8:
        location = train
        train_examples +=1

    elif random_num < 0.9:
        location = validation
        validation_examples +=1

    else:
        location = test
        test_examples += 1


    if int(float(benign_malignant)) == 0:
        shutil.copy(
            "D:/ISIC/images/ISIC_2019_Training_Input/" + img_file + ".jpg",
            location + "benign/" + img_file + ".jpg",
        )
    elif int(float(benign_malignant)) == 1:
        shutil.copy(
            "D:/ISIC/images/ISIC_2019_Training_Input/" + img_file + ".jpg",
            location + "malignant/" + img_file + ".jpg",
        )
