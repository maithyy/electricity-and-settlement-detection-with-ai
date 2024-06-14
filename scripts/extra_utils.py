import numpy as np
import tifffile as tiff

train_class_counts = [0, 0, 0, 0]
test_class_counts = [0, 0, 0, 0]

def compute_class_freqs():

    for i in range(1, 61):
        curr_ground_truth_path = f"data/raw/Train/Tile{i}/groundTruth.tif"
        img = tiff.imread(curr_ground_truth_path)
        img = img - 1

        for class_idx in range(len(train_class_counts)):
            train_class_counts[class_idx] += np.sum(img == class_idx)

    print("Train class counts", train_class_counts)
    total_counts = sum(train_class_counts)
    print("Train relative frequencies", [round(class_count/total_counts, 3) for class_count in train_class_counts])

    for i in range(1, 20):
        curr_ground_truth_path = f"data/raw/Test/Tile{i}/groundTruth.tif"
        img = tiff.imread(curr_ground_truth_path)
        img = img - 1

        for class_idx in range(len(test_class_counts)):
            test_class_counts[class_idx] += np.sum(img == class_idx)

    print("Test class counts", test_class_counts)
    total_test_counts = sum(test_class_counts)
    print("Test relative frequencies", [round(class_count/total_test_counts, 3) for class_count in test_class_counts])

def combine_classes_and_save_gt():
    for i in range(1, 61):
        curr_ground_truth_path = f"data/raw/Train/Tile{i}/groundTruth.tif"
        img = tiff.imread(curr_ground_truth_path)
        img[img == 1] = 3   # combine settlements
        img[img == 2] = 4   # combine non-settlements

        img[img == 3] = 2   # set settlements to be class 2 (later 1)
        img[img == 4] = 1   # set non-settlements to be class 1 (later 0)
        tiff.imwrite(f"data/raw/Train/Tile{i}/groundTruth_combined.tif", img)

if __name__ == '__main__':
    combine_classes_and_save_gt()
