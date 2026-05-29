import os
import shutil
import random

def split_dataset(source_dir, output_dir, split_ratio=0.8):
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        cls_train = os.path.join(train_dir, cls)
        cls_test = os.path.join(test_dir, cls)

        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_test, exist_ok=True)

        images = [f for f in os.listdir(cls_source) if os.path.isfile(os.path.join(cls_source, f))]
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(cls_train, img))
            
        for img in test_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(cls_test, img))

        print(f"[{cls}]: {len(train_images)} train, {len(test_images)} test")

if __name__ == "__main__":
    SOURCE = "ASL_dataset"
    OUTPUT = "ASL_dataset_split"
    print("Splitting dataset into train and test...")
    split_dataset(SOURCE, OUTPUT)
    print("Dataset split complete!")
