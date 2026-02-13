import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

SOURCE_DIR = "dataset/train"                  
TRAIN_OUTPUT = "dataset/train_new"            
VAL_OUTPUT = "dataset/validation"             
VALIDATION_SPLIT = 0.20                       
SEED = 42

def create_class_folders(base_path, class_names):
    """Создаёт подпапки для каждого класса"""
    for cls in class_names:
        (base_path / cls).mkdir(parents=True, exist_ok=True)


def split_dataset():
    source_path = Path(SOURCE_DIR)
    train_path = Path(TRAIN_OUTPUT)
    val_path = Path(VAL_OUTPUT)

    if train_path.exists():
        shutil.rmtree(train_path)
    if val_path.exists():
        shutil.rmtree(val_path)

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    class_files = defaultdict(list)

    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            files = list(class_dir.glob("*.*"))
            random.Random(SEED).shuffle(files) 
            class_files[class_name] = files

    class_names = sorted(class_files.keys())
    print(f"Найдено классов: {len(class_names)}")
    print("Классы:", class_names)

    create_class_folders(train_path, class_names)
    create_class_folders(val_path, class_names)

    total_files = 0
    total_train = 0
    total_val = 0

    print("\nРазбиение по классам:")
    print(f"{'Класс':<12} {'Всего':>6} {'Train':>6} {'Val':>6} {'% val':>6}")

    for class_name, files in class_files.items():
        n_total = len(files)
        n_val = int(n_total * VALIDATION_SPLIT)
        n_train = n_total - n_val


        val_files = files[:n_val]
        train_files = files[n_val:]


        for f in val_files:
            shutil.copy2(f, val_path / class_name / f.name)


        for f in train_files:
            shutil.copy2(f, train_path / class_name / f.name)

        total_files += n_total
        total_train += n_train
        total_val += n_val

        print(f"{class_name:<12} {n_total:>6} {n_train:>6} {n_val:>6} {n_val/n_total*100:>6.1f}%")

    print("\n" + "="*50)
    print(f"Всего файлов:     {total_files:>6}")
    print(f"В train_new:      {total_train:>6} ({total_train/total_files*100:.1f}%)")
    print(f"В validation:     {total_val:>6} ({total_val/total_files*100:.1f}%)")
    print("Готово!")


if __name__ == "__main__":
    print("Запуск разделения датасета FER-2013...")
    print(f"Источник:          {SOURCE_DIR}")
    print(f"Train →            {TRAIN_OUTPUT}")
    print(f"Validation →       {VAL_OUTPUT}")
    print(f"Доля валидации:    {VALIDATION_SPLIT*100:.0f}%")
    print("-" * 60)

    split_dataset()