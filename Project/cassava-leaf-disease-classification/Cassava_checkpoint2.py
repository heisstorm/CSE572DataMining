from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import pandas as pd


def create_model(input_shape):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # 添加Dropout
        BatchNormalization(),  # 添加BatchNormalization
        Dense(5, activation='softmax')  # 假设有5个类别
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # 假设 WORK_DIR 和 train_labels 已经被定义和加载
    WORK_DIR = '../cassava-leaf-disease-classification'
    train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    BATCH_SIZE = 64
    TARGET_SIZE = 64
    # 计算类权重
    labels = train_labels['label'].values
    # 数据增强
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        preprocessing_function=None,
        validation_split=0.2
    )
    train_labels.label = train_labels.label.astype('str')

    # 重新定义图像生成器
    train_generator = data_gen.flow_from_dataframe(
        train_labels,
        directory=os.path.join(WORK_DIR, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training"
    )

    validation_generator = data_gen.flow_from_dataframe(
        train_labels,
        directory=os.path.join(WORK_DIR, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation"
    )


    # 调整目标大小
    input_shape = (TARGET_SIZE, TARGET_SIZE, 3)

    # 初始化和编译模型
    model = create_model(input_shape)

    # 检查点和提前停止
    model_save = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=0.001, mode='min')

    # 模型训练
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_labels)*0.8 / BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(train_labels)*0.24 / BATCH_SIZE,
        callbacks=[model_save, early_stop, reduce_lr]
    )
