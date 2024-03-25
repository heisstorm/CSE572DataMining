from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import numpy as np
import os

# 假设 WORK_DIR 和 train_labels 已经被定义和加载
# WORK_DIR = '../input/cassava-leaf-disease-classification'
# train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

# 计算类权重
labels = train_labels['label'].values
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(labels),
                                                 y=labels)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

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
    preprocessing_function=None,  # 可以插入预处理函数
    validation_split=0.2  # 验证集比例
)

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

# 定义改进后的模型
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

# 调整目标大小
TARGET_SIZE = 64  # 或者根据您的数据集和GPU内存调整
input_shape = (TARGET_SIZE, TARGET_SIZE, 3)

# 初始化和编译模型
model = create_model(input_shape)

# 检查点和提前停止
model_save = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

# 模型训练
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weights_dict,  # 使用类别权重
    callbacks=[model_save, early_stop, reduce_lr]
)

