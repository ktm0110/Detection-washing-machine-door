import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 경로 설정
train_dir = r'C:\Users\ktmth\Desktop\train_up\train'
validation_dir = r'C:\Users\ktmth\Desktop\train_up\validation'
predict_dir = r'C:\Users\ktmth\Desktop\train_up\predict'  # 예측할 이미지들이 있는 폴더 경로

# 이미지 데이터 제너레이터 설정
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# VGG16 모델 불러오기 (최종 분류 레이어 제외)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 새로운 분류 레이어 추가
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# 전체 모델 구성
model = Model(inputs=base_model.input, outputs=predictions)

# 기존 레이어는 학습되지 않도록 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# 모델 학습
history = model.fit(train_generator, epochs=100, validation_data=validation_generator, callbacks=[early_stopping])

# 모델 평가
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy}')

# 손실 그래프
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 예측 함수 정의
def predict_image(model, image_path):
    """
    주어진 이미지에서 세탁기 문이 닫혔는지 열렸는지 예측합니다.

    Parameters:
    model (Model): 학습된 Keras 모델
    image_path (str): 예측할 이미지의 경로

    Returns:
    str: 'Closed' 또는 'Opened'
    """
    img = load_img(image_path, target_size=(224, 224))  # 이미지를 224x224로 로드
    img_array = img_to_array(img)  # 이미지를 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 차원을 추가
    img_array = preprocess_input(img_array)  # 전처리

    prediction = model.predict(img_array)  # 예측
    return 'Closed' if prediction < 0.5 else 'Opened'  # 임계값을 기준으로 결과 반환

# 폴더 내 모든 이미지를 예측하는 함수 정의
def predict_images_in_folder(model, folder_path):
    """
    주어진 폴더 내 모든 이미지에 대해 세탁기 문이 닫혔는지 열렸는지 예측합니다.

    Parameters:
    model (Model): 학습된 Keras 모델
    folder_path (str): 예측할 이미지들이 있는 폴더의 경로
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            result = predict_image(model, image_path)
            print(f'The washing machine door in {filename} is {result}')

# 예측 예시
predict_images_in_folder(model, predict_dir)
