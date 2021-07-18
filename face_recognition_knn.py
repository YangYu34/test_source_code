"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

# -*- coding: UTF-8 -*-

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

"""
    
    얼굴 인식을 위해 knn (k-nearest neighbors) 분류기를 훈련시킨다.
    <<train 함수의 parameters 소개>
    1. 'train_dir': 유명인에 대한 각 하위 디렉토리를 포함하는 디렉토리

    <train_dir 트리 구조>
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    2. model_save_path: (옵션) 디스크에 저장된 knn 모델 경로
    3. n_neighbors: (옵션) 분류에서 가중치를 부여할 인접값 수. 지정되지 않으면 자동으로 선택됨.
    4. knn_algo: (옵션) knn을 지원하는 기본 데이터 구조. 기본값은 ball_tree임.
    5. verbose: 학습 진행 상황 보여줄 건지 지정하는 부분 (0은 아무것도 안 보여줌 / 1은 epoch, loss값 다 보여줌)

    return: 주어진 데이터에 대해 훈련 knn 분류기 리턴함.

    """

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []

    # 훈련셋에 있는 사람들 훈련함. Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        #현재 사용자에 대한 각 훈련 이미지 반복 Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face"
                        if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # KNN 분류기의 가중치에서 사용한 인접 항목수 결정 Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # KNN 분류기 생성 및 훈련 Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

    

"""
    Recognizes faces in given image using a trained KNN classifier
    훈련된 KNN 분류기를 이용하여 주어진 이미지 속 얼굴 인식하기

    <<predict 함수의 parameters 소개>

    1. X_img_path: 인식될 이미지 경로
    2. knn_clf: (옵션) knn 분류기 객체. 만약 지정하지 않으면 train()함수에서 지정한 model_save_path가 사용됨
    3. model_path: (옵션) 훈련된? knn 분류기 경로(path to a pickled knn classifier.) 
       만약 지정하지 않으면 train()함수에서 지정한 model_save_path가 knn_clf이 됨.
    4. param distance_threshold: (옵션) 얼굴 분류에 대한 임계값(맞다/아니다 분류하는 경계).
       값이 클수록 잘못 분류할 가능성이 커짐

    return: 이미지에서 인식된 얼굴에 대한 '이름', 얼굴 위치값 반환 [(name, bounding box), ...]
    인식되지 않은 사람의 얼굴은 'unknown'이라 반환됨
    """
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 파일에서 이미지 파일 로드하고 얼굴 위치값 찾음. Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # 만약 이미지에서 얼굴이 발견되지 않으면 빈 결과값이 반환됨. If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # 테스트 이미지에서 얼굴들에 대한 인코딩 찾음. Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # KNN 모델 사용하여 테스트 얼굴에서 가장 적합한 일치 항목(이름)을 찾음. Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # 클래스 예측하고 임계값 안에 없는 분류 제거. Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    얼굴 인식 결과를 시각적으로 보여줌.

    <<show_prediction_labels_on_image 함수의 parameters 소개>

    1. param img_path: 인식된 이미지 경로 path to image to be recognized
    2. param predictions: 예측 함수 결과 results of the predict function
    return: 없음.
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("fonts/SeoulNamsanM.ttf", 15) #한글 표현할 수 있는 폰트 사용해야 깨지지 않음.

    for name, (top, right, bottom, left) in predictions:
        # Pillow 모듈 사용하여 얼굴에 box 그려줌 Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # 한글 출력시에는 utf-8로 변환 안해줘도 됨.
        # name = name.encode("UTF-8")

        # 얼굴 밑에 이름(레이블) 달아줌 Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left+6, bottom - text_height - 8), name, font=font, fill=(255, 255, 255, 255))

    # Pillow 문서에 따라 메모리에서 그려주는 라이브러리 제거 Remove the drawing library from memory as per the Pillow docs
    del draw

    # 이미지 결과 보여줌 Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: KNN 분류기 훈련하고 디스크에 저장 Train the KNN classifier and save it to disk
    # 모델이 훈련된 후 저장될 때, 다음 단계로 넘어갈 수 있음. Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: 훈련된 분류기 사용하여 unkown 이미지 예측하기 Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # 훈련된 분류기 모델을 사용하여 이미지에서 모든 사람 찾기 Find all people in the image using a trained classifier model
        # 분류기 파일 이름 또는 분류기 모델 인스턴스를 전달할 수 있음. Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # 콘솔에 결과 출력. Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # 이미지에 결과 중첩하여 표시함. Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
