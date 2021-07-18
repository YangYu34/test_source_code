from PIL import Image
import face_recognition

# 이미지 파일 숫자 배열(numpy)로 가져옴
image = face_recognition.load_image_file("img01.jpg")

'''
기본 HOG 기반 모델을 사용하여 이미지에서 모든 면을 찾는다.

*HOG 기반 모델이란? (Histogram of Oriented Gradient)
: 대상 영역을 일정 크기의 셀로 분할하고, 각 셀마다 edge 픽셀(gradient magnitude가 일정 값 이상인 픽셀)들의
방향에 대한 히스토그램을 구한 후 이들 히스토그램 bin 값들을 일렬로 연결한 벡터이다.

이 방법은 꽤 정확하지만 GPU를 사용하지 않고, CNN 모델만큼 정확하진 않다.
cnn모델을 사용한 코드: find_faces_in_picture_cnn.py
'''

face_locations = face_recognition.face_locations(image)

print("사진에서 얼굴이 {}개 발견되었습니다.".format(len(face_locations)))

for face_location in face_locations:

    # 이미지의 각 얼굴 위치값 출력
    top, right, bottom, left = face_location
    print("얼굴 픽셀 위치 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show() #화면에 보여줌

