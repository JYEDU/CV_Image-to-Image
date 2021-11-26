# 2021 Computer Vision Term Project
4개 분야 선정하여 베이스라인 찍기
## My Repository
1. [2D Objection Detection](https://github.com/JYEDU/CV_YOLOv5)
2. [Scene Text Recognition](https://github.com/JYEDU/CV_Scene_Text_Recognition)
3. [Super Resolution](https://github.com/JYEDU/CV_Super_Resolution)
4. [Image to Image(Ther2RGB)](https://github.com/JYEDU/CV_Image-To-Image)


## Image to Image (Ther2RGB)

1. Challenge Leaderboard
    - [LINK](http://203.250.148.129:3088/web/challenges/challenge-page/39/overview)
    
2. Image to Image Translation (I2IT) 챌린지 개요
    - Image to Image Translation(I2IT)는 A 도메인 영상을 B 도메인으로 변환시키는 작업을 의미함
    - I2IT task 중 열화상 영상(Thermal image)를 입력으로 하여 RGB 영상(Visible image)를 만드는 것이 챌린지의 목표
    - 열화상 영상은 적외선 파장들 중 가장 긴 파장대에 속하기 때문에 LWIR(Long Wave Infrared)라고 부르기도 함    
    - 열화상 카메라는 단어 그대로 물체에서 나오는 열을 포착하여 촬영하기에 빛이 없는 야간이나, 앞을 보기 힘든 기상환경(소나기, 폭설 등)에서도 강인하게 물체를 촬영할 수 있으며, 이러한 장점 덕분에 자율주행 차량이나 무인 지게차 등 다양한 실내 외 환경에서 사용될 수 있음

![image](https://user-images.githubusercontent.com/87462769/143511778-5a9c5a77-ddd5-4911-8b09-e3a3284a53e2.png){: .center}

3. 방법론 
    - 열화상 영상을 RGB 영상으로 변환하는 작업은 Image to Image Translation task에 속하기 때문에 해당 분야에서 일반적으로 사용하는 Generative Adversarial Network(GAN) 중 하나인 pix2pix 네트워크가 베이스라인으로 설정됨
    - GAN은 A 도메인을 입력으로 B 도메인에 가짜 영상을 생성하는 생성 모델과 생성 모델이 만든 가짜 영상과 진짜 영상을 보고 무엇이 진짜인지 구분하는 판별 모델까지 총 2개로 이루어져있음
    - 입력 영상 x를 생성모델(G)에 넣어 y도메인과 유사한 가짜영상(G(x))을 생성함
    - 원래의 입력 영상 x와 가짜 영상 G(x)를 concat하여 Decoder의 입력으로 태워주게 됨
    - Decoder는 진짜 영상과 가짜 영상을 구분하기 위한 classification 문제를 열심히 학습하게 되며, Generator는 Discriminator가 진짜와 가짜를 구분 못하는 방향으로 설계된 loss를 통하여 학습하게 됨
    - 베이스라인 코드는 Pix2PixHD 기반으로 작성되었으며, 생성모델만 간단하게 U-Net 구조의 네트워크 사용

4. 데이터 셋
    - I2IT task는 크게 Paired한 데이터 셋과 Unpaired 데이터 셋으로 구분할 수 있음
    - Paired Dataset은 x도메인과 y도메인이 마치 동일한 구도에서 촬영된 것처럼 같은 high frequency(edge,structure 등) 성분들을 가지고 있으며 low frequency(texture, color) 성분만이 다른 데이터 셋을 의미함
    - Unpaired Dataset의 경우 low frequency 성분 뿐만 아니라 high frequency 성분들도 다르기 때문에 Paired 데이터 셋보다 학습이 어려운 편에 속함
    - 이번 챌리지에서 수행되는 Thermal2RGB task는 동일한 장면을 촬영한 데이터셋이므로 Paired 데이터 셋에 속함
    - 하드웨어를 통해 두영상의 align을 맞추었기 때문에 대부분의 경우 정확하게 align이 맞지만 차량에서 촬영한 데이터 셋이다보니, 특수한 경우(방지턱 넘기, 급작스런 커브 등)에서는 misalign문제가 발생하기도 함
    - 대부분의 경우 align이 맞기 때문에 Paired 데이터 셋이라는 가정하에 Paired 데이터 셋 전용 모델을 베이스라인으로 설정
    - 데이터셋의 구성 
        - Train Dataset : 열화상 영상 총 2661장과 align이 맞는 RGB 영상 2661장
        - Test Dataset : 열화상 영상 총 1276장

![image](https://user-images.githubusercontent.com/87462769/143512190-1e0ba4bd-f01a-4537-af4d-c11eb3d74480.png){: .center}


5. 참고자료
    - [[Paper](https://arxiv.org/pdf/1611.07004.pdf)] Image-to-Image Translation with Conditional Adversarial Networks
    - [[Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)] pytorch-CycleGAN-and-pix2pix
    - [[Github](https://github.com/sjmin99/Ther2RGB-Translation)] Ther2RGB-Translation
    - [[Youtube](https://www.youtube.com/watch?v=z3HnZAOMbaQ&list=PL1xKqHsVFgvnM3zhBkbTZy5l_13x5R3Jq&index=13)] 컴비 텀프로젝트 소개영상_18011784_신정민
    - [[Paper](https://arxiv.org/abs/1703.10593)] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

## 원복 과정에 대한 챌린지 참여 파일 : 
1. eval.ai 리더보드상의 기록 캡쳐본
2. 베이스라인을 찍기 위한 나의 repository
3. 제출 과정에 대한 동영상 링크(베이스라인 알고리즘 설명 및 베이스라인을 찍기 위한 과정을 설명)

1. 
