# EyeNet

## 개요
 
Siraj Raval의 의학 분류에 관한 [유튜브 강의](https://youtu.be/DCcmFXXAHf4)를 위한 코드입니다.

## 딥러닝을 활용한 당뇨병성 망막증 진단

## 목표

당뇨병성 망막증은 선진국 노동 인구의 실명을 유발하는 가장 큰 원인입니다. 9,300만 명 이상에게 영향을 끼치는 것으로 추산됩니다.

정확하고 자동화된 당뇨병성 망막증 진단의 필요성은 지속적으로 논의되었으며, 사진 분류와 패턴 인식, 그리고 머신러닝을 이용한 이전의 노력들이 좋은 발전을 만들었습니다. 목표는 안구 사진을 입력값으로 받아 실제 진단에 사용될 가능성을 가진 새로운 모델을 만들어내는 것입니다.

본 프로젝트의 동기는 두 가지입니다:

* 사진 분류는 빅데이터 분류와 더불어 몇 년간 유망 분야였습니다.

* 환자들이 아래 사진과 같이 안구를 스캔하고, 의사들에게 진단받고, 다음 약속을 잡는 데 많은 시간이 낭비됩니다. EyeNet은 실시간으로 사진을 처리함으로써 진단과 예약을 같은 날에 할 수 있도록 할 것입니다.  


<p align = "center">
<img align="center" src="images/readme/dr_scan.gif" alt="Retinopathy GIF"/>
</p>


## 목차
1. [데이터](#데이터)
2. [예비 데이터 분석](#예비-데이터-분석)
3. [전처리](#전처리)
    * [EC2로 사진 저장](#EC2로-사진-저장)
    * [사진 자르기 & 크기 변경](#사진-자르기-및-크기-변경)
    * [사진 회전 및 반전](#사진-회전-및-반전)
4. [CNN 설계](#CNN-설계)
5. [결과](#결과)
6. [다음 단계](#다음-단계)
7. [참고 문헌](#참고-문헌)


## 데이터

데이터는 [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection)에서 가져왔습니다. 하지만 이는 대표적인 Kaggle 데이터의 예시가 아닙니다. 대부분의 Kaggle competition에서는 데이터가 이미 정리되어 있어, 데이터 과학자들이 전처리할 부분이 거의 없습니다. 이 데이터는 예외입니다. 

모든 사진은 제각기 다른 사람으로부터 다른 카메라를 사용하였으며 다른 크기입니다. [전처리](#전처리) 부분에 관하여, 이 데이터는 아주 노이즈가 많으며 모든 사진을 모델 학습에 사용 가능한 형태로 만들기 위해서는 여러 단계의 전처리 과정을 거쳐야 합니다.

학습 데이터는 35,126장의 사진으로 이루어져 있으며, 전처리 과정 중에 늘어납니다.


## 예비 데이터 분석

가장 처음으로 분석할 것은 학습 레이블입니다. 예측해야 할 범주는 5개인데 반해, 아래 그래프는 원래 데이터의 각 범주 간 심각한 불균형이 있음을 보여줍니다.

<p align = "center">
<img align="center" src="images/eda/DR_vs_Frequency_tableau.png" alt="EDA - Class Imbalance" height="458" width="736" />
</p>

원래 학습 데이터에서 25,810장의 사진은 망막증을 가지지 않은 것으로 분류된 반면, 9,316장은 망막증을 가진 것으로 분류되었습니다. 

이런 범주 간 불균형 때문에, 모델을 학습시키기 위해서는 [전처리](#전처리) 과정에서 불균형을 수정하기 위한 단계가 필요합니다.

또한, 안구 사진 간 분산이 너무 높습니다. 첫 두 줄의 사진은 class 0(망막증 없음)이고, 다음 두 줄은 class 4(증식성 망막증)입니다. 


<p align = "center">
<img align="center" src="images/readme/No_DR_white_border_1.png" alt="No DR 1"/>
<img align="center" src="images/readme/No_DR_white_border_2.png" alt="No DR 2"/>
<br></br>
<img align="center" src="images/readme/Proliferative_DR_white_border_1.png" alt="Proliferative DR 1"/>
<img align="center" src="images/readme/Proliferative_DR_white_border_2.png" alt="Proliferative DR 2"/>
</p>



## 전처리

전처리 과정은 다음과 같습니다:


1. [Download script](src/download_data.sh)를 이용하여 모든 사진을 EC2로 저장합니다.
2. [Resizing script](src/resize_images.py)와 [preprocessing script](src/preprocess_images.py)를 이용하여 모든 사진을 자르고 크기를 변경합니다.
3. [Rotation script](src/rotate_images.py)를 이용하여 모든 사진을 회전하고 반전시킵니다.
4. [Conversion script](src/image_to_array.py)를 이용하여 모든 사진을 NumPy 배열로 변환합니다.

### EC2로 사진 저장
사진들은 Kaggle CLI를 통해 저장됩니다. EC2에서 이를 실행하면 사진을 모두 저장하는 데 30분 정도 걸릴 것입니다. 그럼 모든 사진들이 각자의 폴더에 저장되고 압축이 해제될 것입니다. 데이터는 모두 합쳐 총 35 GB 입니다.

### 사진 자르기 및 크기 변경
모든 사진들은 `256 * 256`으로 조정되어 있습니다. 학습에 오래 걸리더라도, `128 * 128`보다 이 크기의 사진에 더욱 자세한 정보가 많습니다.

추가적으로, 403장의 사진이 학습셋에서 사라집니다. 이 사진들은 색 공간이 없어 Scikit-Image가 크기 변경 과정에서 경고한 사진들입니다. 따라서 완전히 검은색인 사진은 학습 데이터에서 제거됩니다. 

### 사진 회전 및 반전
모든 사진들은 회전 및 반전됩니다. 망막증이 없는 사진은 반전되고, 망막증이 있는 사진은 반전 및 90도, 120도, 180도, 그리고 270도 회전됩니다.

첫 사진은 검은 경계를 따라 두 쌍의 눈을 보여줍니다. 사진 자르기와 회전이 어떻게 대부분의 노이즈를 제거하는지 확인하세요.

![Unscaled Images](images/readme/sample_images_unscaled.jpg)
![Rotated Images](images/readme/17_left_horizontal_white.jpg)

회전과 반전이 끝나면, 망막증을 가진 사진이 몇 천장 더 많은 정도로 범주 간 불균형이 수정됩니다. 총 106,386장의 사진이 신경망에 의해 처리됩니다.


<p align = "center">
<img align="center" src="images/eda/DR_vs_frequency_balanced.png" alt="EDA - Corrected Class Imbalance" width="664" height="458" />
</p>


## CNN 설계

모델에는 백엔드로 텐서플로를 사용하는 Keras가 사용되었습니다. 텐서플로는 Theano보다 더 성능이 좋고 TensorBoard를 이용하여 신경망을 시각화하는 능력이 더 뛰어납니다. 

두 범주를 예측하기 위하여 EyeNet은 3개의 합성곱 층을 사용하며, 각각 32의 깊이를 가지고 있습니다. 최대 풀링 층은 마지막에 (2,2) 크기의 합성곱 층 세 개 모두에 적용됩니다.

풀링이 끝나면 데이터는 크기 128의 단일 고밀도 층에 입력되고, 2개의 소프트맥스 노드를 가진 출력망으로 이동합니다.

![TensorBoard CNN](images/readme/cnn_two_classes_tensorboard.png)

## 결과
EyeNet 분류기는 환자가 망막증을 가졌는지를 판단하기 위해 만들어졌습니다. 현 모델은 다음과 같은 결과를 출력합니다.

| 측정 기준 | 값 |
| :-----: | :-----: |
| 정확도 (학습) | 82% |
| 정확도 (테스트) | 80% |
| 정밀도 | 88% |
| 재현율 | 77% |


그래서, 신경망은 왜 이와 같이 작동하는 것일까요? 범주 간 불균형 외에, 사진 자르기가 신경망의 성능에 아주 큰 도움을 줍니다. 사진 내에 검은 부분이 없기 때문에 신경망이 안구 자체만을 처리할 수 있습니다. 

## 다음 단계
1. 새로운 사진들로 재학습시키기 위해 신경망을 프로그래밍합니다. 이는 흔한 과정이며, 모델을 최적화하기 위해 사용됩니다. 저품질 사진들이 분류기를 지나치게 바꾸는 것을 방지하기 위해 사진들이 분류기에 입력되기 전에 검사할 수 있습니다.

2. Keras 모델을 CoreML로 변환하고 EyeNet iOS 애플리케이션에 배포합니다. CoreML은 애플이 만든 프레임워크로 iOS 장치에 머신러닝을 추가합니다. 이는 파이썬 개발자들이 그들의 모델을 `.mlmodel` 파일로 변환하여 iOS 개발 사이클에 추가할 수 있도록 합니다. 

또한, 이 모델은 로컬 장치에서 분류를 수행할 수도 있습니다. 어플이 실행되기 위해서 별도의 인터넷 연결이 필요하지 않습니다. 이 때문에, 원격으로 EyeNet을 사용하는 것이 아주 쉬워집니다. 


## 참고 문헌

1. [What is Diabetic Retinopathy?](http://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/basics/definition/con-20023311)

2. [Diabetic Retinopathy Winners' Interview: 4th place, Julian & Daniel](http://blog.kaggle.com/2015/08/14/diabetic-retinopathy-winners-interview-4th-place-julian-daniel/)

3. [TensorFlow: Machine Learning For Everyone](https://youtu.be/mWl45NkFBOc)

## 기술 스택
<img align="center" src="images/tech_stack/tech_stack_banner.png" alt="tech_stack_banner"/>

## 크레딧
이 코드의 저작권은 [gregwchase](https://github.com/gregwchase/dsi-capstone)에게 있습니다. 저는 단지 사람들이 쉽게 시작할 수 있도록 만들었을 뿐입니다.
