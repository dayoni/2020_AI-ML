# 2020_AI-ML 프로젝트 개요

## 데이터 집합 소개 및 출처
------------
본 프로젝트는 위스콘신 주립 대학교 병원에서 1989년 ~ 1991년 사이에 수집된 유방암 조직검사 결과를 이용하여 유방암을 진단하는 인공지능 시스템을 설계한다. 본 프로젝트를 위해 9가지의 측정 데이터와 양성 또는 악성으로 판단된 결과 데이터가 제공되며 그 데이터의 총 개수는 683개이다. 이번에 설계한 분류 시스템은 크게 5단계를 거쳐 설계되었다.

+ 데이터의 출처
	+ UCI Machine Learning Repository
+ 데이터 제공자
	+ Dr. WIlliam H. Wolberg (physician)
		University of Wisconsin Hospitals
		Madison, Wisconsin
		USA
	+ Donor: Olvi Mangasarian (mangasarian@cs.wisc.edu)
		Received by David W. Aha (aha@cs.jhu.edu)
		-- Date: 15 July 1992

## 프로젝트 설명
------------
+ 모델의 설계 및 구조와 흐름 
	+ 본 프로젝트는 9입력 2클래스의 분류를 위한 시스템을 설계하는 것이 목적
		1. 모델 : 시그모이드 함수(Sigmoid function)를 이용한 로지스틱 회귀 모델(Logistic-Regression)
		2. 비용함수 : 교차 엔트로피 오차(CEE : Cross-Entropy_Error)
		3. 최적화 : 켤레 경사 하강법(conjugate gradient method)
		4. 검증 : K-Fold 교차검증(K-Fold Cross Validation)

## 구조
------------
+ Breast_cancer_data : 훈련 데이터 집합 (Training Data set)
+ Sample_test_data : 테스트 데이터 집합(Test Data set)
+ breastCancerClassification_dayoni : 9입력 2클래스 분류기
+ parameter : 비용함수인 CEE의 매개변수의 값
