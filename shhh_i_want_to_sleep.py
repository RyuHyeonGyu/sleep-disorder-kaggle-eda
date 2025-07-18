# Sleep Disorder Classification - Exploratory Data Analysis (EDA)

#이 코드는 Kaggle에서 공유된 Amal Yasser 님의 노트북  
#["Shhh I want to sleep"](https://www.kaggle.com/code/amalyasser/shhh-i-want-to-sleep)을 참고하여,  
#개인 학습 목적 및 코드 분석 용도로 일부 커스터마이징한 EDA 코드입니다.

## 주요 변경 사항
#- 코드에 상세한 한글 주석 추가
#- `termcolor` 및 `pandas styling`을 활용해 콘솔 및 테이블 시각적 개선
#- seaborn, matplotlib, plotly를 활용한 다양한 시각화 방법 실습
#- 코드 오류 수정 및 환경에 맞는 스타일 지정 (`plt.style.use` 등)

## 데이터셋
#- `Sleep_health_and_lifestyle_dataset.csv`
#- 원본 출처: [Kaggle Dataset](https://www.kaggle.com/datasets/equbs/sleep-health-and-lifestyle-dataset)

## 참고한 원본 코드
#- Amal Yasser, [Shhh I want to sleep](https://www.kaggle.com/code/amalyasser/shhh-i-want-to-sleep)

# 본 코드는 비상업적, 학습 목적의 참고용으로만 사용됩니다.





#Reading data
import pandas as pd # 데이터 조작 및 분석


#Fixings warnings
import warnings 
warnings.filterwarnings('ignore')


#For mathematical operations
import numpy as np # 수치 계산 라이브러리


#Visualisation
import seaborn as sns # 데이터 시각화 라이브러리 (matplotlib보다 예쁜 그래프)
import plotly.express as px
from termcolor import colored # 터미널 출력 텍스트에 색상과 스타일을 입히는 라이브러리
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.figure_factory as ff
#seaborn, matplotlib은 그래프가 이미지처럼 고정, plotly는 줌인/아웃이 가능 툴팁 표시 그래프를 마우스로 이동할 수 있음

#Data spliting
from sklearn.model_selection import train_test_split
#데이터 분할 기능 학습용과 테스트용으로 나눌 때 사용

sleep_data=pd.read_csv(r'C:\Users\PC2401\project file\archive\Sleep_heath_and_lifestyle_dataset\Sleep_health_and_lifestyle_dataset.csv')

#head() 상위 5개 행 출력
sleep_data.head().style.set_properties(**{'background-color': '#4A235A',
                                          'color': '#E2EEF3'}) #for colored output
#sleep_data.shape 데이터의 크기를 튜플로 가져옴
shape = colored(sleep_data.shape, "magenta",None, attrs=["blink"]) #튜플 스타일링
print('The dimention of data is :',shape)

sleep_data.info() # for empty and type of values
#데이터 타입 구분

#for statistical info
sleep_data.describe().style.background_gradient(cmap='BuPu') #for colored output
#수치형 컬럼 요약 통계 - 평균, 표준편차, 최소값, 최대값
#백그라운드 그라데이션 설정

#for statistical info including string values
sleep_data.describe(include='O').style.set_properties(**{'background-color': '#4A235A',
                                                      'color': '#E2EEF3'}) 
#범주형 데이터 요약 + 스타일링
#include ='O' :문자열 컬럼들만 추려 나타냄


columns_name=colored(sleep_data.columns, 'magenta',None, attrs=["blink"]) #for show names of columns
print(columns_name)

#컬럼명 출력 색 보라색 + 깜빡임 효과 적용


#for colored text output ( Text ,Text colors ,Text highlights , Attributes)
number_of_values=colored(sleep_data.nunique(), "magenta",None, attrs=["blink"])
# 데이터 프레임의 컬럼별로 고유한 값의 개수 계산
              
print(number_of_values) #for number of values of columns


#plt.style.use('seaborn-white') < 코드 진행시 오류 발생 matplolib 라이브러리 사용하려 했으나 seaborn-white 스타일을 찾을 수 없어서
print(plt.style.available) #설치된 스타일 목록을 열어 확인
plt.style.use('seaborn-v0_8-white')
sns.pairplot(data=sleep_data.drop('Person ID',axis=1),hue='Sleep Disorder',palette='mako')
plt.legend()
plt.show()


classes=colored(sleep_data['Sleep Disorder'].unique(), "magenta",None, attrs=["blink"])
print('The outputs from the classification are :',classes)
#'Sleep Disorder'컬럼에 있는 고유한 값들을 배열 형태로 반환
# The outputs from the classification are : [nan 'Sleep Apnea' 'Insomnia']


sleep_data['Sleep Disorder'].value_counts()
#각 클래스의 개수 세기


fig=px.histogram(sleep_data,x='Sleep Disorder', 
                 barmode="group",color='Sleep Disorder',
                 color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 text_auto=True)
#Sleep Disorder에 해당하는 사람이 몇 명인지 비교

#레이아웃 설정
fig.update_layout(title='<b>Distribution of persons have sleep disorder or not</b>..',
                 title_font={'size':25},
                 paper_bgcolor='#EBDEF0',
                 plot_bgcolor='#EBDEF0',
                 showlegend=True)


#y축 격자선 제거
fig.update_yaxes(showgrid=False)


fig.show()
#plotly 라이브러리만 설치했을 경우 그래프 표시 오류 발생
#주피터 환경에서 plotly그래프를 렌더링하려면 추가 패키지인 nbformat 설치 필요


Gender=colored(sleep_data['Gender'].unique(), "magenta",None, attrs=["blink"])
print('The values of Sex column are :',Gender)
#Gender 컬럼에 고유 값을 추출 보라색 + 깜빡이로 출력

sleep_data.groupby('Sleep Disorder')['Gender'].value_counts()
#어떤 수면장애에 남녀가 몇명씩 있는지 출력


sleep_data.groupby('Sleep Disorder')['Gender'].value_counts().plot.pie(autopct ='%1.1f%%',figsize=(15,7),
                                                                       colors=['#4A235A','pink','#4A235A','pink','#4A235A','pink'])
#어떤 수면장애에 남녀가 몇명씩 있는지 시각화 (파이 모양 그래프로 나타내기) 


plt.title('The relationship between (sex) and (Sleep Disorder)')
plt.axis('equal')
plt.show()
#matplotlib 


jobs=colored(sleep_data['Occupation'].unique(), "magenta",None, attrs=["blink"])
print('The types of jobs that exist are :',jobs)


sleep_data.groupby('Sleep Disorder')['Occupation'].value_counts()
#Sleep Disorder(수면장애)로 그룹을 묶고
#각 그룹의 Occupation(직종)의 개수를 세어줌
#수면 장애를 가진 사람들이 어떤 직종에 많은 지 확인 가능

"""
fig=px.treemap(sleep_data,path=[px.Constant('Jobs'),'Sleep Disorder','Occupation'],
               color='Sleep Disorder',
              color_discrete_sequence=['#EBDEF0','#C39BD3','#4A235A'])


fig.update_layout(title='<b>The effect of job on sleep</b>..',
                 title_font={'size':20})


fig.show()
"""

print(sleep_data.columns)
filtered_data = sleep_data.dropna(subset=['Sleep Disorder', 'Occupation'])
# 이 코드를 추가한 이유는 노드에 None이나 Nan값이 포함이 되면 오류가 생기는데 그 오류를 지우기 위해 Sleep Disorder나 Occupation에 Nan이 하나라도 있으면 그 행 전체 제거
fig=px.treemap(filtered_data,path=[px.Constant('Jobs'),'Sleep Disorder','Occupation'],
               color='Sleep Disorder',
              color_discrete_sequence=['#EBDEF0','#C39BD3','#4A235A']) 

fig.update_layout(title='<b>The effect of job on sleep</b>..',
                 title_font={'size':20})


fig.show()



sleep_data.pivot_table(index='Quality of Sleep',columns='Sleep Disorder',values='Sleep Duration',aggfunc='mean').style.background_gradient(cmap='BuPu')
#pivot_table(...).style.background_gradient(...) 실행 시 테이블이 자동으로 렌더링(Jupyter 환경 혹은 IPython환경)


'''
fig=px.sunburst(sleep_data,path=[px.Constant('Sleep quality'),'Sleep Disorder','Quality of Sleep'],
               color='Sleep Disorder',values='Sleep Duration',
              color_discrete_sequence=['pink','#4A235A','#FFF3FD'],
              hover_data=['Gender'])

fig.update_layout(title='<b>The effect of quality of sleep on sleep </b>..',
                 title_font={'size':25})

fig.show()
'''

filtered_data = sleep_data.dropna(subset=['Sleep Disorder', 'Quality of Sleep'])
fig=px.sunburst(filtered_data,path=[px.Constant('Sleep quality'),'Sleep Disorder','Quality of Sleep'],
               color='Sleep Disorder',values='Sleep Duration',
              color_discrete_sequence=['pink','#4A235A','#FFF3FD'],
              hover_data=['Gender'])


fig.update_layout(title='<b>The effect of quality of sleep on sleep </b>..',
                 title_font={'size':25})

#fig.update_layout(font_family="Malgun Gothic") 한글 깨짐 방지 코드

fig.show()



fig = px.violin(sleep_data, x="Sleep Disorder",y='Physical Activity Level',
                 color='Sleep Disorder',
                 color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 violinmode='overlay')
#분포+밀도+중앙값 해석이 어려운 단점               
    
fig.update_layout(title='<b>The effect of activities on sleep </b>..',
                 title_font={'size':25},
                 paper_bgcolor='#EBDEF0',
                 plot_bgcolor='#EBDEF0')

fig.update_yaxes(showgrid=False)
fig.show()




sleep_data.pivot_table(index='Gender',columns='Sleep Disorder',values='Age',aggfunc='median').plot(kind='bar',color={'#FFF3FD','#4A235A','pink'},
                                                                                                   title='Most affected ages in each type of Sleep Disorder',
                                                                                                    label='Age',alpha=.7)


plt.show()



fig=px.ecdf(sleep_data,x='Age',
            color='Sleep Disorder',
            color_discrete_sequence=['white','#4A235A','#C39BD3'])
#누적 비율 기준, 모든 데이터 반영, 비교용

fig.update_layout(title='<b>The effect of ages on sleep </b>..',
                 title_font={'size':25},
                 paper_bgcolor='#EBDEF0',
                 plot_bgcolor='#EBDEF0')


fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()




fig=px.histogram(sleep_data,x='Sleep Disorder',y='Sleep Duration',
                 color='Sleep Disorder',color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 text_auto=True)
#데이터를 구간으로 나누고 빈도를 막대 그래프로 표시


fig.update_layout(title='<b>The effect of Sleep Duration on Sleep Disorder</b> ..',
                  #titlefont={'size': 24,'family': 'Serif'}, titlefont는 더 이상 plotly에서 지원 X
                  title_font={'size': 24,'family': 'Serif'},
                  showlegend=True, 
                  paper_bgcolor='#EBDEF0',
                  plot_bgcolor='#EBDEF0')



fig.update_yaxes(showgrid=False)




fig.show()


fig=px.scatter_3d(sleep_data,x='BMI Category',y='Blood Pressure',z='Heart Rate',
                  color='Sleep Disorder',width=1000,height=900,
                  color_discrete_sequence=['white','#4A235A','#C39BD3'])
#3d시각화 

fig.update_layout(title='<b>The relationship between (BMI Category , Blood Pressure and Heart Rate) and their effect on  Sleep Disorder</b> ..',
                  title_font={'size': 20,'family': 'Serif'},
                  showlegend=True)



fig.show()



sleep_data.pivot_table(index='Stress Level',columns='Sleep Disorder',aggfunc={'Sleep Disorder':'count'}).style.background_gradient(cmap='BuPu')




fig=px.histogram(sleep_data,x='Sleep Disorder',
                 color='Sleep Disorder',
                 facet_col='Stress Level',
                 barmode='group',
                 color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 opacity=.8)


fig.update_layout(title='<b>The effect of Stress Level on Sleep Disorder</b> ..',title_font={'size':30},
                  paper_bgcolor='#EBDEF0',
                  plot_bgcolor='#EBDEF0')



fig.update_yaxes(showgrid=False)
fig.show()



BMI_Category=colored(sleep_data['BMI Category'].unique(), "magenta",None, attrs=["blink"])
print('The values of BMI Category column are :',BMI_Category)



sleep_data.pivot_table(index='BMI Category',columns='Sleep Disorder',aggfunc={'Sleep Disorder':'count'}).style.background_gradient(cmap='BuPu')



sleep_data.pivot_table(index='BMI Category',columns='Sleep Disorder',aggfunc={'Sleep Disorder':'count'}).plot.pie(autopct ='%1.1f%%',
                                                                                                                  subplots=True,figsize=(20,10),
                                                                                                                  colors=['#C39BD3','#D2B4DE','#EBDEF0','#F4ECF7'])

plt.axis('equal')
plt.show()




sleep_data.isna().sum()



sleep_data.columns



sleep_data['Blood Pressure'].unique()




sleep_data['Blood Pressure']=sleep_data['Blood Pressure'].apply(lambda x:0 if x in ['120/80','126/83','125/80','128/84','129/84','117/76','118/76','115/75','125/82','122/80'] else 1)
# 0 = normal blood pressure
# 1 = abnormal blood pressure
# 전처리 작업


sleep_data["Age"]=pd.cut(sleep_data["Age"],2) # 2개 구간
sleep_data["Heart Rate"]=pd.cut(sleep_data["Heart Rate"],4) #4개 구간
sleep_data["Daily Steps"]=pd.cut(sleep_data["Daily Steps"],4) #4개 구간 
sleep_data["Sleep Duration"]=pd.cut(sleep_data["Sleep Duration"],3) #3개 구간
sleep_data["Physical Activity Level"]=pd.cut(sleep_data["Physical Activity Level"],4) #4개 구간



from sklearn.preprocessing import LabelEncoder #for converting non-numeric data (String or Boolean) into numbers
LE=LabelEncoder()

categories=['Gender','Age','Occupation','Sleep Duration','Physical Activity Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder']
for label in categories:
    sleep_data[label]=LE.fit_transform(sleep_data[label])



sleep_data.drop(['Person ID'], axis=1, inplace=True)



correlation=sleep_data.corr() # 숫자형 컬럼들끼리의 상관관계 행렬 계산 범위는 -1 ~ 1
max_6_corr=correlation.nlargest(6,"Sleep Disorder") # Sleep Disorder와 상관계수가 가장 높은 6개 컬럼 추출
sns.heatmap(max_6_corr,annot=True,fmt=".2F",annot_kws={"size":8},linewidths=0.5,cmap='BuPu') #seaborn
plt.title('Maximum six features affect Sleep Disorder')
plt.show()



# 머신러닝을 위한 데이터 분할과 시작적 출력 포맷팅
x=sleep_data.iloc[:,:-1] #전체 행에서 마지막 열 빼고 다 가져오기 -> 독립변수
y=sleep_data.iloc[:,-1] #전체 행에서 마지막 열만 가져오기 -> 타겟 값 

x_shape=colored(x.shape, "magenta",None, attrs=["blink"])
y_shape=colored(y.shape, "magenta",None, attrs=["blink"])
print('The dimensions of x is : ',x_shape)
print('The dimensions of y is : ',y_shape)


# 학습용/테스트용 데이터 분할 작업
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=32,shuffle=True)

'''
| 변수명       | 의미       | 예시 shape (데이터 300개일 때) 
| --------- | -------- | ---------------------- |
| `x_train` | 학습용 입력값  | `(201, n)`             
| `x_test`  | 테스트용 입력값 | `(99, n)`              
| `y_train` | 학습용 정답   | `(201,)`               
| `y_test`  | 테스트용 정답  | `(99,)`                

'''
x_train_shape=colored(x_train.shape, "magenta",None, attrs=["blink"])
x_test_shape=colored(x_test.shape, "magenta",None, attrs=["blink"])
y_train_shape=colored(y_train.shape, "magenta",None, attrs=["blink"])
y_test_shape=colored(y_test.shape, "magenta",None, attrs=["blink"])

print("x train dimensions :",x_train_shape)
print("x test dimensions: ",x_test_shape)
print("y train dimensions :",y_train_shape)
print("y test dimensions :",y_test_shape)



#LogisticRegression() 로지스틱 회귀 모델 생성 / 분류 문제에 사용되는 알고리즘
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression().fit(x_train,y_train) #.fit = 분리한 학습 데이터로 모델 학습


LR_training_score=colored(round(LR.score(x_train,y_train)*100,2), "magenta",None, attrs=["blink"])
LR_testing_score=colored(round(LR.score(x_test,y_test)*100,2), "magenta",None, attrs=["blink"])

print(f"LR training score :",LR_training_score)
print("LR testing score :",LR_testing_score)


'''
# 정규화
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from termcolor import colored

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.33, random_state=32, shuffle=True)
LR=LogisticRegression().fit(x_train,y_train) #.fit = 분리한 학습 데이터로 모델 학습

LR_training_score=colored(round(LR.score(x_train,y_train)*100,2), "magenta",None, attrs=["blink"])
LR_testing_score=colored(round(LR.score(x_test,y_test)*100,2), "magenta",None, attrs=["blink"])

print(f"LR training score :",LR_training_score)
print("LR testing score :",LR_testing_score)
'''

LR_y_pred=LR.predict(x_test) # 예측 결과 저장
print("예측값:", LR_y_pred[:10])
print("실제값:", y_test[:10].values)



# 선형회귀 vs xgboost 선형회귀는 단순 빠르지만 정규화가 꼭 필요 + 복잡한 패턴은 못 잡음 , xgboost는 정규화 필요 x, 복잡한 관계를 잘 잡지만 해석이 어렵고 시간이 더 걸릴 수 있음
# 약한 모델을 여러개 연결해서 강한 모델을 만드는 Boosting 기법을 사용하는 머신러닝 모델 트리 기반
from xgboost import XGBClassifier
xgb=XGBClassifier().fit(x_train,y_train)




xgb_training_score=colored(round(xgb.score(x_train,y_train)*100,2), "magenta",None, attrs=["blink"])
xgb_testing_score=colored(round(xgb.score(x_test,y_test)*100,2), "magenta",None, attrs=["blink"])

print("xgb training score :",xgb_training_score)
print("xgb testing score :",xgb_testing_score)




xgb_y_pred=xgb.predict(x_test)

print("예측값:", xgb_y_pred[:10])
print("실제값:", y_test[:10].values)




# Categorical + Boosting -> CatBoost 머신러닝 모델 범주형 데이터 처리 트리 기반 xgboost와 성능 및 구조는 비슷, 사용자 편의성, 범주형 처리 방식, 과적합 대응 방식에 차이가 있음
from catboost import CatBoostClassifier
CBC=CatBoostClassifier(verbose=False).fit(x_train,y_train)




CBC_training_score=colored(round(CBC.score(x_train,y_train)*100,2), "magenta",None, attrs=["blink"])
CBC_testing_score=colored(round(CBC.score(x_test,y_test)*100,2), "magenta",None, attrs=["blink"])

print("CBC training score :",CBC_training_score)
print("CBC testing score :",CBC_testing_score)




CBC_y_pred=CBC.predict(x_test)


print("예측값:", CBC_y_pred[:10])
print("실제값:", y_test[:10].values)






# 사이킷 런에서 제공하는 기본적인 그래디언트 Boosting 분류 모델. 작은 데이터에 적합
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier().fit(x_train,y_train)




GBC_training_score=colored(round(GBC.score(x_train,y_train)*100,2), "magenta",None, attrs=["blink"])
GBC_testing_score=colored(round(GBC.score(x_test,y_test)*100,2), "magenta",None, attrs=["blink"])

print("GBC training score :",GBC_training_score)
print("GBC testing score :",GBC_testing_score)



GBC_y_pred=GBC.predict(x_test)


'''
| 항목                | **SVC**                | **XGBoost / CatBoost / GBC**     |
| ----------------- | ---------------------- | -------------------------------- |
| **모델 타입**         | 커널 기반 분류기              | 트리 기반 부스팅 모델                     |
| **비선형 처리**        | 가능 (커널 사용 시)          |  가능 (트리 자체가 비선형)                |
| **입력 정규화 필요**     | 필요함 (스케일에 민감함)       |  필요 없음                          |
| **대용량 데이터에 강한가?** |  느리고 메모리 많이 사용        | 최적화 잘 되어 있음                    |
| **과적합 방지**        | 하이퍼파라미터로 조절 (C, gamma) | 트리 수, 정규화 등으로 조절                 |
| **설명력 (해석)**      | 보통 어려움                 | 트리 모델은 feature importance로 설명 가능 |

'''

# Support Vector Classifier 선형/비선형 분류 모델
# 정규화 작업 필수
from sklearn.svm import SVC
svc = SVC().fit(x_train,y_train)



svc_training_score=colored(round(svc.score(x_train,y_train)*100,2), "magenta",None, attrs=["blink"])
svc_testing_score=colored(round(svc.score(x_test,y_test)*100,2), "magenta",None, attrs=["blink"])

print("svc training score :",svc_training_score)
print("svc testing score :",svc_testing_score)



svc_y_pred=svc.predict(x_test)


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

svc = SVC().fit(x_train_scaled,y_train)



svc_training_score=colored(round(svc.score(x_train_scaled, y_train)*100,2), "magenta",None, attrs=["blink"])
svc_testing_score=colored(round(svc.score(x_test_scaled, y_test)*100,2), "magenta",None, attrs=["blink"])

print("svc training score :",svc_training_score)
print("svc testing score :",svc_testing_score)





#분류 모델 성능 평가 기본 도구
from sklearn.metrics import confusion_matrix




models_predictions=[LR_y_pred,xgb_y_pred,CBC_y_pred,GBC_y_pred,svc_y_pred]
model={1:'LR_y_pred',2:'xgb_y_pred',3:'CBC_y_pred',4:'GBC_y_pred',5:'svc_y_pred'}


plt.figure(figsize=(15,7))
for i,y_pred in enumerate(models_predictions,1) :
    
    cm = confusion_matrix(y_test,y_pred)
    #confusion_matrix? 모델의 예측값과 실제값을 비교, 각 클래스별로 정답과 오답이 몇 개인지 행렬로 보여주는 함수
    
    plt.subplot(2,3,i)
    sns.heatmap(cm,cmap='BuPu',linewidth=3,fmt='',annot=True,
                xticklabels=['(None)','(Sleep_Apnea)','(Insomnia)'],
                yticklabels=['(None)','(Sleep_Apnea)','(Insomnia)'])
    
    
    plt.title(' CM of  '+ model[i])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)



    

    import shap

shap_values = shap.TreeExplainer(xgb).shap_values(x_test)
#shap.summary_plot(shap_values, x_test,class_names=['None','Sleep_Apnea','Insomnia'])
shap.summary_plot(shap_values, x_test,class_names=['None','Sleep_Apnea','Insomnia'], plot_type="bar")
# SHAP은 Matpkoitlib과 JS 시각화 기능을 동시에 사용하는 특수 도구 이기 때문에 VScode에서는 제대로 표시되지 않을 수가 있다.
# bar plot은 JS 없이 matplotlib으로만 그려서 VS코드에서도 잘 보인다.
