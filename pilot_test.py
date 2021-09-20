# coding=utf-8
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# 데이터 불러오기
df = pd.read_csv("data/kakao_pilot_test.csv")

# 데이터 전처리
df_drop = df.drop([df.columns[0], df.columns[-1]], axis=1)  # 타임스탬프, 전화번호 칼럼 삭제
df_drop = df_drop.dropna(axis=0)  # 결측값이 있는 행 전체 삭제

df_drop_colnames = df_drop.columns.tolist()  # 칼럼명 리스트 생성

change_value_dict = {
    "항상 사용함": 5,
    "자주 사용함": 4,
    "가끔 사용함": 3,
    "사용하지 않음": 2,
    "알지 못함": 1,
}
for colname in df_drop_colnames[8:15]:  # 범주형 변수 숫자화
    df_drop = df_drop.replace({f"{colname}": change_value_dict})

df_drop = df_drop.apply(pd.to_numeric, errors="coerce").fillna(0)  # 데이터 타입 숫자로 통

# 새로운 데이터프레임 생성
"""
칼럼 범주화 인덱스
- 접근성: 0, 1
- 네트워크성: 2, 3 
- 편의성: 4, 5, 6, 7 
- 가시성: 8, 9, 10, 11, 12
- 표현성: 13
- 기록성: 14

- 호혜성: 15, 16, 17, 18, 19, 20, 21, 22, 23
- 인간관계: 24, 25, 26, 27, 28, 29
- 사용의도: 30
"""

df_drop["accessibility"] = np.mean(df_drop[df_drop_colnames[0:2]].T)  # 접근성
df_drop["network"] = np.mean(df_drop[df_drop_colnames[2:4]].T)  # 네트워크성
df_drop["convenience"] = np.mean(df_drop[df_drop_colnames[4:8]].T)  # 편의성
df_drop["visibility"] = np.mean(df_drop[df_drop_colnames[8:13]].T)  # 가시성

df_drop["reciprocity"] = np.mean(df_drop[df_drop_colnames[15:24]].T)  # 호혜성
df_drop["relationship"] = np.mean(df_drop[df_drop_colnames[24:30]].T)  # 인간관계
df_drop.rename(columns={df_drop_colnames[30]: "intention"}, inplace=True)  # 사용의도 (기존 칼럼에서 칼럼명만 변경)

df_drop_colnames_new = df_drop.columns.tolist()

df_clean = df_drop[df_drop_colnames_new[-7:]]  # 범주별 평균 데이터프레임 생성
df_clean_colnames = df_clean.columns.tolist()


# Multiple Linear Regression

def fit_xy_model(y_name):
    xy_model = ols(f"{y_name} ~ accessibility + network + convenience + visibility", df_clean)
    xy_res = xy_model.fit()
    y_name_ko = "호혜성" if y_name == "reciprocity" else "인간관계"
    print(f"< X 독립변수와 {y_name_ko} >")
    print(xy_res.summary())
    return xy_model


reciprocity_model = fit_xy_model("reciprocity")
relationship_model = fit_xy_model("relationship")

yz_model = ols("intention ~ reciprocity + relationship", df_clean)
yz_res = yz_model.fit()
print("< Y 독립변수와 사용의도 >")
print(yz_res.summary())


# VIF 계산

def get_vif(model):
    pd_vif = pd.DataFrame({"column": name, "VIF": variance_inflation_factor(model.exog, idx)}
                         for idx, name in enumerate(model.exog_names)
                         if name != "Intercept")  # 절편의 VIF 생략
    return pd_vif


xy_vif = get_vif(reciprocity_model)
yz_vif = get_vif(yz_model)

print("< X 변수의 다중공선성 >")
print(xy_vif)
print("\n")
print("< Y 변수의 다중공선성 >")
print(yz_vif)
