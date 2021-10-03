# coding=utf-8
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer

# 데이터 불러오기
df = pd.read_excel("data/kakao_final.xlsx")

# 데이터 전처리
df_drop = df.drop(["No", "Q1_1", "Q1_2", "Q1_3", "Q1_4", "Q1_5", "Q2", "성별", "나이"], axis=1)
df_drop_colnames = df_drop.columns.tolist()  # 칼럼명 리스트 생성

df_drop = df_drop.apply(pd.to_numeric, errors="coerce").fillna(0)  # 데이터 타입 숫자로 통일


# 탐색적 요인분석

# [1] x 값 DF 생성
df_fa = df_drop.drop(["Q7_1", "Q7_2", "Q7_3", "Q7_4", "Q7_5", "Q7_6", "Q7_7", "Q7_8", "Q7_9", "Q8_1", "Q8_2", "Q8_3", "Q8_4", "Q8_5", "Q9_1"], axis=1)
df_fa_colnames = df_fa.columns.tolist()

# [2] 요인 수 선택
fa = FactorAnalyzer(n_factors=15, rotation=None)
fa.fit(df_fa)
ev, v = fa.get_eigenvalues()

plt.scatter(range(1, df_fa.shape[1]+1), ev)
plt.plot(range(1, df_fa.shape[1]+1), ev)
plt.title("Scree Plot", size=15)
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()  # Eigen 값이 1 이상일 때까지 or 그래프 기울기가 완만해지기 전까지 나누어야 데이터 결함 최소화

# [3] 요인분석 실시
fa_final = FactorAnalyzer(n_factors=3, rotation="varimax")  # ml: 최대우도 방법
fa_final.fit(df_fa)
efa_result = pd.DataFrame(fa_final.loadings_, index=df_fa_colnames)

plt.figure(figsize=(5, 8))
sns.heatmap(efa_result, cmap="Blues", annot=True, fmt=".2f")
plt.title("Factor Analyze", size=15)
plt.show()

"""
[요인분석 결과 index]
A: 0, 1, 2, 3, 4, 5, 6, 7, 8
B: 9, 10, 11, 12
C: 13, 14
** 0.5 이하 변수: 9, 12
"""


# FA 기준 새로운 데이터프레임 생성
df_drop["A"] = np.mean(df_fa[df_fa_colnames[0:9]].T)
df_drop["B"] = np.mean(df_fa[df_fa_colnames[9:13]].T)
df_drop["C"] = np.mean(df_fa[df_fa_colnames[13:15]].T)

df_drop["reciprocity"] = np.mean(df_drop[df_drop_colnames[15:24]].T)  # 호혜성
df_drop["relationship"] = np.mean(df_drop[df_drop_colnames[24:29]].T)  # 인간관계
df_drop.rename(columns={df_drop_colnames[29]: "intention"}, inplace=True)  # 사용의도 (기존 칼럼에서 칼럼명만 변경)

df_drop_colnames_new = df_drop.columns.tolist()

df_clean = df_drop[df_drop_colnames_new[-6:]]  # 범주별 평균 데이터프레임 생성
df_clean_colnames = df_clean.columns.tolist()


# Multiple Linear Regression

def fit_xy_model(y_name):
    xy_model = ols(f"{y_name} ~ A + B + C", df_clean)
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