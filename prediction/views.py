from json import loads

from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import render

import pickle
import pandas as pd
import sklearn
from django.views.decorators.csrf import csrf_exempt

from category_encoders import OrdinalEncoder

DEPARTMENT = {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2, 'Other': 3}
EDUCATION = {
    'Life Sciences': 0,
    'Other': 1,
    'Medical': 2,
    'Marketing': 3,
    'Technical Degree': 4,
    'Human Resources': 5,
}
GENDER = {'Female': 0, 'Male': 1}
MARITAL = {'Single': 0, 'Married': 1, 'Divorced': 2}
ROLE = {
    'Sales Executive': 0,
    'Research Scientist': 1,
    'Laboratory Technician': 2,
    'Manufacturing Director': 3,
    'Healthcare Representative': 4,
    'Manager': 5,
    'Sales Representative': 6,
    'Research Director': 7,
    'Human Resources': 8,
}

MAP_BUSINESS = [
    {'col': 'BusinessTravel', 'mapping': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}}
]
MAP_OVERTIME = [{'col': 'OverTime', 'mapping': {'No': 0, 'Yes': 1}}]
MAP_EDUCATION = [
    {
        'col': 'Education',
        'mapping': {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5},
    }
]


def fill_dummies(x, enum, key):
    res = [0 for _ in range(len(enum))]
    if x[key] in enum:
        res[enum[x[key]]] = 1
    return pd.Series(res)


def fit_oe(data, maplist):
    oe = OrdinalEncoder(mapping=maplist)
    return oe.fit_transform(data)


def set_data(data):
    df = pd.DataFrame([data])
    # Доп параметры на основе полученных данных
    df['MonthlyIncome/Age'] = df['MonthlyIncome'] / df['Age']
    df["Age_risk"] = (df["Age"] < 34).astype(int)
    df["Distance_risk"] = (df["DistanceFromHome"] >= 20).astype(int)
    df["YearsAtCo_risk"] = (df["YearsAtCompany"] < 4).astype(int)
    df['NumCompaniesWorked'] = df['NumCompaniesWorked'].replace(0, 1)
    df['AverageTenure'] = df["TotalWorkingYears"] / df["NumCompaniesWorked"]
    df['JobHopper'] = ((df["NumCompaniesWorked"] > 2) & (df["AverageTenure"] < 2.0)).astype(int)
    df["AttritionRisk"] = df["Age_risk"] + df["Distance_risk"] + df["YearsAtCo_risk"] + df['JobHopper']
    # Dummies
    df[['Dep_sales', 'Dep_research_dev', 'Dep_hum_res', 'Dep_other']] = df.apply(
        lambda x: fill_dummies(x, DEPARTMENT, 'Department'), axis=1
    )
    df[['Educ_lifesc', 'Educ_other', 'Educ_medic', 'Educ_market', 'Educ_tech', 'Educ_hum_res']] = df.apply(
        lambda x: fill_dummies(x, EDUCATION, 'EducationField'), axis=1
    )
    df[['Gender_fem', 'Gender_male']] = df.apply(lambda x: fill_dummies(x, GENDER, 'Gender'), axis=1)
    df[['MaritalStatus_sing', 'MaritalStatus_marr', 'MaritalStatus_divorce']] = df.apply(
        lambda x: fill_dummies(x, MARITAL, 'MaritalStatus'), axis=1
    )
    df[
        [
            'JobRole_sales_exef',
            'JobRole_res_sc',
            'JobRole_lab_tech',
            'JobRole_man_dir',
            'JobRole_health',
            'JobRole_manager',
            'JobRole_sales_rep',
            'JobRole_res_dir',
            'JobRole_hum_resb',
        ]
    ] = df.apply(lambda x: fill_dummies(x, ROLE, 'JobRole'), axis=1)

    df.drop(['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'], axis=1, inplace=True)

    # Mapping
    df['BusinessTravel'] = fit_oe(df['BusinessTravel'], MAP_BUSINESS)
    df['OverTime'] = fit_oe(df['OverTime'], MAP_OVERTIME)
    df['Education'] = fit_oe(df['Education'], MAP_EDUCATION)
    return df[
        [
            'Age',
            'BusinessTravel',
            'DistanceFromHome',
            'Education',
            'EnvironmentSatisfaction',
            'JobInvolvement',
            'JobSatisfaction',
            'MonthlyIncome',
            'NumCompaniesWorked',
            'OverTime',
            'RelationshipSatisfaction',
            'StockOptionLevel',
            'TotalWorkingYears',
            'TrainingTimesLastYear',
            'WorkLifeBalance',
            'YearsAtCompany',
            'YearsSinceLastPromotion',
            'MonthlyIncome/Age',
            'Age_risk',
            'Distance_risk',
            'YearsAtCo_risk',
            'AverageTenure',
            'JobHopper',
            'AttritionRisk',
            'JobRole_sales_exef',
            'JobRole_res_sc',
            'JobRole_lab_tech',
            'JobRole_man_dir',
            'JobRole_health',
            'JobRole_manager',
            'JobRole_sales_rep',
            'JobRole_res_dir',
            'JobRole_hum_resb',
            'MaritalStatus_sing',
            'MaritalStatus_marr',
            'MaritalStatus_divorce',
            'Gender_fem',
            'Gender_male',
            'Educ_lifesc',
            'Educ_other',
            'Educ_medic',
            'Educ_market',
            'Educ_tech',
            'Educ_hum_res',
            'Dep_sales',
            'Dep_research_dev',
            'Dep_hum_res',
            'Dep_other',
        ]
    ]


def get_prediction(data):
    df = set_data(data)
    with open('prediction/model_gbc.pkl', 'rb') as f:
        model = pickle.load(f)
        prediction = model.predict(df)
    return int(prediction[0])


def validate_data(data):
    for int_key in [
        "Age",
        "DistanceFromHome",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
    ]:
        try:
            data[int_key] = int(data[int_key])
        except ValueError:
            raise ValidationError('Wrong int')


@csrf_exempt
def post_result(request):
    data = loads(request.body)
    validate_data(data)
    res = get_prediction(data)
    res = get_prediction(data)
    return JsonResponse({'prediction': res})


def prediction_html(request):
    template_name = 'prediction.html'
    return render(request, template_name)


def about_html(request):
    template_name = 'about.html'
    return render(request, template_name)
