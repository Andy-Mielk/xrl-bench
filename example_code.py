from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment

env = Environment(environment_name="lunarLander")
dataset = env.get_dataset()
actions = dataset['action']
states = dataset.drop(['action', 'reward'], axis=1)
# 下一行会报错：TypeError: The passed model is not callable and cannot be analyzed directly with the given masker! Model: LGBMClassifier()
# 解决方法：在xrlbench/custom_explainers/tabular_shap.py中的_fit_model函数中添加一行代码：model.fit(self.X_enc, self.y) 
explainer = Explainer(method="sarfa", state=states, action=actions)
# explainer = Explainer(method="sarfa", state=states, action=actions)    # 会报错: TypeError: __init__() missing 1 required positional argument: 'model'
shap_values = explainer.explain()
evalutor = Evaluator(metric="AIM", environment=env)
aim = evalutor.evaluate(states, actions, shap_values, k=1)
print(aim)