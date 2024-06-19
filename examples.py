# -*- coding: utf-8 -*-

from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment


# Tabular数据测试
def tabular_input_test(environment, method, metric, k=3):
    environment = Environment(environment_name=environment)
    df = environment.get_dataset(generate=False)
    df_sample = df.sample(n=5000, random_state=42)
    action_sample = df_sample['action']
    state_sample = df_sample.drop(['action', 'reward'], axis=1)
    if method == "tabularShap":
        explainer = Explainer(method=method, state=state_sample, action=action_sample)
    else:
        explainer = Explainer(method=method, state=state_sample, action=action_sample, model=environment.model)
    importance = explainer.explain()
    evaluator = Evaluator(metric=metric, environment=environment)
    if metric == "RIS":
        performance = evaluator.evaluate(state_sample, action_sample, importance, explainer=explainer)
    else:
        performance = evaluator.evaluate(state_sample, action_sample, importance, k=k)
    return performance


# Image数据测试
def image_input_test(environment, method, metric, k=50):
    environment = Environment(environment_name=environment)
    dataset = environment.get_dataset(generate=False, data_format="h5")
    explainer = Explainer(method=method, state=dataset.observations, action=dataset.actions,
                          model=environment.model)
    importance = explainer.explain()
    # for breakOut env, dataset's state: (S-A pair size=3776, 3, 84, 84), action: (S-A pair size=3776, 4), importance: (3776, 3, 84, 84, 4)
    # for pong env, dataset's state: (S-A pair size=4000, 3, 84, 84), action: (S-A pair size=4000, 6), importance: (4000, 3, 84, 84, 6)
    # print(importance)

    # Sample observations from the dataset
    sample_observations = dataset.observations[:5]  # (5, 3, 84, 84) 0-255

    # Get the corresponding importance for the sampled observations
    sample_importance = importance[:5]  # (5, 3, 84, 84, 6) -5～5?

    # Print the sampled observations and their importance
    for obs, imp in zip(sample_observations, sample_importance):
        print("Observation:")
        print(obs)
        print("Importance:")
        print(imp)
        print("--------------------")
    
    evaluator = Evaluator(metric=metric, environment=environment)
    if metric == "RIS":
        performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, explainer=explainer)
    else:
        performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, k=k)
    return performance


if __name__ == "__main__":
    # # Tabular数据测试
    # # it worked, 加载的是预训练好的数据和模型
    # environment = "lunarLander"
    # method = "integratedGradient"   # 'tabularShap', 'sarfa', 'perturbationSaliency', 'tabularLime', 'deepShap', 'gradientShap', 'integratedGradient', 
    # metric = "AIM"
    # performance = tabular_input_test(environment, method, metric, k=3)
    # print(performance)

    
    # Image数据测试
    # it worked
    environment = "pong"    # 'breakOut', 'pong'
    method = "imagePerturbationSaliency"    # 'imagePerturbationSaliency', 'imageSarfa', 'imageDeepShap', 'imageGradientShap', 'imageIntegratedGradient'
    metric = "imageAIM"
    performance = image_input_test(environment, method, metric, k=50)













