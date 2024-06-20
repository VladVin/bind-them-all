import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from xgboost import XGBRegressor
import pickle
import os
import joblib
from pathlib import Path

def plot_hyperopt_results(best, trials, pm, model_name, name):
    parameters = best.keys()
    num_params = len(parameters)
    f, axes = plt.subplots(ncols=num_params, figsize=(15, 5))
    cmap = plt.cm.jet
    num_trials = len(trials.trials)

    for i, param in enumerate(parameters):
        xs = np.array([t['misc']['vals'][param] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]

        for j in range(num_trials):
            color = cmap(float(j) / num_trials)
            axes[int(i)].scatter(xs[j], ys[j], s=20, linewidth=0.01, alpha=0.5, color=color)

        axes[int(i)].set_title(param)

    plt.savefig( pm['fig_dir'] + '/' + model_name + '_%s_hyp_search.png' % name)

def log_results(name, best, cv_score, test_score, accuracy, pm):
    with open('Output_txts/'+ pm['Project_name'] + pm['output_filename'], 'a') as fileOutput:
        fileOutput.write('----MODEL: ' + name + '----\n')
        fileOutput.write(name + ' best params: ' + str(best) + '\n')
        fileOutput.write('5-fold cross validation: ' + str(cv_score) + '\n')
        fileOutput.write('Test Score: ' + str(test_score) + 'Accuracy Score: ' + str(accuracy) + '\n')


def log_best_params(name, best, pm):
    with open('Output_txts/'+ pm['Project_name'] + pm['output_filename'], 'a') as fileOutput:
        fileOutput.write(name + ' best params: ' + str(best) + '\n')


def save_model_to_dir(model, pm, model_name, name):
    model_path = pm['dir'] + '/' + pm['model_dir'] + '/' + pm['Project_name'] + pm['model_dir'] + '/' + model_name + '_' + name + '_model.h5'
    if not os.path.isfile(model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)


def save_model_xgb(model, pm, model_name, name):
    model_path = pm['model_dir'] + '/' + model_name + '_' + name + '_model.json'
    if not os.path.isfile(model_path):
        model.save_model(model_path)

def Do_XGradientBoost_regression(X, y, testSet, y_test, name, pm, scored, seed):
    m = 'XGBr'
    scaler = MinMaxScaler()

    # Fit and transform X and transform testSet
    X = scaler.fit_transform(X)
    testSet_norm = scaler.transform(testSet)
    # Save normalizing parameters
    scaler_path =  pm['model_dir'] + '/' + name + '_scaler.pkl'
    scaler_path = Path(scaler_path)
    joblib.dump(scaler, scaler_path)

    print(
        "|---------------------------------------------------\nStarting Hyp Search for XGBoost Regressor\n|---------------------------------------------------")

    if pm['hyp_tune'] == 1:
        X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=seed)

        def hyperopt_train_test(params):
            modelxgb = XGBRegressor(**params, random_state=seed)
            return -cross_val_score(modelxgb, X_tune, y_tune, scoring='neg_mean_absolute_error', cv=5, n_jobs=-2).mean()

        space4xgb = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 100, 1)),
                     'learning_rate': hp.choice('learning_rate', [0.015, 0.03, 0.06, 0.12, 0.25]),
                     'max_depth': scope.int(hp.quniform('max_depth', 10, 20, 1)),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 20, 1)),
                     'gamma': scope.int(hp.quniform('gamma', 0, 9, 1)),
                     'reg_alpha': scope.int(hp.quniform('reg_alpha', 20, 180, 1))}

        best_score = 10
        best_params = None

        def f(params):
            nonlocal best_score
            nonlocal best_params
            acc = hyperopt_train_test(params)
            print('acc:', acc)
            if acc < best_score:
                best_score = acc
                best_params = params
                print('new best:', best_score, params)
            return {'loss': acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space4xgb, algo=tpe.suggest, max_evals=pm['maxevals'], trials=trials)

        # Print the best parameters
        print("Best parameters:", best_params)

        log_best_params(name, best_params, pm)

        plot_hyperopt_results(best_params, trials, pm, 'XGB', name)

    model = XGBRegressor(**best_params, random_state=seed)
    scored['cv'] = str(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-2))
    model.fit(X, y)

    print('5-fold cross validation: ', scored['cv'])
    y_pred = model.predict(testSet_norm)

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    accuracy = (mean_squared_error(y_test, y_pred)) ** 0.5
    scored['score'] = accuracy
    print('Test rmse: ', accuracy)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scored[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

    print(f'Model Evaluation Metrics for {name}:')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R²: {r2}')

    save_model_xgb(model, pm, 'XGB', name)
    # log_results(name, best_params, scored['cv'], scored['score'], accuracy, pm)

    score_xgb = {**scored}
    score_xgb = pd.DataFrame.from_dict(score_xgb, orient='index', columns=[name])
    return score_xgb
