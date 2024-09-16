import os
import sys
import numpy as np
from urllib.parse import urlparse
import mlflow
import dagshub
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from dataclasses import dataclass
from src.mlproject.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifact','model.pkl')
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
    
  def eval_metrics(self,actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    
    return rmse,mae,r2
    
  def initiate_model_trainer(self,train_array,test_array):
    try:
      logging.info("split training and test data")
      x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
      
      models = {
        "Linear Regression":LinearRegression(),
        "AdaBoost":AdaBoostRegressor(),
        "Random Forest":RandomForestRegressor(),
        "GradientBoost":GradientBoostingRegressor(),
        "Decision tree":DecisionTreeRegressor()
        }
      
      params = {"Linear Regression":{},
          "AdaBoost":{
          'learning_rate':[.1,.01,.05,.001],
          #'loss':['linear','square','exponential'],
          'n_estimators':[8,16,32,64,128,256]
        },
        "Random Forest":{
          #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
          #'max_features':['sqrt','log2',None],
          'n_estimators':[8,16,32,64,128,256]
        },
        "GradientBoost":{
          #'loss':['squared_error','huber','absolute_error','quantile'],
          'learning_rate':[.1,.01,.05,.001],
          'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
          #'criterion':['squared_error','friedman_mse'],
          #'max_features':['auto','sqrt','log2'],
          'n_estimators':[8,16,32,64,128,256]
          
        },
        "Decision tree":{
          'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
          #'splitter':['best','random'],
          #'max_features':['sqrt','log2']
        }        
      }
      
      model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)
      
      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      
      best_model = models[best_model_name]
      
      print("This is the best model:")
      print(best_model_name)
      
      model_names = list(params.keys())
      actual_name = ""
      
      for model in model_names:
        if model == best_model_name:
          actual_name += model
      
      best_params = params[actual_name]
      
      mlflow.set_registry_uri("https://dagshub.com/abhisek-13/MLops.mlflow")
      tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme
      
      
      dagshub.init(repo_owner='abhisek-13', repo_name='MLops', mlflow=True)
      
      # mlflow
      with mlflow.start_run():
        predicted_qualities = best_model.predict(x_test)
        
        rmse,mae,r2 = self.eval_metrics(y_test,predicted_qualities)
        
        mlflow.log_param("Best Parameter",best_params)
        
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)
      
        if tracking_url_type_score != "file":
          mlflow.sklearn.log_model(best_model,"model",registered_model_name = actual_name)
        else:
          mlflow.sklearn.log_model(best_model,"model")
      
      
      
      if best_model_score < 0.6:
        raise CustomException("No best model found")
      logging.info(f'Best found model on both training and testing dataset')
      
      save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
      
      predicted = best_model.predict(x_test)
      
      r2 = r2_score(y_test,predicted)
      return r2
      
    except Exception as e:
      raise CustomException(e,sys)
    