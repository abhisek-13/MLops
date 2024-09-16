import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error
from dataclasses import dataclass
from src.mlproject.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifact','model.pkl')
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
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
      
      if best_model_score < 0.6:
        raise CustomException("No best model found")
      logging.info(f'Best found model on both training and testing dataset')
      
      save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
      
      predicted = best_model.predict(x_test)
      
      r2 = r2_score(y_test,predicted)
      return r2
      
    except Exception as e:
      raise CustomException(e,sys)
    