import importlib
import PPO.environment as environment
from common.fileManager import Config, File
importlib.reload(environment)
import numpy as np
np.set_printoptions(precision=6, suppress=True)
action_list = [-0.8825346,0.44341397,-1.0720923,-1.2481807,-1.5970774,0.12113419,0.82465374,0.20973554,-0.6991998,-0.003223047,-0.63272613,-0.988931,-0.55648994,0.010717839,-0.60529315,-0.7844695,0.34349868,-0.16062322,-1.4823828,-0.57580817,-0.37807506,-0.61315584,-0.23614007,-0.35307956,-0.27526766,-0.06650478,-1.3188577,0.16075681,-0.6535168,0.91349775,-0.16915025,-0.51264954,0.24208795,0.016758457,-0.37561572,0.005944073,0.88521576,-0.36042,0.2325444,-0.5211466,-0.39855617,-0.18360297,0.021541312,-0.27526137,-0.48348498,-0.7221449,-0.45249024,-0.75809973,-1.254751,0.11704171,-0.46443993,-0.30440778,0.08699142,-0.05408309,0.51617324,0.72540313,0.29040885,-0.5783853,0.20174468,-0.70291793,-0.57283545,-0.123116195,0.6216713,-0.21963935,-0.098082095,-0.009893075,-1.4782554,0.39699358,0.77109694,0.3628115,0.1997031,-1.4616209,-1.8414756,0.12029716,-1.933087,-0.079728216,0.39432243,-1.0416086]

stock_config = Config.load_config("config/StockConfig.yaml")
env = environment.StockEnvironment(stock_config)

for i in range(1000):
    print(i)
    state, info = env.reset()
    #print(state["num"])
    #print(info)
    #print("=====================")
    for action in action_list:
        next_state, reward, done, truncated, info = env.step(action)
        
        #print(next_state["num"])
        #print(info)
        #print("=====================")
        state = next_state

        if done or truncated:
            break
    
    