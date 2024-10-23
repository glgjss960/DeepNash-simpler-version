import pickle 
from absl.testing 
import absltest 
import jax 
from jax import numpy as jnp 
import numpy as np 
from open_spiel.python.algorithms.rnad import rnad 
#from mpl_toolkits import mplot3d 
import tkinter 
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d 
#matplotlib.use('TkAgg') 
from open_spiel.python import policy as policy_lib 
from open_spiel.python.algorithms import exploitability as exploitab_lib 
from open_spiel.python.algorithms import get_all_states 
import pyspiel 

class RNADTest(absltest.TestCase): 
    def test_run_kuhn(self): 
        #log=open("NashConv.txt",mode="a",encoding="utf-8") 
        log_1=open("pi_target.txt",mode="a",encoding="utf-8") 
        log_2=open("logits_target.txt",mode="a",encoding="utf-8") 
        log_3=open("logpi_target.txt",mode="a",encoding="utf-8") 
        log_4=open("v.txt",mode="a",encoding="utf-8") 
        np.set_printoptions(threshold=np.inf) 
        solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="kuhn_poker")) 
        for _ in range(4000): 
            solver.step() 
        env_step_list = solver.test_params(solver.vision_params_target) 
        nash_conv = [] 
        game = pyspiel.load_game("kuhn_poker") 
        #states=policy_lib.get_tabular_policy_states(game) 
        states=get_all_states.get_all_states(game) 
        tabular_policy = policy_lib.TabularPolicy(game) 
        for params in solver.vision_params_target: 
            for state_index, state in enumerate(tabular_policy.states): 
                action_probabilities,_,_,_ = solver.network.apply( params,solver._state_as_env_step(state) )
                tabular_policy.action_probability_array[state_index, :] = action_probabilities
            nash_conv.append(exploitab_lib.nash_conv(game, tabular_policy)) 
        
        for _ in range(len(env_step_list)): 
            print("state",end='',file=log_1) 
            print(_,end='',file=log_1) 
            print(":",end='',file=log_1) 
            print(env_step_list[_].obs,file=log_1) 
            aux_x = np.array(solver.vision_policy_1[_]) 
            aux_x = list(aux_x.flatten()) 
            
            print("policy_p:",file=log_1) 
            print(aux_x,file=log_1) 
            print("state",end='',file=log_2) 
            print(_,end='',file=log_2) 
            print(":",end='',file=log_2) 
            print(env_step_list[_].obs,file=log_2) 
            
            aux_x = np.array(solver.vision_logit_1[_]) 
            aux_x = list(aux_x.flatten()) 
            print("logits_p:",file=log_2) 
            print(aux_x,file=log_2) 
            print("state",end='',file=log_3) 
            print(_,end='',file=log_3) 
            print(":",end='',file=log_3) 
            print(env_step_list[_].obs,file=log_3) 
            aux_x = np.array(solver.vision_log_policy_1[_]) 
            aux_x = list(aux_x.flatten()) 
            print("logpi_p:",file=log_3) 
            print(aux_x,file=log_3) 
            print("state",end='',file=log_4) 
            print(_,end='',file=log_4) 
            print(":",end='',file=log_4) 
            print(env_step_list[_].obs,file=log_4) 
            aux = np.array(solver.vision_value[_]) 
            aux = list(aux.flatten()) 
            print("value:",file=log_4) 
            print(aux,file=log_4) 
            
    
    def test_serialization(self): 
        solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="kuhn_poker")) 
        solver.step() 
        state_bytes = pickle.dumps(solver)
        solver2 = pickle.loads(state_bytes) 
        self.assertEqual(solver.config, solver2.config) 
        np.testing.assert_equal( jax.device_get(solver.params), jax.device_get(solver2.params)) 
        
    if __name__ == "__main__": 
        absltest.main()