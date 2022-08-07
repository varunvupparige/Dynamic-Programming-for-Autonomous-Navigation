import numpy as np
import gym
import copy
from utils import *
from example import example_use_of_gym_env

v_inf = 999999

def gen_state_space(env,info):

    grid_height = env.height
    grid_width = env.width
    orientation = [np.array([[1],[0]]),np.array([[0],[1]]), np.array([[-1],[0]]),np.array([[0],[-1]])]
    keydoor=[np.array([[1],[0]]),np.array([[1],[1]]), np.array([[0],[0]])]
    #state_space_size = grid_height*grid_height*4*2*2
    state_space = []
    
    for i in range(grid_width):
        for j in range(grid_height):
            for k in keydoor:
                for l in orientation:

                        state = {}
                        state['position']=np.array([[i],[j]])
                        state['keydoor'] = k
                        state['orientation']= l
                        
                        if((env.grid.get(i,j)==None)):
                            state_space.append(state)
                        elif(env.grid.get(i,j).type!='wall'):
                            state_space.append(state)

    #state_space = np.array(state_space).reshape((state_space_size,5))
    return state_space


def motion_model(present_state,action,door,key,env):
    
    
    orientation = [np.array([[1],[0]]),np.array([[0],[1]]), np.array([[-1],[0]]),np.array([[0],[-1]])]
    
    next_state=copy.deepcopy(present_state)
    
    if action==0: 

        #temp_state = copy.deepcopy(next_state)
        #temp_state['position'] = p_state['position'] + p_state['orientation']
        #if(env.grid.get(temp_state['position'][0][0],temp_state['position'][0][0]) == 'wall'):
            #next_state = copy.deepcopy(present_state)
        
        if(next_state['position'][0][0]==door[0] and next_state['position'][1][0]==door[1] and next_state['keydoor'][1][0]!=1):
            return next_state

        next_state['position']=next_state['position']+next_state['orientation']
    
    if action==1: 

        if(next_state['orientation'][0] == orientation[0][0] and next_state['orientation'][1] == orientation[0][1]):
            next_state['orientation'] = orientation[3]
        else:
            index = np.where(orientation == next_state['orientation'])[0][0]
            next_state['orientation'] = orientation[index-1]
        
        #if(np.array_equal(next_state['orientation'],np.array([[1],[0]]))):
         #   next_state['orientation'] = np.array([[0],[-1]])

        #if(np.array_equal(next_state['orientation'],np.array([[0],[-1]]))):
         #   next_state['orientation'] = np.array([[-1],[0]])
        
        #if(np.array_equal(next_state['orientation'],np.array([[-1],[0]]))):
         #   next_state['orientation'] = np.array([[0],[1]])

        #if(np.array_equal(next_state['orientation'],np.array([[0],[1]]))):
         #   next_state['orientation'] = np.array([[1],[0]])

    if action==2:

        if(next_state['orientation'][0] == orientation[3][0] and next_state['orientation'][1] == orientation[3][1]):
            next_state['orientation'] = orientation[0]
        else:
            index = np.where(orientation == next_state['orientation'])[0][0]
            next_state['orientation'] = orientation[index+1]

        #if(np.array_equal(next_state['orientation'],np.array([[1],[0]]))):
           # next_state['orientation'] = np.array([[0],[1]])

        #if(np.array_equal(next_state['orientation'],np.array([[0],[1]]))):
         #   next_state['orientation'] = np.array([[-1],[0]])
        
        #if(np.array_equal(next_state['orientation'],np.array([[-1],[0]]))):
         #   next_state['orientation'] = np.array([[0],[-1]])

        #if(np.array_equal(next_state['orientation'],np.array([[0],[-1]]))):
         #   next_state['orientation'] = np.array([[1],[0]])
    
    if action==3: 
        if(next_state['position'][0][0] + next_state['orientation'][0][0]==key[0] and next_state['position'][1][0]+next_state['orientation'][1][0]==key[1]):
            next_state['keydoor'][0][0]=1
    
    if action==4: 
        if(next_state['position'][0][0] + next_state['orientation'][0][0]==door[0] and next_state['position'][1][0] + next_state['orientation'][1][0]==door[1] and next_state['keydoor'][0][0]==1):
            next_state['keydoor'][1][0]=1
    
    return next_state


def cost_s2s(state_space,env,door,key):

    v_inf = 999999
    c_ij = np.full((len(state_space),len(state_space)),v_inf)

    for i in range(len(state_space)):
        for j in range(len(state_space)):

            if(i!=j):
                
                for k in range(5):
                    next_state_list = motion_model(state_space[i],k,door,key,env)
                    if(np.array_equal(next_state_list['position'],state_space[j]['position']) and np.array_equal(next_state_list['keydoor'],state_space[j]['keydoor']) and np.array_equal(next_state_list['orientation'],state_space[j]['orientation'])):
                        #print(j," ",i)
                        c_ij[i,j] = 1
                        
            elif(i == j):
                c_ij[i,j] = 0
    
    return c_ij

def action(policy,info,stateSpace):
    
    
    arrPolicy=np.asarray(policy)
    
    startState={}
    startState['position']=info['init_agent_pos'].reshape(2,1)
    startState['keydoor']=np.array([[0],[0]])
    startState['orientation']=info['init_agent_dir'].reshape(2,1)
   
    for i in range(len(stateSpace)):
        if(np.array_equal(startState['position'],stateSpace[i]['position']) and np.array_equal(startState['keydoor'],stateSpace[i]['keydoor']) and np.array_equal(startState['orientation'],stateSpace[i]['orientation'])):
                       
            index = i
            break
    
    action=[]
    action.append(index)
    tempAction=index
    
    for i in range(len(policy)-1,-1,-1):
        tempAction=arrPolicy[i][tempAction]
        action.append(tempAction)
    optimal_action=[]
    
    for j, k in zip(action,action[1:]):
        if(np.array_equal(stateSpace[j]['position'],stateSpace[k]['position']) and np.array_equal(stateSpace[j]['keydoor'],stateSpace[k]['keydoor']) and np.array_equal(stateSpace[j]['orientation'],stateSpace[k]['orientation'])):
            continue
        
        optimal_action.append(action2state(stateSpace[j],stateSpace[k]))
    return optimal_action


def action2state(present_state,next_state):
    
    if(np.array_equal(next_state['position'], present_state['position'] + present_state['orientation'])):
        return 0
    
    if(np.array_equal(next_state['orientation'], np.array([[0, 1], [-1, 0]]).dot(present_state['orientation']))):
        return 1
    
    if (np.array_equal(next_state['orientation'],np.array([[0, -1], [1, 0]]).dot(present_state['orientation']))):  # Turn right
        return 2
   
    if (next_state['keydoor'][0][0] == present_state['keydoor'][0][0]+1):
        return 3
   
    if (next_state['keydoor'][1][0] == present_state['keydoor'][1][0]+1):
        return 4

def doorkey_problem(env,info):

    door = info['door_pos']
    key = info['key_pos']
    
    state_space = gen_state_space(env,info)
    goal = info['goal_pos']
    
    time_horizon = len(state_space) - 1
    value_fn = v_inf*np.ones((len(state_space),time_horizon))
    goal_index = []
    policy = []

    for i in range(len(state_space)):
        if((state_space[i]['position'][0][0] == goal[0]) and (state_space[i]['position'][1][0] == goal[1])):
            goal_index.append(i)
        
    for j in goal_index:
        value_fn[j,-1] = 0

    c_ij = cost_s2s(state_space,env,door,key)

    for k in range(time_horizon - 2,-1,-1):

        q = np.zeros((len(state_space),len(state_space)))

        q = c_ij + value_fn[:,k+1]
        policy.append(np.argmin(q,axis=1))
        #print(policy)
        value_fn[:,k]=np.amin(q, axis=1)
        if(np.array_equal(value_fn[:,k],value_fn[:,k+1])):
            #print(j)
            break
    
    policy=np.asarray(policy) 
    opt_action = action(policy,info,state_space)
    
    
    return policy.T, value_fn, opt_action

def partA():
    env_path = './envs/example-8x8.env'
    env, info = load_env(env_path) # load an environment
    policy, value_fn, seq = doorkey_problem(env) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save
    

if __name__ == '__main__':
    partA()
    

        
        
    
