import numpy as np
def forward(A, B, pi, O):
    #Algorithm compexity is TN^2
    number_of_state=len(pi) #N
    time_steps = len(O)     #T
    state_list = np.zeros((number_of_state,time_steps))
    branch=0
    for i in range(time_steps): #T
        
        for state in range(number_of_state):#N
            if(i==0):
                #print(B[state][O[i]])
                #print(pi[state])
                res = pi[state]*B[state][O[i]]
                state_list[state][i]=res
            else:
                for second_state in range(number_of_state): #N
                    branch+=A[second_state][state]*B[state][O[i]]*state_list[second_state][i-1]
                state_list[state][i]=branch
                branch=0   
    result = state_list.sum(axis=0)
    x_result=result[time_steps-1]          
    return x_result, state_list
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """


def viterbi(A, B, pi, O):
    
     #Algorithm compexity is TN^2
    number_of_state=len(pi) #N
    time_steps = len(O)     #T
    state_list = np.zeros((number_of_state,time_steps))
    branch=0
    max=-1
    backward_state_list=np.zeros((number_of_state,time_steps-1))
    result_list=np.zeros((time_steps))
    
    for i in range(time_steps): #T
        
        for state in range(number_of_state):#N
            if(i==0):
                #print(B[state][O[i]])
                #print(pi[state])
                res = pi[state]*B[state][O[i]]
                state_list[state][i]=res
            else:
                for second_state in range(number_of_state): #N
                    branch=A[second_state][state]*B[state][O[i]]*state_list[second_state][i-1]
                    if branch > max:
                        max=branch
                        max_state=second_state
                state_list[state][i]=max   
                backward_state_list[state][i-1]=int(max_state)
                max=-1
    arr = np.argmax(state_list,axis=0)      
    last_state=int(arr[time_steps-1])
    result_list[time_steps-1]=last_state
    for i in range(time_steps-1):
        result_list[time_steps-2-i]=backward_state_list[last_state][time_steps-i-2]
        last_state=int(result_list[time_steps-2-i])
             
    return result_list, state_list
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
