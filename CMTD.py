import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

################################################################################################################################
#### Cette fonction est pour faire un regime transitoire

class Mar()
def markov(M, pi0, n):
 
    # Conditions
    if not isinstance(M, np.ndarray) or not isinstance(pi0, np.ndarray):
        print("Error becuase entries have to be numpy arrays")
    
    if M.shape[0] != M.shape[1]:
        print("La matrice de transition doit être carrée.")
    
    if pi0.shape[0] != M.shape[0]:
        print("Le nombre de lignes du vecteur d'état initial doit être égal au rang de la matrice.")

    if n < 1:
        print ("Le nombre d'itérations (m) doit être supérieur à 1.")
    
    # RQ: linlag is the library I used to perform linear algebre functions. One of the operations it supports is the matrix_power
    result = np.dot(pi0, np.linalg.matrix_power(M, n))
    
    return result

################################################################################################################################
#### Cette fonction est pour vizualizer un graphe a partir d'une matrice d'adj/transition



def plot_graph_from_trans_matrix(matrix):
    # G is going to be the directed graph
    G = nx.DiGraph(matrix)
    # Draw the graph
    nx.draw(G,with_labels=True, font_weight='bold', node_size=800, node_color='green',
            font_color='black', font_size=8, edge_color='gray', font_family='sans-serif')
    
    plt.title("Graphe associé à la Matrice d'Adjacence")
    plt.show()


################################################################################################################################
#### Cette fonction est pour verifier si une matrice est irreductible ou reductible



def is_irreducible(matrix):
    # Vérifier que la matrice est carrée
    if not matrix.shape[0] == matrix.shape[1]:
        print("La matrice doit être carrée.")

    # Créer un graphe dirigé à partir de la matrice d'adjacence
    G = nx.DiGraph(matrix)

    # Vérifier la forte connexité du graphe
    return nx.is_strongly_connected(G)



################################################################################################################################
#### Cette fonction est pour vretourner les etats absorbants


import numpy as np

def get_absorbing_states(transition_matrix):
    num_states = transition_matrix.shape[0]
    absorbing_states = []

    for state in range(num_states):
        # Vérifier si la probabilité de rester dans l'état est égale à 1 (p_{ii} = 1)
        if transition_matrix[state, state] == 1:
            absorbing_states.append(state)

    return absorbing_states



################################################################################################################################
#### Retourner les etats reccurants et transitif


def get_recurrent_and_transient_states(transition_matrix):
    num_states = transition_matrix.shape[0]
    graph = nx.DiGraph(transition_matrix)

    recurrent_states = []
    transient_states = []

    for state in range(num_states):
        # Vérifier la récurrence en vérifiant si l'état peut être atteint depuis n'importe quel autre état
        is_recurrent = all(nx.has_path(graph, source=other_state, target=state) for other_state in range(num_states) if other_state != state)

        if is_recurrent:
            recurrent_states.append(state)
        else:
            transient_states.append(state)

    return recurrent_states, transient_states


################################################################################################################################
#### Retourner les etats Periodiques et A-periodiques

def is_periodic_state(transition_matrix, state):
    num_states = transition_matrix.shape[0]

    # Check if the state has a self-loop
    if transition_matrix[state, state] > 0:
        return False

    for k in range(2, num_states + 1):
        prob_km = np.linalg.matrix_power(transition_matrix, k)[state, state]

        if prob_km > 0 and (k % np.gcd(k, num_states)) == 0:
            return True

    return False

################################################################################################################################
#### Verifier si une CMTD est periodique ou Aperiodique

def CMTD_periodique(Mat_trans):
  nb_states_periodique=0
  for state in range(Mat_trans.shape[0]):
    if is_periodic_state(Mat_trans, state):
      nb_states_periodique =nb_states_periodique+1
  return  Mat_trans.shape[0]==nb_states_periodique
        
  


################################################################################################################################
#### retourner la periode pour une CMTD periodique

def get_markov_chain_period(transition_matrix):
    num_states = transition_matrix.shape[0]
    graph = nx.DiGraph(transition_matrix)

    cycles = nx.simple_cycles(graph) # get the cycles
    cycle_lengths = [len(cycle) for cycle in cycles] # get all length of cycles
    if CMTD_periodique(transition_matrix):  # if the CMTD est periodique
      return np.gcd.reduce(cycle_lengths) # get its cycles as PGCD
    else: 
      print("La CMTD est A-periodique donc elle n'a pas de periode")


################################################################################################################################
#### Verifier si une CMTD va se converger en se basant sur la condition suffissante.

def permanant(Mat):
  if is_irreducible(Mat) and CMTD_periodique(Mat)==False :
    return True 
  else : 
    return False
    
