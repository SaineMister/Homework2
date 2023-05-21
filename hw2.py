import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

### ПОДЗАДАЧА 1 ####

# Считываем информацию о цитируемости статей из файла
with open('../../Dwnlds/alg_phys-cite.txt') as f:
    lines = f.readlines()

# Создаем пустой граф
G = nx.DiGraph()

# Добавляем вершины в граф
for line in lines:
    data = line.strip().split()
    node_id = int(data[0])
    G.add_node(node_id)

# Добавляем ребра в граф
for line in lines:
    data = line.strip().split()
    source_node = int(data[0])
    for i in range(1, len(data)):
        target_node = int(data[i])
        G.add_edge(target_node, source_node)

# Вычисляем распределение входящих степеней вершин
in_degrees = dict(G.in_degree())
degree_count = {}
for degree in in_degrees.values():
    if degree not in degree_count:
        degree_count[degree] = 0
    degree_count[degree] += 1

# Пронормируем распределение
total_nodes = float(len(G))
for degree in degree_count:
    degree_count[degree] /= total_nodes

# Построим лог-лог график


x = list(degree_count.keys())
y = list(degree_count.values())

plt.loglog(x, y, 'ro', markersize=3)
plt.xlabel('Входящая степень вершины')
plt.ylabel('Доля вершин')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

### --- ####

### Подзадача 2 ####

def generate_random_graph(n, p):
    V = set(range(1,n))
    E = set()

    for i in V:
        for j in V:
            if i != j and random.uniform(0, 1) < p:
                if random.choice([True, False]):
                    E.add((i, j))
                else:
                    E.add((j, i))

    return V, E

degree_distribution_random = {}
N = 3  # количество случайных графов
n = len(G.nodes())  # число вершин в реальном графе

# генерируем N случайных графов и считаем распределения входящих степеней
for _ in range(N):
    V, E = generate_random_graph(n, 0.0001)  # вероятность p выбрана так же, как в задании
    graph_random = nx.DiGraph()
    graph_random.add_edges_from(E)
    in_degrees = list(dict(graph_random.in_degree()).values())
    for degree in in_degrees:
        degree_distribution_random[degree] = degree_distribution_random.get(degree, 0) + 1

# усредняем результаты
for degree, count in degree_distribution_random.items():
    degree_distribution_random[degree] = count / (N * n)

# рисуем графики
degrees_real, probs_real = zip(*degree_count.items())
degrees_random, probs_random = zip(*degree_distribution_random.items())

plt.loglog(degrees_real, probs_real, '.', label='Реальный граф')
plt.loglog(degrees_random, probs_random, '.', label='Случайный граф')
plt.legend()
plt.xlabel('Входящая степень')
plt.ylabel('Вероятность')
plt.show()

### --- ####

### ПОДЗАДАЧА 3 ###

def generate_graph_2(n, m):
    if n <= 1 or m <= 1 or m >= n:
        return None
    
    V = set(range(m))
    E = set((i, j) for i in V for j in V if i!=j)
    
    for i in range(m, n):
        totindeg = sum([len([j for j in V if (j, i) in E])])
        V_prime = set(random.choices(list(V), weights=[len([j for j in V if (j, k) in E]) + 1 for k in V], k=m))
        V.add(i)
        E_prime = set((i, j) for j in V_prime)
        E.update(E_prime)
    
    return V, E

V = n #25836
E = len(G.edges)
m = len(G.edges) // V
print('Найденное значение n -',V)
print('Найденное значение m -',m)
### --- ####

### ПОДЗАДАЧА 4 ###
class DPATrial:

    def __init__(self, num_nodes):
        self._num_nodes = num_nodes
        self._node_set = set(range(num_nodes))
        self._adj_list = {node: set() for node in self._node_set}
        self._node_degrees = [0] * num_nodes

        # начинаем с полностью связоного графа из m узлов
        for node in self._node_set:
            self._adj_list[node].update(self._node_set.difference({node}))
            self._node_degrees[node] = num_nodes - 1

    def run_trial(self, num_nodes):
        new_node_neighbors = set()
        if num_nodes <= 0:
            return new_node_neighbors

        # выбираем m существующих узлов с одинаковой вероятностью
        m_choices = random.choices(list(self._node_set), weights=self._node_degrees, k=num_nodes)

        # добавьте новый узел на график и создайте ребра для выбранных узлов
        new_node = max(self._node_set) + 1
        self._adj_list[new_node] = set()
        for chosen_node in m_choices:
            self._adj_list[chosen_node].add(new_node)
            self._adj_list[new_node].add(chosen_node)
            self._node_degrees[chosen_node] += 1
            new_node_neighbors.add(chosen_node)

        self._node_degrees.append(len(new_node_neighbors))
        self._node_set.add(new_node)

        return new_node_neighbors


def dpa_algorithm(n, m):
    if n <= 1 or m <= 1 or m >= n:
        return None

    dpa = DPATrial(m)
    for i in range(m, n):
        neighbors = dpa.run_trial(m)

    V = set(dpa._node_set)
    E = set()
    for i in V:
        for j in dpa._adj_list[i]:
            E.add((j, i))

    return V, E


n = 25836

V, E = dpa_algorithm(n, m)

print("Количество вершин:", len(V))
print("Количество рёбер:", len(E))

in_degrees = [0] * n
for edge in E:
    in_degrees[edge[1]] += 1

plt.hist(in_degrees, bins=np.logspace(np.log10(1), np.log10(max(in_degrees) + 1), 50), alpha=0.5, density=True)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Распределение входящих степеней")
plt.xlabel("Вхождения")
plt.ylabel("Частота")
plt.show()

in_degrees = [0] * n
for edge in E:
    in_degrees[edge[1]] += 1

freq, bins = np.histogram(in_degrees, bins=np.logspace(np.log10(1), np.log10(max(in_degrees) + 1), 50))

fig, ax = plt.subplots()
ax.plot(bins[:-1], freq, 'o', color='black')
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('Вхождения')
plt.ylabel('Частота')
plt.title('Расрпеделение степеней')
plt.show()
### --- ####