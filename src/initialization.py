dim = iface.number_of_dimensions()
S = 0.6
EPOCHS = 500  # at least len(data)
max_syn = 3
update_step = S / (len(data))
buffer_value = S / (2 * update_step)
max_update_steps = iface.number_of_dimensions()
max_width = S / 2
count_syn = 0
count = 0
count_max = 2  # S/update_step
break_out_flag = False
break_out_flag_2 = False
minimal_change = 0.0001
forced_syn = True
stop_syn1 = False
count_sny2 = 0

# finally, nodes will contain {'center_val': value that has to be contained

# in all neighbouring intervals,'neighbours': a (dim x undef) list with node
# indexes that are direct neighbours. }

nodes = []

# build up node map and instantiate all nodes with start values
start_f_val = np.mean([iface.evaluate_f(d) for d in data[:100]])
# take average of first 100 values
discret_start_f_val = round(start_f_val / update_step)

### range(iface.number_of_nodes()) = {0,...,iface.number_of_nodes()-1}
for i in range(iface.number_of_nodes()):
    nodes.append(
        {
            "center_val": discret_start_f_val,
            "lower_buffer": buffer_value,
            "upper_buffer": buffer_value,
        }
    )

# create a list all possible combinations of node IDs and assign each of them
# to a node:
# find all combinations of thresholds
dimension_combination = []

# find all combinations per dimensions
for dim in range(iface.hypercube_dim):
    dimension_combination.append(list(itertools.product(*thresholds[dim][:])))

threshold_combinations = list(itertools.product(*dimension_combination))

for i, node in enumerate(nodes):
    threshold_node = [list(x) for x in threshold_combinations[i]]
    # save it as list of lists instead as tuple of tuples
    node["threshold"] = threshold_node

# build up node <--> data_point reference for later use
node_of_data_set = []
for data_set in data:
    node_ids = iface.get_node_id(data_set)
    node_of_data_set.append(node_ids)

# compute f value for each datapoint
f_of_data_set = []
for data_set in data:
    f_of_data_set.append(iface.evaluate_f(data_set))

# build up two lists one which contains all DP-neighbours
# (nodes that share an interval in any dimension)
# now one node_id has the length of num_attribute

for id_a, node in enumerate(nodes):
    neighbours = [[] for _ in range(iface.number_of_dimensions())]
    for id_b in range(iface.number_of_nodes()):
        if id_a == id_b:
            # we do not want the node itself listed as neighbour
            continue

        else:
            neighbour_dims = iface.nodes_are_neighbours(id_a, id_b)
            [neighbours[dim].append(id_b) for dim in neighbour_dims]

    node["neighbours"] = neighbours


# Assign Nodes to Ranks

adapted_nodes_total = []  # to save all adjusted nodes with arbitrary rank
adapted_nodes_rank0 = []  # to save all adjusted nodes with rank 0
# i.e. nodes which contain a trainingpkt
adapted_nodes_rank1 = []  # to save all adjusted nodes with rank 1
adapted_nodes_rank2 = []  # to save all adjusted nodes with rank 2
adapted_nodes_rank3 = []  # to save all adjusted nodes with rank 3
adapted_nodes_rank4 = []  # to save all adjusted nodes with rank 4
adapted_nodes_rank5 = []  # to save all adjusted nodes with rank 5

# collect all nodes with rank 0
for i_d, _ in enumerate(data):
    node_id = node_of_data_set[i_d]
    node = nodes[node_id]
    if node_id not in adapted_nodes_total:
        adapted_nodes_total.append(node_id)
        adapted_nodes_rank0.append(node_id)

# collect all nodes of rank 1
for i_d, _ in enumerate(data):
    node_id = node_of_data_set[i_d]
    node = nodes[node_id]
    for dim in range(iface.number_of_dimensions()):
        for n_node_rk1_id in node["neighbours"][dim]:
            n_node_rk1 = nodes[n_node_rk1_id]

            if n_node_rk1_id not in adapted_nodes_total:
                adapted_nodes_total.append(n_node_rk1_id)
                adapted_nodes_rank1.append(n_node_rk1_id)

# collect all nodes of rank 2
for n_node_rk1_id in adapted_nodes_rank1:
    for dim2 in range(iface.number_of_dimensions()):
        n_node_rk1 = nodes[n_node_rk1_id]
        for n_node_rk2_id in n_node_rk1["neighbours"][dim2]:
            n_node_rk2 = nodes[n_node_rk2_id]

            if n_node_rk2_id not in adapted_nodes_total:
                adapted_nodes_total.append(n_node_rk2_id)
                adapted_nodes_rank2.append(n_node_rk2_id)

# collect all nodes of rank 3
for n_node_rk2_id in adapted_nodes_rank2:
    for dim3 in range(iface.number_of_dimensions()):
        n_node_rk2 = nodes[n_node_rk2_id]
        for n_node_rk3_id in n_node_rk2["neighbours"][dim3]:
            n_node_rk3 = nodes[n_node_rk3_id]

            if n_node_rk3_id not in adapted_nodes_total:
                adapted_nodes_total.append(n_node_rk3_id)
                adapted_nodes_rank3.append(n_node_rk3_id)

# collect all nodes of rank 4
for n_node_rk3_id in adapted_nodes_rank3:
    for dim4 in range(iface.number_of_dimensions()):
        n_node_rk3 = nodes[n_node_rk3_id]
        for n_node_rk4_id in n_node_rk3["neighbours"][dim4]:
            n_node_rk4 = nodes[n_node_rk4_id]

            if n_node_rk4_id not in adapted_nodes_total:
                adapted_nodes_total.append(n_node_rk4_id)
                adapted_nodes_rank4.append(n_node_rk4_id)

# collect all nodes of rank 5 or higher
for id, _ in enumerate(nodes):
    if id not in adapted_nodes_total:
        adapted_nodes_total.append(id)
        adapted_nodes_rank5.append(id)
