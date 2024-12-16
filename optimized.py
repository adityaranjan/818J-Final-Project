import math
import torch
from collections import deque
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid


# each step is a multiplication overlapped by an addition so 4 cycles
STEP_TO_CYCLES = 4


class Multiplier:
    def __init__(self):
        """
        Initialize a MAC unit with default values.
        """

        self.row_idx = None  # The row index of the adjacency matrix
        self.col_idx = None  # The column index of the feature matrix
        self.nnz_idx = None
        self.current_result = 0.0

    def set_indices(self, row_idx, col_idx, nnz_idx):
        """
        Assign the row index of the adjacency matrix and the column index of the feature matrix.
        """

        self.row_idx = row_idx
        self.col_idx = col_idx
        self.nnz_idx = nnz_idx

    def multiply(self, value1, value2):
        """
        Perform the multiply-add operation: multiply two numbers and add the result to the current partial result.
        """

        self.current_result = value1 * value2

    def get_current_result(self):
        """
        Return the current partial result.
        """

        return self.current_result

    def reset(self, row_idx=None, col_idx=None, nnz_idx=None):
        """
        Reset the MAC unit's state for reuse.
        Optionally update the row and column indices.
        """

        self.current_result = 0.0
        if row_idx is not None:
            self.row_idx = row_idx
        if col_idx is not None:
            self.col_idx = col_idx
        if nnz_idx is not None:
            self.nnz_idx = nnz_idx


class OptimizedSimulator:
    def __init__(self, dataset: Dataset, num_multipliers=1024):
        """
        Initialize the simulator with parameters such as the number of MACs.
        """

        self.dataset = dataset
        self.data = self.dataset[0]

        self.num_multipliers = num_multipliers
        self.multiplier_array = [Multiplier() for _ in range(num_multipliers)]

        self.adj_matrix = None
        self.feature_matrix = None
        self.weight_matrix = None
        self.output_matrix = None

        # Metrics tracking
        self.completed_tasks = 0
        self.average_latency = 0
        self.max_latency = float("-inf")
        self.min_latency = float("inf")

        self.agg_steps = 0
        self.add_steps = 0

    def load_data(self, num_features=None):
        """
        Load data from a PyTorch Geometric graph object.
        """

        # normalize adjacency matrix
        gcn_norm = T.GCNNorm()
        self.data = gcn_norm.forward(self.data)

        # create adjacency matrix as sparse csr tensor
        self.num_nodes = self.data.x.shape[0]
        self.adj_matrix = torch.sparse_coo_tensor(
            self.data.edge_index,
            self.data.edge_weight,
            (self.num_nodes, self.num_nodes),
        ).to_sparse_csr()

        # option to specify number of features for testing with "synthetic" data
        if num_features is None:
            self.feature_matrix = self.data.x
            self.num_features = self.data.x.shape[1]
        else:
            self.feature_matrix = torch.rand(self.num_nodes, num_features)
            self.num_features = num_features

        self.aggregation_result = torch.zeros(self.num_nodes, self.num_features)

        print("Data loaded successfully.")

    def set_weight_matrix(self, num_classes=None):
        """
        Set the weight matrix for the layer.
        """

        # option to specify number of classes for testing with "synthetic" data
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = self.dataset.num_classes

        full_weight = torch.empty(self.num_features, self.num_classes)
        torch.nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        self.weight_matrix = full_weight

        self.output_matrix = torch.zeros(
            (self.num_features, self.num_classes), dtype=torch.float32
        )

    def simulate(self):
        """
        Simulate MAC operations with fine-grained dynamic workload assignment.
        """

        # contains the steps at which each row becomes available for combination
        self.row_entry_list = [float("inf") for _ in range(self.num_nodes)]
        self.row_ready_list = [-1 for _ in range(self.num_nodes)]

        # Work queue: each entry is (row_idx, col_idx, nonzero_indices, nonzero_values, curr_nnz_idx)
        work_queue = deque()
        total_tasks = self.num_nodes * self.num_features

        # Populate the initial work queue
        for row_idx in range(self.num_nodes):
            row = self.adj_matrix[row_idx]
            nonzero_indices = row.indices()[0]
            if len(nonzero_indices) > 0:
                nonzero_values = row.values()
                for curr_col in range(self.num_features):
                    work_queue.append(
                        [
                            row_idx,  # row of aggregation
                            curr_col,  # col of aggregation
                            nonzero_indices,
                            nonzero_values,
                            0,  # current unprocessed nnz index
                            None,  # start step
                        ]
                    )

        multiplier_assignments, steps = [None] * self.num_multipliers, 0

        while work_queue or any(multiplier_assignments):
            steps += 1

            # Assign work to idle Multipliers
            for multiplier_idx in range(self.num_multipliers):
                if multiplier_assignments[multiplier_idx] is None and work_queue:
                    queue_head = work_queue[0]

                    # if an element of the aggregation matrix is done being progressed
                    if queue_head[4] == len(queue_head[3]):
                        self.completed_tasks += 1

                        row_idx, curr_col, _, _, curr_nnz_idx, _ = work_queue.popleft()

                        self.row_ready_list[row_idx] = max(
                            self.row_ready_list[row_idx], steps
                        )

                        # if a row of the aggregation matrix is done being processed,
                        # calculate the number of extra aggregation steps needed till it is ready for combination
                        if curr_col == self.num_features - 1:
                            if curr_nnz_idx > len(self.multiplier_array):
                                agg_add_steps = math.ceil(
                                    (
                                        (
                                            math.floor(
                                                curr_nnz_idx
                                                / len(self.multiplier_array)
                                            )
                                            * math.ceil(
                                                math.log2(len(self.multiplier_array))
                                            )
                                        )
                                        + math.ceil(
                                            math.log2(
                                                curr_nnz_idx
                                                % len(self.multiplier_array)
                                            )
                                        )
                                    )
                                    / 2
                                )
                            else:
                                agg_add_steps = math.ceil(
                                    math.ceil(math.log2(curr_nnz_idx)) / 2
                                )
                            self.row_ready_list[row_idx] += agg_add_steps

                            # add trailing add steps
                            if len(work_queue) == 0:
                                self.add_steps += agg_add_steps

                    if work_queue:
                        queue_head = work_queue[0]

                        if queue_head[5] is None:
                            # keep track of when the row started being processed
                            self.row_entry_list[queue_head[0]] = min(
                                self.row_entry_list[queue_head[0]], steps
                            )
                            work_queue[0][5] = steps

                        # assign work to idle multiplier
                        multiplier_assignments[multiplier_idx] = (
                            queue_head[0],
                            queue_head[1],
                            queue_head[2],
                            queue_head[3],
                            queue_head[4],
                        )
                        work_queue[0][4] += 1

            # Process each assigned Multiplier
            for multiplier_idx in range(self.num_multipliers):
                assignment = multiplier_assignments[multiplier_idx]
                if assignment is not None:
                    row_idx, curr_col, nonzero_indices, nonzero_values, nnz_idx = (
                        assignment
                    )

                    # reset the multipler with its new assignment
                    curr_multiplier = self.multiplier_array[multiplier_idx]
                    curr_multiplier.reset(
                        row_idx=row_idx, col_idx=curr_col, nnz_idx=nnz_idx
                    )

                    # perform the multiplication for the assignment
                    value = nonzero_values[nnz_idx]
                    col_idx = nonzero_indices[nnz_idx]
                    curr_multiplier.multiply(
                        value, self.feature_matrix[col_idx][curr_col]
                    )

                    # update the aggregation result
                    self.aggregation_result[row_idx][
                        curr_col
                    ] += curr_multiplier.get_current_result()

                    multiplier_assignments[multiplier_idx] = None

            if self.completed_tasks % 100 == 0:
                print(
                    f"Progress: {self.completed_tasks}/{total_tasks} ({100 * self.completed_tasks / total_tasks:.2f}%)"
                )

        # simulate combination
        total_steps = self.combination_simulate()

        # calculate the total number of aggregation steps
        self.agg_steps = max(self.row_ready_list)

        return self.agg_steps, total_steps

    def combination_simulate(self):
        # keep track of combination start step so overlap can be calculated
        self.combination_start_step = float("inf")

        comb_add_steps = 0
        work_queue = deque()
        total_tasks = self.num_nodes * self.num_classes
        self.completed_tasks = 0

        multiplier_assignments = [None] * self.num_multipliers
        steps = min(self.row_ready_list)

        # work queue is organized so that we process vertices in order
        sorted_indices = [
            i for i, _ in sorted(enumerate(self.row_ready_list), key=lambda x: x[1])
        ]
        for row_idx in sorted_indices:
            for curr_col in range(self.num_classes):
                # row, col of the output matrix, current nonzero index, start step
                work_queue.append([row_idx, curr_col, 0, None])

        while work_queue or any(multiplier_assignments):
            steps += 1

            # if you haven't hit the step for the next vertex yet, don't do anything
            if steps < self.row_ready_list[work_queue[0][0]]:
                continue
            else:
                self.combination_start_step = min(self.combination_start_step, steps)

            # Assign work to idle Multipliers
            for multiplier_idx in range(self.num_multipliers):
                if multiplier_assignments[multiplier_idx] is None and work_queue:
                    queue_head = work_queue[0]

                    if queue_head[2] == self.num_features:
                        self.completed_tasks += 1
                        work_queue.popleft()

                        # calculate combination add steps to add to vertex latency
                        if queue_head[1] == self.num_classes - 1:
                            if self.num_features > len(self.multiplier_array):
                                comb_add_steps = math.ceil(
                                    (
                                        (
                                            math.floor(
                                                self.num_features
                                                / len(self.multiplier_array)
                                            )
                                            * math.ceil(
                                                math.log2(len(self.multiplier_array))
                                            )
                                        )
                                        + math.ceil(
                                            math.log2(
                                                self.num_features
                                                % len(self.multiplier_array)
                                            )
                                        )
                                    )
                                    / 2
                                )
                            else:
                                comb_add_steps = math.ceil(
                                    math.ceil(math.log2(self.num_features)) / 2
                                )

                            current_vertex_latency = (
                                steps + comb_add_steps
                            ) - self.row_entry_list[queue_head[0]]
                            self.average_latency += current_vertex_latency
                            self.max_latency = max(
                                self.max_latency, current_vertex_latency
                            )
                            self.min_latency = min(
                                self.min_latency, current_vertex_latency
                            )

                            if len(work_queue) == 0:
                                self.add_steps += comb_add_steps

                    if work_queue:
                        queue_head = work_queue[0]

                        # if you have reached the step where the data has arrived from aggregation, start processing
                        if steps >= self.row_ready_list[queue_head[0]]:
                            if queue_head[3] is None:
                                work_queue[0][3] = steps

                            # assigning different elements of the dot product to different multipliers
                            work_queue[0][2] += 1

            # TODO: process the actual combination steps, left out now for speed reasons

            if self.completed_tasks % 100 == 0:
                print(
                    f"Progress: {self.completed_tasks}/{total_tasks} ({100 * self.completed_tasks / total_tasks:.2f}%)"
                )

        self.output_matrix = self.aggregation_result @ self.weight_matrix

        return steps + comb_add_steps

    def get_output(self):
        """
        Retrieve the final output matrix after processing.
        """

        return self.output_matrix

    def get_latency_metrics(self):
        """
        Retrieve latency metrics: average, maximum, and minimum latencies.
        """

        self.average_latency = self.average_latency / self.num_nodes
        return {
            "average_latency": STEP_TO_CYCLES * self.average_latency,
            "max_latency": STEP_TO_CYCLES * self.max_latency,
            "min_latency": STEP_TO_CYCLES * self.min_latency,
        }


# Example usage
if __name__ == "__main__":
    # Example graph from PyTorch Geometric
    simulator = OptimizedSimulator(
        Planetoid(name="Cora", root="/Users/adi/Downloads/Planetoid")
    )
    simulator.load_data()
    simulator.set_weight_matrix()  # Example weight matrix

    agg_steps, total_steps = simulator.simulate()

    print("")
    print(
        "Number of Agg Only Cycles: "
        + str(STEP_TO_CYCLES * simulator.combination_start_step)
    )
    print(
        "Number of Combination Overlapped Cycles: "
        + str(STEP_TO_CYCLES * (agg_steps - simulator.combination_start_step))
    )
    print(
        "Number of Combination Only Cycles: "
        + str(STEP_TO_CYCLES * (total_steps - agg_steps))
    )
    print("Number of Total Cycles: " + str(STEP_TO_CYCLES * total_steps))

    print("")
    print(simulator.get_latency_metrics())

    output = simulator.get_output()

    print("")
    print(
        "Output Correct: "
        + str(
            torch.equal(
                output,
                simulator.adj_matrix
                @ simulator.feature_matrix
                @ simulator.weight_matrix,
            )
        )
    )

    print("")
    print(simulator.row_ready_list)
