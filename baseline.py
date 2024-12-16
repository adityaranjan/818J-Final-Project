import math
import torch
from collections import deque
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid


# each step is a multiplication followed by an addition so 4 + 2 = 6 cycles
STEP_TO_CYCLES = 6


class MAC:
    def __init__(self):
        """
        Initialize a MAC unit with default values.
        """

        self.row_idx = None  # The row index of the adjacency matrix
        self.col_idx = None  # The column index of the feature matrix
        self.current_result = 0.0  # Accumulated result

    def set_indices(self, row_idx, col_idx):
        """
        Assign the row index of the adjacency matrix and the column index of the feature matrix.
        """

        self.row_idx = row_idx
        self.col_idx = col_idx

    def multiply_add(self, value1, value2):
        """
        Perform the multiply-add operation: multiply two numbers and add the result to the current partial result.
        """

        self.current_result += value1 * value2

    def get_current_result(self):
        """
        Return the current accumulated result.
        """

        return self.current_result

    def reset(self, row_idx=None, col_idx=None):
        """
        Reset the MAC unit's state for reuse.
        Optionally update the row and column indices.
        """

        self.current_result = 0.0
        if row_idx is not None:
            self.row_idx = row_idx
        if col_idx is not None:
            self.col_idx = col_idx


class BaselineSimulator:
    def __init__(self, dataset: Dataset, num_macs=1024):
        """
        Initialize the simulator with parameters such as the number of MACs.
        """

        self.dataset = dataset
        self.data = self.dataset[0]

        self.num_macs = num_macs
        self.mac_array = [MAC() for _ in range(num_macs)]

        # systolic array will be of size self.systolic_dim x self.systolic_dim
        self.systolic_dim = 32

        self.adj_matrix = None
        self.feature_matrix = None
        self.weight_matrix = None
        self.output_matrix = None

        # Metrics tracking
        self.average_latency = 0
        self.max_latency = float("-inf")
        self.min_latency = float("inf")

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

    def simulate(self):
        """
        Simulate MAC operations with fine-grained dynamic workload assignment.
        """

        self.aggregation_locality = [float("-inf")] * self.num_nodes
        self.row_start_step = [float("inf")] * self.num_nodes

        # Work queue: each entry is (row_idx, col_idx, nonzero_indices, nonzero_values, progress, start_step)
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
                        (row_idx, curr_col, nonzero_indices, nonzero_values, 0, None)
                    )  # No start step yet

        mac_assignments = [None] * self.num_macs
        steps, completed_tasks = 0, 0

        while work_queue or any(mac_assignments):
            steps += 1

            # Assign work to idle MACs at each step
            for mac_idx in range(self.num_macs):
                if mac_assignments[mac_idx] is None and work_queue:
                    task = work_queue.popleft()
                    if task[5] is None:  # If start_step is not set
                        self.row_start_step[task[0]] = min(
                            self.row_start_step[task[0]], steps
                        )
                        task = (*task[:-1], steps)  # Add the current step as start_step
                    mac_assignments[mac_idx] = task

            # Process each assigned MAC
            for mac_idx in range(self.num_macs):
                # skip MACs that don't have any assignment
                assignment = mac_assignments[mac_idx]
                if assignment is not None:
                    (
                        row_idx,
                        curr_col,
                        nonzero_indices,
                        nonzero_values,
                        progress,
                        start_step,
                    ) = assignment

                    # if the MAC is just starting on its assignment, reset it
                    curr_mac = self.mac_array[mac_idx]
                    if progress == 0:
                        curr_mac.reset(row_idx=row_idx, col_idx=curr_col)

                    # accumulate the current product
                    value = nonzero_values[progress]
                    col_idx = nonzero_indices[progress]
                    curr_mac.multiply_add(value, self.feature_matrix[col_idx][curr_col])

                    progress += 1

                    # update MAC assignment based on the progress made
                    if progress < len(nonzero_values):
                        mac_assignments[mac_idx] = (
                            row_idx,
                            curr_col,
                            nonzero_indices,
                            nonzero_values,
                            progress,
                            start_step,
                        )
                    else:
                        self.aggregation_locality[row_idx] = max(
                            self.aggregation_locality[row_idx], steps
                        )

                        completed_tasks += 1

                        # update aggegration result
                        self.aggregation_result[row_idx][
                            curr_col
                        ] = curr_mac.get_current_result()

                        mac_assignments[mac_idx] = None

            if completed_tasks % 100 == 0:
                print(
                    f"Progress: {completed_tasks}/{total_tasks} ({100 * completed_tasks / total_tasks:.2f}%)"
                )

        # perform combination
        self.systolic_array_multiply(self.aggregation_result, self.weight_matrix)

        self.aggregation_steps = steps

        return self.aggregation_steps, self.combination_steps

    def systolic_array_multiply(self, aggregated_matrix, weight_matrix):
        """
        Perform multiplication using a systolic array.
        """

        self.output_matrix = aggregated_matrix @ weight_matrix

        # calculate total number of steps for the combination given the number of features, number of classes, and the systolic array size
        self.combination_steps = (
            math.ceil(self.num_features / self.systolic_dim)
            * math.ceil(self.num_classes / self.systolic_dim)
            * ((2 * self.systolic_dim) + self.num_nodes + self.num_classes)
        )

        return self.combination_steps

    def get_output(self):
        """
        Retrieve the final output matrix after processing.
        """

        return self.output_matrix

    def get_latency_metrics(self):
        """
        Retrieve latency metrics: average, maximum, and minimum latencies.
        """

        for i in range(self.num_nodes):
            current_vertex_latency = (
                self.aggregation_steps
                + ((i + 1) * (self.combination_steps // self.num_nodes))
                - self.row_start_step[i]
            )
            self.average_latency += current_vertex_latency
            self.min_latency = min(self.min_latency, current_vertex_latency)
            self.max_latency = max(self.max_latency, current_vertex_latency)
        self.average_latency /= self.num_nodes

        return {
            "average_latency": STEP_TO_CYCLES * self.average_latency,
            "max_latency": STEP_TO_CYCLES * self.max_latency,
            "min_latency": STEP_TO_CYCLES * self.min_latency,
        }


# Example usage
if __name__ == "__main__":
    # Example graph from PyTorch Geometric
    simulator = BaselineSimulator(
        Planetoid(name="Cora", root="/Users/adi/Downloads/Planetoid")
    )
    simulator.load_data()
    simulator.set_weight_matrix()  # Example weight matrix

    agg_steps, comb_steps = simulator.simulate()

    print("")
    print("Number of Agg Cycles: " + str(STEP_TO_CYCLES * agg_steps))
    print("Number of Combination Cycles: " + str(STEP_TO_CYCLES * comb_steps))
    print("Number of Total Cycles: " + str((STEP_TO_CYCLES * (agg_steps + comb_steps))))

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
    print(simulator.aggregation_locality)
