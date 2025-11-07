import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter


def get_data(dataset_name, fill_invalid_with=np.nan):
    """
    Load and process data from a specified dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        fill_invalid_with (float or str): Value to replace invalid RSS values. Use "No_Op" to skip replacement.

    Returns:
        tuple: Processed RSS and combined features for testing and training sets.
    """
    # Folder and file paths
    folder = r'data/'
    file_name = f"{dataset_name}.npz"
    path = folder + file_name

    # Load data
    data = np.load(path)
    tr_rss = data['tr_rss']
    ts_rss = data['ts_rss']
    tr_crd = data['tr_crd']
    ts_crd = data['ts_crd']
    tr_tms = data['tr_tms']
    ts_tms = data['ts_tms']
    invalid_value = data['nan_value']
    multi_flr_id = data['multi_fl_id']
    multi_bd_id = data['multi_bd_id']
    fl_type = data['fl_type']

    # Initialize floor data
    tr_floor_cls = tr_floor_reg = ts_floor_cls = ts_floor_reg = None

    # Process floor data
    if multi_flr_id == 1:
        fl_ind = -2 if multi_bd_id == 1 else -1

        if fl_type == 'rel':  # Convert floor to regression values if needed
            tr_crd[:, fl_ind] = fl_cls2reg(tr_crd[:, fl_ind], dataset_name)
            ts_crd[:, fl_ind] = fl_cls2reg(ts_crd[:, fl_ind], dataset_name)

        # Extract regression and classification floor features
        tr_floor_cls = fl_reg2cls(tr_crd[:, fl_ind])
        ts_floor_cls = fl_reg2cls(ts_crd[:, fl_ind])
        tr_floor_reg = tr_crd[:, fl_ind]
        ts_floor_reg = ts_crd[:, fl_ind]
    else:
        # No multi-floor information
        tr_floor_cls = np.zeros(tr_crd.shape[0])
        ts_floor_cls = np.zeros(ts_crd.shape[0])
        tr_floor_reg = np.zeros(tr_crd.shape[0])
        ts_floor_reg = np.zeros(ts_crd.shape[0])

    # Replace invalid RSS values
    if fill_invalid_with != "No_Op":
        tr_rss[tr_rss == invalid_value] = fill_invalid_with
        ts_rss[ts_rss == invalid_value] = fill_invalid_with

    # Combine all features into a single array
    if multi_bd_id == 1:
        # Include building data as a column
        tr_building = tr_crd[:, -1]
        ts_building = ts_crd[:, -1]
        tr_combined = np.column_stack((tr_floor_cls, tr_building, tr_crd[:, 0], tr_crd[:, 1], tr_floor_reg))
        ts_combined = np.column_stack((ts_floor_cls, ts_building, ts_crd[:, 0], ts_crd[:, 1], ts_floor_reg))
    else:
        # Exclude building data
        tr_combined = np.column_stack((tr_floor_cls, tr_crd[:, 0], tr_crd[:, 1], tr_floor_reg))
        ts_combined = np.column_stack((ts_floor_cls, ts_crd[:, 0], ts_crd[:, 1], ts_floor_reg))

    return ts_rss, ts_combined, ts_tms, tr_rss, tr_combined, tr_tms

def fl_cls2reg(fl,dataset_name):
    fl_high = {
        'UJI':3.0,
        'UTS': 3.0,
    }
    dif = fl_high[dataset_name]
    min_value = np.min(fl)
    scaled_categories = (np.array(fl) - min_value) * dif
    return scaled_categories

def fl_reg2cls(fl):
    unique_values = np.unique(fl)
    value_to_category = {value: index for index, value in enumerate(unique_values)}
    categories = np.array([value_to_category[value] for value in fl])
    return categories

class CoordinateRemapper:
    """
    A class to fit and transform 2D or 3D coordinates by shifting the bottom-left
    point to a specified reference point while preserving z-values if present.
    """

    def __init__(self, ref_point=(1, 1)):
        """
        Initialize the remapper with a target reference point.

        Parameters:
            ref_point (tuple): The target bottom-left point (default: (1,1)).
        """
        self.ref_point = ref_point
        self.shift_x = None
        self.shift_y = None

    def fit(self, coords):
        """
        Fit the remapper to a set of coordinates by determining the required shift.

        Parameters:
            coords (np.ndarray): A NumPy array of shape (n, 2) or (n, 3).

        Returns:
            self
        """
        if coords.shape[1] < 2:
            raise ValueError("Coordinates must have at least two dimensions (x, y).")

        # Find bottom-left (min x, min y)
        min_x, min_y = np.min(coords[:, :2], axis=0)

        # Compute shift based on the reference point
        self.shift_x = self.ref_point[0] - min_x
        self.shift_y = self.ref_point[1] - min_y

        return self

    def transform(self, coords):
        """
        Transform a set of coordinates using the fitted shift values.

        Parameters:
            coords (np.ndarray): A NumPy array of shape (n, 2) or (n, 3).

        Returns:
            np.ndarray: The transformed coordinates.
        """
        if self.shift_x is None or self.shift_y is None:
            raise RuntimeError("Remapper must be fitted before calling transform.")

        if coords.shape[1] < 2:
            raise ValueError("Coordinates must have at least two dimensions (x, y).")

        # Apply shift to x and y
        transformed_coords = coords.copy()
        transformed_coords[:, 0] += self.shift_x
        transformed_coords[:, 1] += self.shift_y

        return transformed_coords

    def fit_transform(self, coords):
        """
        Fit the remapper and transform the given coordinates in one step.

        Parameters:
            coords (np.ndarray): A NumPy array of shape (n, 2) or (n, 3).

        Returns:
            np.ndarray: The transformed coordinates.
        """
        return self.fit(coords).transform(coords)


class SequentialBlockBuilder:
    def __init__(self, grid_size=3.0, time_steps=5):
        """
        Initialize block sequence builder.
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.block_dict = {}
        self.x_min = None
        self.y_min = None

    def _assign_blocks(self, Y):
        """
        Assign block IDs using only (x, y) coordinates.
        """
        x_vals = Y[:, -3]
        y_vals = Y[:, -2]
        self.x_min, self.y_min = x_vals.min(), y_vals.min()

        x_bin = ((x_vals - self.x_min) // self.grid_size).astype(int)
        y_bin = ((y_vals - self.y_min) // self.grid_size).astype(int)

        block_ids = np.zeros(len(Y), dtype=int)
        block_map = {}
        next_block_id = 0

        for i in range(len(Y)):
            key = (x_bin[i], y_bin[i])
            if key not in block_map:
                # Compute center of the block
                x_center = self.x_min + (x_bin[i] + 0.5) * self.grid_size
                y_center = self.y_min + (y_bin[i] + 0.5) * self.grid_size
                block_map[key] = next_block_id
                self.block_dict[next_block_id] = {
                    'x': x_center,
                    'y': y_center
                }
                next_block_id += 1
            block_ids[i] = block_map[key]

        return block_ids

    def fit(self, Y):
        """
        Assign block IDs and build the block dictionary.
        Returns updated label array with block ID as new last column.
        """
        Y = np.array(Y)
        block_ids = self._assign_blocks(Y)
        return np.hstack([Y, block_ids.reshape(-1, 1)])

    def transform(self, X, Y_with_block, T, block_col=-1, floor_col=0):
        """
        Generate temporal sequences for each (floor, block) group.

        Returns:
            X_seq: [num_seq, time_steps, F]
            Y_seq: [num_seq, D] — label from last step
        """
        X = np.array(X)
        Y = np.array(Y_with_block)
        T = np.array(T)

        X_seq = []
        Y_seq = []

        all_blocks = np.unique(Y[:, block_col])
        for block in all_blocks:
            for floor in np.unique(Y[:, floor_col]):
                # Group by block and floor
                mask = (Y[:, block_col] == block) & (Y[:, floor_col] == floor)
                X_blk = X[mask]
                Y_blk = Y[mask]
                T_blk = T[mask]

                # Sort by time
                sort_idx = np.argsort(T_blk)
                X_blk = X_blk[sort_idx]
                Y_blk = Y_blk[sort_idx]

                # Create sequences
                for i in range(len(X_blk) - self.time_steps + 1):
                    x_seq = X_blk[i:i+self.time_steps]
                    y_seq = Y_blk[i+self.time_steps-1]  # label from last step
                    X_seq.append(x_seq)
                    Y_seq.append(y_seq)

        return np.array(X_seq), np.array(Y_seq)

    def get_block_info(self):
        return self.block_dict

    def assign_blocks_to_new_data(self, Y_new, max_search_scale=6, step=0.1):
        """
        Assign block IDs to new data (e.g., test set), using previously fitted grid.
        If no match, gradually increase the tolerance to find the nearest block center.
        """
        if self.x_min is None or self.y_min is None:
            raise RuntimeError("Must call fit() on training data first.")

        x_vals = Y_new[:, -3]
        y_vals = Y_new[:, -2]
        x_bin = ((x_vals - self.x_min) // self.grid_size).astype(int)
        y_bin = ((y_vals - self.y_min) // self.grid_size).astype(int)

        block_ids = np.full(len(Y_new), -1, dtype=int)

        for i in range(len(Y_new)):
            key = (x_bin[i], y_bin[i])
            found = False

            for blk_id, center in self.block_dict.items():
                cx = (center['x'] - self.x_min) // self.grid_size
                cy = (center['y'] - self.y_min) // self.grid_size
                if key == (int(cx), int(cy)):
                    block_ids[i] = blk_id
                    found = True
                    break

            #  If not found, expand search radius
            if not found:
                x0, y0 = x_vals[i], y_vals[i]
                radius = step
                while radius <= self.grid_size * max_search_scale:
                    for blk_id, center in self.block_dict.items():
                        dist = np.sqrt((center['x'] - x0) ** 2 + (center['y'] - y0) ** 2)
                        if dist <= radius:
                            block_ids[i] = blk_id
                            found = True
                            break
                    if found:
                        break
                    radius += step

        return np.hstack([Y_new, block_ids.reshape(-1, 1)])

    def transform_with_padding(self, X, Y_with_block, T, block_col=-1, floor_col=0):
        """
        Generate sequences using sliding window. For leftover data at the end of each
        (block, floor) group, pad once to create one final sequence.

        Returns:
            X_seq: [num_seq, time_steps, F]
            Y_seq: [num_seq, D]
        """
        X = np.array(X)
        Y = np.array(Y_with_block)
        T = np.array(T)

        X_seq = []
        Y_seq = []

        all_blocks = np.unique(Y[:, block_col])
        for block in all_blocks:
            for floor in np.unique(Y[:, floor_col]):
                # Filter by block and floor
                mask = (Y[:, block_col] == block) & (Y[:, floor_col] == floor)
                X_blk = X[mask]
                Y_blk = Y[mask]
                T_blk = T[mask]

                if len(X_blk) == 0:
                    continue

                # Sort by time
                sort_idx = np.argsort(T_blk)
                X_blk = X_blk[sort_idx]
                Y_blk = Y_blk[sort_idx]

                max_start = len(X_blk) - self.time_steps + 1
                if max_start > 0:
                    # Normal sliding window
                    for i in range(max_start):
                        x_seq = X_blk[i:i + self.time_steps]
                        y_seq = Y_blk[i + self.time_steps - 1]
                        X_seq.append(x_seq)
                        Y_seq.append(y_seq)

                    # Final leftover part, if any
                    if len(X_blk) > max_start:
                        partial = X_blk[-(self.time_steps - 1):]
                        pad_len = self.time_steps - len(partial)
                        x_padded = np.vstack([partial, np.tile(partial[-1], (pad_len, 1))])
                        X_seq.append(x_padded)
                        Y_seq.append(Y_blk[-1])
                else:
                    # Not enough for one full window → pad from entire block
                    pad_len = self.time_steps - len(X_blk)
                    x_padded = np.vstack([X_blk, np.tile(X_blk[-1], (pad_len, 1))])
                    X_seq.append(x_padded)
                    Y_seq.append(Y_blk[-1])

        return np.array(X_seq), np.array(Y_seq)

    def test_transform(self, X_test, Y_test_blocked, T_test, X_train, Y_train_with_block, block_col=-1,
                       floor_col=0):
        """
        Generate sequences for test set. If a block has fewer than `time_steps` samples,
        pad the rest using samples from the training set of the same (block, floor).

        Returns:
            X_seq: [num_seq, time_steps, F]
            Y_seq: [num_seq, D]
        """
        X_test = np.array(X_test)
        Y_test = np.array(Y_test_blocked)
        T_test = np.array(T_test)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train_with_block)

        X_seq = []
        Y_seq = []

        all_blocks = np.unique(Y_test[:, block_col])
        for block in all_blocks:
            for floor in np.unique(Y_test[:, floor_col]):
                mask_test = (Y_test[:, block_col] == block) & (Y_test[:, floor_col] == floor)
                X_blk_test = X_test[mask_test]
                Y_blk_test = Y_test[mask_test]
                T_blk_test = T_test[mask_test]

                if len(X_blk_test) == 0:
                    continue

                sort_idx = np.argsort(T_blk_test)
                X_blk_test = X_blk_test[sort_idx]
                Y_blk_test = Y_blk_test[sort_idx]

                if len(X_blk_test) >= self.time_steps:
                    for i in range(len(X_blk_test) - self.time_steps + 1):
                        x_seq = X_blk_test[i:i + self.time_steps]
                        y_seq = Y_blk_test[i + self.time_steps - 1]
                        X_seq.append(x_seq)
                        Y_seq.append(y_seq)
                else:
                    pad_len = self.time_steps - len(X_blk_test)

                    # Get training data for same block/floor
                    mask_train = (Y_train[:, block_col] == block) & (Y_train[:, floor_col] == floor)
                    X_blk_train = X_train[mask_train]

                    if len(X_blk_train) == 0:
                        filler = np.tile(X_blk_test[-1], (pad_len, 1))
                    else:
                        idxs = np.random.choice(len(X_blk_train), pad_len, replace=True)
                        filler = X_blk_train[idxs]

                    x_padded = np.vstack([X_blk_test, filler])
                    y_seq = Y_blk_test[-1]
                    X_seq.append(x_padded)
                    Y_seq.append(y_seq)

        return np.array(X_seq), np.array(Y_seq)



def save_sequence_dataset(
    dataset_name,
    X_train_seq, Y_train_seq,
    X_test_seq, Y_test_seq,
    X_test, Y_test,
    X_train, Y_train,
    block_info,
    save_dir='.'
):
    """
    Save sequential data and block dictionary to a compressed .npz file.

    Args:
        dataset_name: e.g., "UJI", "UTS"
        X_train_seq, Y_train_seq: training data
        X_test_seq, Y_test_seq: testing data
        block_info: dictionary of block_id → x/y/floor
        save_dir: folder to save to (default: current)
    """
    filename = f"data/{save_dir}/dataset_{dataset_name}.npz"
    np.savez(
        filename,
        X_train_seq=X_train_seq,
        Y_train_seq=Y_train_seq,
        X_test_seq=X_test_seq,
        Y_test_seq=Y_test_seq,
        X_test=X_test,
        Y_test=Y_test,
        X_train=X_train,
        Y_train=Y_train,
        block_info=np.array([block_info], dtype=object)
    )
    print(f"Saved to: {filename}")




def visualize_block_boxes(block_info, grid_size=3.0, annotate=True):
    """
    Visualize all blocks with rectangles using block centers.

    Args:
        block_info: dict → block_id → {'x', 'y'}
        grid_size: size of each block (assumed square)
        annotate: whether to annotate block IDs
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    xs = []
    ys = []

    for block_id, info in block_info.items():
        x_center = info['x']
        y_center = info['y']

        # Compute bottom-left corner
        x0 = x_center - grid_size / 2
        y0 = y_center - grid_size / 2

        # Store for axis limit scaling
        xs.append(x_center)
        ys.append(y_center)

        # Draw rectangle
        rect = Rectangle(
            (x0, y0),
            grid_size,
            grid_size,
            edgecolor='black',
            facecolor='lightblue',
            linewidth=1.2,
            alpha=0.6
        )
        ax.add_patch(rect)

        if annotate:
            ax.text(x_center, y_center, str(block_id),
                    ha='center', va='center', fontsize=7, color='black')

    # Auto-scale axes
    if xs and ys:
        margin = grid_size
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_aspect('equal')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Block Layout with Edges")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_block_sample_distribution(label_data, sort_by_count=False):
    """
    Plot number of samples per block as a bar chart.

    Args:
        label_data: [N, D+1] array, with block ID in last column
        sort_by_count: if True, sort bars by descending frequency
    """
    block_ids = label_data[:, -1].astype(int)
    counter = Counter(block_ids)

    blocks = np.array(list(counter.keys()))
    counts = np.array(list(counter.values()))

    if sort_by_count:
        sort_idx = np.argsort(-counts)
        blocks = blocks[sort_idx]
        counts = counts[sort_idx]
    else:
        sort_idx = np.argsort(blocks)
        blocks = blocks[sort_idx]
        counts = counts[sort_idx]

    plt.figure(figsize=(14, 6))
    plt.bar(blocks, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Block ID")
    plt.ylabel("Number of Samples")
    plt.title("Number of Samples per Block")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def remove_strange_data(x, y, t, predetermined_value):
    """
    Removes rows from x where all values are equal to the predetermined value
    and removes the corresponding labels from y. Also prints the number of removed rows.

    Parameters:
    x (numpy.ndarray): The input data array.
    y (numpy.ndarray): The labels array.
    predetermined_value (int or float): The value to check for removal.

    Returns:
    numpy.ndarray, numpy.ndarray: The filtered x and y arrays.
    """
    mask = ~(np.all(x == predetermined_value, axis=1))
    removed_count = np.sum(~mask)
    print(f"Number of removed rows: {removed_count}")

    return x[mask], y[mask], t[mask]

def normalize_zscore_rowwise(X):
    """
    Apply Z-score normalization row-wise (per time step), then transpose.
    Supports both 2D (F, T) and 3D (N, F, T) → and converts to (N, T, F)
    """
    X = np.array(X)

    if X.ndim == 3:
        # [N, T, F] expected as final output
        X_norm = []
        for seq in X:
            # normalize each row (time step)
            norm = (seq - seq.mean(axis=1, keepdims=True)) / (seq.std(axis=1, keepdims=True) + 1e-8)
            X_norm.append(norm)
        return np.array(X_norm)

    elif X.ndim == 2:
        # [T, F] sequence
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    else:
        raise ValueError(f"Input must be 2D or 3D, got shape: {X.shape}")
