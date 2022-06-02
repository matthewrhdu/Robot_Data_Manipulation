import numpy as np
from typing import Optional, Tuple


def SegmentPlane(distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 100,
        probability: float = 0.99999999,
        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if probability <= 0 or probability > 1:
        print("Probability must be > 0 or <= 1.0")
        raise ValueError

    # RANSACResult result;

    # Initialize the best plane model.
    best_plane_model = np.array([0, 0, 0, 0])

    num_points = self.points_.size
    sampler = RandomSampler(num_points, seed)

    # Return if ransac_n is less than the required plane model parameters.
    if ransac_n < 3:
        print("ransac_n should be set to higher than or equal to 3.")
        return np.array([0, 0, 0, 0]), np.ndarray([])

    if num_points < ransac_n:
        print("There must be at least 'ransac_n' points.")
        return np.array([0, 0, 0, 0]), np.ndarray([])

    # Use size_t here to avoid large integer which acceed max of int.
    # size_t break_iteration = std::numeric_limits<size_t>::max();
    break_iteration = 1e32
    iteration_count = 0

    for itr in range(num_iterations):
        if iteration_count > break_iteration:
            continue

        sampled_indices = sampler(ransac_n)  ## O(n)
        inliers = sampled_indices

        # Fit model to num_model_parameters randomly selected points among the inliers.
        # Eigen::Vector4d plane_model;
        if ransac_n == 3:
            plane_model = TriangleMesh.ComputeTrianglePlane(self.points_[inliers[0]], self.points_[inliers[1]], self.points_[inliers[2]]);
        else:
            plane_model = GetPlaneFromPoints(self.points_, inliers)  # O(n)

        if plane_model.isZero(0):
            continue

        error = 0
        inliers.clear()
        this_result = EvaluateRANSACBasedOnDistance(points_, plane_model, inliers, distance_threshold, error)
        if this_result.fitness_ > result.fitness_ or (this_result.fitness_ == result.fitness_ and this_result.inlier_rmse_ < result.inlier_rmse_):
            result = this_result
            best_plane_model = plane_model
            if result.fitness_ < 1.0:
                break_iteration = min(np.log(1 - probability) / np.log(1 - pow(result.fitness_, ransac_n)), num_iterations)
            else:
                # Set break_iteration to 0 to force to break the loop.
                break_iteration = 0

        iteration_count += 1


    # Find the final inliers using best_plane_model.
    # std::vector<size_t> final_inliers;
    if not best_plane_model.isZero(0):
        for idx in range(self.points_.size()):
            point = np.array([self.points_[idx](0), self.points_[idx](1), self.points_[idx](2), 1])
            distance = abs(best_plane_model.dot(point))

            if distance < distance_threshold:
                final_inliers.emplace_back(idx)

    # Improve best_plane_model using the final inliers.
    best_plane_model = GetPlaneFromPoints(points_, final_inliers)

    print("RANSAC | Inliers: {:d}, Fitness: {:e}, RMSE: {:e}, Iteration: {:d}", final_inliers.size(), result.fitness_, result.inlier_rmse_, iteration_count)
    return best_plane_model, final_inliers