std::tuple<Eigen::Vector4d, std::vector<size_t>> PointCloud::SegmentPlane(
        const double distance_threshold /* = 0.01 */,
        const int ransac_n /* = 3 */,
        const int num_iterations /* = 100 */,
        const double probability /* = 0.99999999 */,
        utility::optional<int> seed /* = utility::nullopt */) const {
    if (probability <= 0 || probability > 1) {
        utility::LogError("Probability must be > 0 or <= 1.0");
    }

    RANSACResult result;

    // Initialize the best plane model.
    Eigen::Vector4d best_plane_model = Eigen::Vector4d(0, 0, 0, 0);

    size_t num_points = points_.size();
    RandomSampler<size_t> sampler(num_points, seed);

    // Return if ransac_n is less than the required plane model parameters.
    if (ransac_n < 3) {
        utility::LogError(
                "ransac_n should be set to higher than or equal to 3.");
        return std::make_tuple(Eigen::Vector4d(0, 0, 0, 0),
                               std::vector<size_t>{});
    }
    if (num_points < size_t(ransac_n)) {
        utility::LogError("There must be at least 'ransac_n' points.");
        return std::make_tuple(Eigen::Vector4d(0, 0, 0, 0),
                               std::vector<size_t>{});
    }

    // Use size_t here to avoid large integer which acceed max of int.
    size_t break_iteration = std::numeric_limits<size_t>::max();
    int iteration_count = 0;

#pragma omp parallel for schedule(static)
    for (int itr = 0; itr < num_iterations; itr++) {
        if ((size_t)iteration_count > break_iteration) {
            continue;
        }

        const std::vector<size_t> sampled_indices = sampler(ransac_n);
        std::vector<size_t> inliers = sampled_indices;

        // Fit model to num_model_parameters randomly selected points among the
        // inliers.
        Eigen::Vector4d plane_model;
        if (ransac_n == 3) {
            plane_model = TriangleMesh::ComputeTrianglePlane(
                    points_[inliers[0]], points_[inliers[1]],
                    points_[inliers[2]]);
        } else {
            plane_model = GetPlaneFromPoints(points_, inliers);
        }

        if (plane_model.isZero(0)) {
            continue;
        }

        double error = 0;
        inliers.clear();
        auto this_result = EvaluateRANSACBasedOnDistance(
                points_, plane_model, inliers, distance_threshold, error);
#pragma omp critical
        {
            if (this_result.fitness_ > result.fitness_ ||
                (this_result.fitness_ == result.fitness_ &&
                 this_result.inlier_rmse_ < result.inlier_rmse_)) {
                result = this_result;
                best_plane_model = plane_model;
                if (result.fitness_ < 1.0) {
                    break_iteration = std::min(
                            log(1 - probability) /
                                    log(1 - pow(result.fitness_, ransac_n)),
                            (double)num_iterations);
                } else {
                    // Set break_iteration to 0 to force to break the loop.
                    break_iteration = 0;
                }
            }
            iteration_count++;
        }
    }

    // Find the final inliers using best_plane_model.
    std::vector<size_t> final_inliers;
    if (!best_plane_model.isZero(0)) {
        for (size_t idx = 0; idx < points_.size(); ++idx) {
            Eigen::Vector4d point(points_[idx](0), points_[idx](1),
                                  points_[idx](2), 1);
            double distance = std::abs(best_plane_model.dot(point));

            if (distance < distance_threshold) {
                final_inliers.emplace_back(idx);
            }
        }
    }

    // Improve best_plane_model using the final inliers.
    best_plane_model = GetPlaneFromPoints(points_, final_inliers);

    utility::LogDebug(
            "RANSAC | Inliers: {:d}, Fitness: {:e}, RMSE: {:e}, Iteration: "
            "{:d}",
            final_inliers.size(), result.fitness_, result.inlier_rmse_,
            iteration_count);
    return std::make_tuple(best_plane_model, final_inliers);
}