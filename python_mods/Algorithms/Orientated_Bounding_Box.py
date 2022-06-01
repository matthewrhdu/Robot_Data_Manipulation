## \class OrientedBoundingBox
##
## \brief A bounding box oriented along an arbitrary frame of reference.
##
## The oriented bounding box is defined by its center position, rotation
## maxtrix and extent.
from __future__ import annotations
import numpy as np
from Point_cloud import PointCloud
from typing import Tuple

class Qhull:
    def ComputeConvexHull(self, points: np.ndarray, joggle_inputs: bool) -> Tuple[TriangleMesh, np.ndarray]:
        convex_hull = TriangleMesh()
        pt_map = []

        qhull_points_data = []
        for pidx in range(len(points.size)):
            pt = points[pidx]
            qhull_points_data[pidx * 3 + 0] = pt(0);
            qhull_points_data[pidx * 3 + 1] = pt(1);
            qhull_points_data[pidx * 3 + 2] = pt(2);
        }

        orgQhull::PointCoordinates qhull_points(3, "");
        qhull_points.append(qhull_points_data);

        orgQhull::Qhull qhull;
        std::string options = "Qt";
        if (joggle_inputs) {
            options += " QJ";
        }
        qhull.runQhull(qhull_points.comment().c_str(), qhull_points.dimension(),
                       qhull_points.count(), qhull_points.coordinates(),
                       options.c_str());

        orgQhull::QhullFacetList facets = qhull.facetList();
        convex_hull->triangles_.resize(facets.count());
        std::unordered_map<int, int> vert_map;
        std::unordered_set<int> inserted_vertices;
        int tidx = 0;
        for (orgQhull::QhullFacetList::iterator it = facets.begin();
             it != facets.end(); ++it) {
            if (!(*it).isGood()) continue;

            orgQhull::QhullFacet f = *it;
            orgQhull::QhullVertexSet vSet = f.vertices();
            int triangle_subscript = 0;
            for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
                 vIt != vSet.end(); ++vIt) {
                orgQhull::QhullVertex v = *vIt;
                orgQhull::QhullPoint p = v.point();

                int vidx = p.id();
                convex_hull->triangles_[tidx](triangle_subscript) = vidx;
                triangle_subscript++;

                if (inserted_vertices.count(vidx) == 0) {
                    inserted_vertices.insert(vidx);
                    vert_map[vidx] = int(convex_hull->vertices_.size());
                    double* coords = p.coordinates();
                    convex_hull->vertices_.push_back(
                            Eigen::Vector3d(coords[0], coords[1], coords[2]));
                    pt_map.push_back(vidx);
                }
            }

            tidx++;
        }

        auto center = convex_hull->GetCenter();
        for (Eigen::Vector3i& triangle : convex_hull->triangles_) {
            triangle(0) = vert_map[triangle(0)];
            triangle(1) = vert_map[triangle(1)];
            triangle(2) = vert_map[triangle(2)];

            Eigen::Vector3d e1 = convex_hull->vertices_[triangle(1)] -
                                 convex_hull->vertices_[triangle(0)];
            Eigen::Vector3d e2 = convex_hull->vertices_[triangle(2)] -
                                 convex_hull->vertices_[triangle(0)];
            auto normal = e1.cross(e2);

            auto triangle_center = (1. / 3) * (convex_hull->vertices_[triangle(0)] +
                                               convex_hull->vertices_[triangle(1)] +
                                               convex_hull->vertices_[triangle(2)]);
            if (normal.dot(triangle_center - center) < 0) {
                std::swap(triangle(0), triangle(1));
            }
        }

        return std::make_tuple(convex_hull, pt_map);
    }


class OrientedBoundingBox(Geometry3D):
    ## \brief Default constructor.
    ##
    ## Creates an empty Oriented Bounding Box.
    def __init__(self):
        Geometry3D.__init__(self)
        self.center_ = np.array([0, 0, 0])
        self.R_ = ([[0, 0, 0] for _ in range(3)])
        self.extent_ = np.array([0, 0, 0])
        self.color_ = np.array([1, 1, 1])

    def get_center(self) -> np.ndarray:
        return self.center_

    def get_oriented_bounding_box(self):
        return self

    def get_box_points(self) -> np.ndarray:
        x_axis = self.R_ * np.array([self.extent_[0] / 2, 0, 0])
        y_axis = self.R_ * np.array([0, self.extent_[1] / 2, 0])
        z_axis = self.R_ * np.array([0, 0, self.extent_[2] / 2])
        points = [self.center_ - x_axis - y_axis - z_axis, self.center_ + x_axis - y_axis - z_axis,
                  self.center_ - x_axis + y_axis - z_axis, self.center_ - x_axis - y_axis + z_axis,
                  self.center_ + x_axis + y_axis + z_axis, self.center_ - x_axis + y_axis + z_axis,
                  self.center_ + x_axis - y_axis + z_axis, self.center_ + x_axis + y_axis - z_axis]
        return np.array(points)

    ## Return indices to points that are within the bounding box.

    ## Creates an oriented bounding box using a PCA.
    ## Note, that this is only an approximation to the minimum oriented
    ## bounding box that could be computed for example with O'Rourke's
    ## algorithm (cf. http://cs.smith.edu/~jorourke/Papers/MinVolBox.pdf,
    ## https://www.geometrictools.com/Documentation/MinimumVolumeBox.pdf)
    ## \param points The input points
    ## \param robust If set to true uses a more robust method which works
    ##               in degenerate cases but introduces noise to the points
    ##               coordinates.
    def create_from_points(self, points: np.ndarray, robust: bool = False) -> OrientedBoundingBox:
        hull_pcd = PointCloud()
        # mesh = TriangluarMesh() # Shared Pointer, whatever that means
        qhull = Qhull()
        mesh, hull_point_indices = qhull.ComputeConvexHull(points, robust)
        hull_pcd.points_ = mesh.vertices_

        mean, cov = hull_pcd.ComputeMeanAndCovariance()

        es = SelfAdjointEigenSolver(cov)
        evals = es.eigenvalues()
        R = es.eigenvectors()

        if evals[1] > evals[0]:
            evals[1], evals[0] = evals[0], evals[1]
            R.col[0], R.col[1] = R.col[1], R.col[0]

        if evals[2] > evals[0]:
            evals[2], evals[0] = evals[0], evals[2]
            R.col[0], R.col[2] = R.col[2], R.col[0]

        if evals[2] > evals[1]
            evals[2], evals[1] = evals[1], evals[2]
            R.col[1], R.col[2] = R.col[2], R.col[1]

        R.col[0] /= R.col[0].norm()
        R.col[1] /= R.col[1].norm()
        R.col[2] = R.col[0].cross(R.col(1))

        for i in range(len(hull_point_indices)):
            hull_pcd.points_[i] = R.transpose() * (points[hull_point_indices[i]] - mean)

        aabox = hull_pcd.GetAxisAlignedBoundingBox()

        obox = OrientedBoundingBox()
        obox.center_ = R * aabox.GetCenter() + mean
        obox.R_ = R
        obox.extent_ = aabox.GetExtent()

        return obox
